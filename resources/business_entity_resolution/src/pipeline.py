"""
Business Entity Resolution Pipeline
Offline-only: uses only the supplied train/test TSV files.

Stages:
1. Load/normalize records.
2. Learn blocking indexes from Source 2 + Source 3.
3. Generate a compact union of candidates for every Source 1 record.
4. Build pairwise features.
5. Train an XGBoost binary classifier on training positives + hard negatives.
6. Calibrate a precision-heavy decision threshold on a validation split.
7. Score test candidates and write matching_results.tsv + candidate_pairs.tsv.
"""

from __future__ import annotations
import argparse
import math
import re
from collections import defaultdict, Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, WRatio
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier


NAME_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "pvt", "private", "plc",
    "gmbh", "sarl", "sas"
}

ADDRESS_STOP = {"road", "rd", "street", "st", "near", "opposite", "opp",
                 "behind", "beside", "floor", "building", "block", "plot"}


def norm_text(x) -> str:
    x = "" if pd.isna(x) else str(x).lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def norm_name(x) -> str:
    toks = [t for t in norm_text(x).split() if t not in NAME_SUFFIXES]
    return " ".join(toks)


def norm_address(x) -> str:
    s = norm_text(x)
    replacements = {
        " rd ": " road ", " st ": " street ", " ave ": " avenue ",
        " av ": " avenue ", " blvd ": " boulevard ", " hwy ": " highway ",
        " pvt ": " private ", " ltd ": " limited "
    }
    s = f" {s} "
    for a, b in replacements.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s):
    return set(s.split()) if s else set()


def jaccard(a, b):
    A, B = tokens(a), tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def char_ngrams(s, n=3):
    if not s:
        return set()
    s = re.sub(r"\s+", " ", s)
    if len(s) <= n:
        return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}


def containment(a, b):
    if not a or not b:
        return 0.0
    return max(
        len(a) / max(len(b), 1) if a in b else 0.0,
        len(b) / max(len(a), 1) if b in a else 0.0
    )


def prepare(df, source):
    x = df.copy()
    x["source"] = source
    for c in ["business_name", "business_address", "country"]:
        if c not in x:
            x[c] = ""
    x["name_n"] = x["business_name"].map(norm_name)
    x["addr_n"] = x["business_address"].map(norm_address)
    x["country_n"] = x["country"].map(norm_text)
    x["name_tokens"] = x["name_n"].map(tokens)
    x["addr_tokens"] = x["addr_n"].map(tokens)
    return x


class Blocker:
    """Multi-block candidate generator. Candidate count is bounded per block."""

    def __init__(self, max_name=8, max_addr=8, max_char=10, max_total=30):
        self.max_name = max_name
        self.max_addr = max_addr
        self.max_char = max_char
        self.max_total = max_total
        self.rows = None
        self.name_index = defaultdict(list)
        self.addr_index = defaultdict(list)
        self.country_name_index = defaultdict(list)
        self.vectorizer = None
        self.nn = None
        self.matrix = None

    def fit(self, target):
        self.rows = target.reset_index(drop=True)
        for i, r in self.rows.iterrows():
            if r["name_n"]:
                self.name_index[r["name_n"]].append(i)
                for t in sorted(r["name_tokens"], key=lambda z: (len(z), z))[:3]:
                    if len(t) >= 4:
                        self.country_name_index[(r["country_n"], t)].append(i)
            for t in r["addr_tokens"]:
                if len(t) >= 5 and t not in ADDRESS_STOP:
                    self.addr_index[t].append(i)

        # Character TF-IDF index is fitted only on target records.
        corpus = self.rows["name_n"].fillna("")
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          min_df=1, max_features=120000)
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute",
                                   n_neighbors=min(self.max_char, len(self.rows)))
        self.nn.fit(self.matrix)
        return self

    def generate_one(self, q):
        ids = set()

        # Block A: exact normalized name.
        ids.update(self.name_index.get(q["name_n"], [])[:self.max_name])

        # Block B: country + rare-ish name tokens.
        for t in sorted(q["name_tokens"], key=lambda z: (len(self.country_name_index.get((q["country_n"], z), [])), -len(z)))[:3]:
            if len(t) >= 4:
                ids.update(self.country_name_index.get((q["country_n"], t), [])[:self.max_name])

        # Block C: informative address token overlap.
        addr_hits = Counter()
        for t in q["addr_tokens"]:
            if len(t) >= 5 and t not in ADDRESS_STOP:
                for i in self.addr_index.get(t, [])[:self.max_addr]:
                    addr_hits[i] += 1
        ids.update(i for i, _ in addr_hits.most_common(self.max_addr))

        # Block D: character similarity retrieval on name.
        if self.nn is not None and q["name_n"]:
            v = self.vectorizer.transform([q["name_n"]])
            _, nn_ids = self.nn.kneighbors(v, n_neighbors=min(self.max_char, len(self.rows)))
            ids.update(int(i) for i in nn_ids[0])

        # Final compactness rule: if union is huge, rank by cheap lexical signal.
        if len(ids) > self.max_total:
            scored = []
            for i in ids:
                r = self.rows.iloc[i]
                s = 0.65 * ratio(q["name_n"], r["name_n"]) + 0.35 * ratio(q["addr_n"], r["addr_n"])
                scored.append((s, i))
            ids = {i for _, i in sorted(scored, reverse=True)[:self.max_total]}
        return sorted(ids)

    def generate(self, queries):
        rows = []
        for _, q in queries.iterrows():
            for i in self.generate_one(q):
                rows.append((q["entity_id"], self.rows.iloc[i]["entity_id"]))
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def pair_features(a, b):
    an, bn = a["name_n"], b["name_n"]
    aa, ba = a["addr_n"], b["addr_n"]
    ac, bc = a["country_n"], b["country_n"]
    nt_a, nt_b = a["name_tokens"], b["name_tokens"]
    at_a, at_b = a["addr_tokens"], b["addr_tokens"]

    common_name = len(nt_a & nt_b)
    common_addr = len(at_a & at_b)
    return {
        "country_exact": float(bool(ac) and ac == bc),
        "name_ratio": ratio(an, bn) / 100.0,
        "name_wratio": WRatio(an, bn) / 100.0,
        "name_jaccard": jaccard(an, bn),
        "name_containment": containment(an, bn),
        "name_len_diff": abs(len(an) - len(bn)),
        "name_token_overlap": common_name / max(1, min(len(nt_a), len(nt_b))),
        "addr_ratio": ratio(aa, ba) / 100.0,
        "addr_jaccard": jaccard(aa, ba),
        "addr_containment": containment(aa, ba),
        "addr_len_diff": abs(len(aa) - len(ba)),
        "addr_token_overlap": common_addr / max(1, min(len(at_a), len(at_b))),
        "same_postal_like": float(bool(re.search(r"\b\d{5,6}\b", aa)) and
                                  bool(re.search(r"\b\d{5,6}\b", ba)) and
                                  set(re.findall(r"\b\d{5,6}\b", aa)) &
                                  set(re.findall(r"\b\d{5,6}\b", ba))),
    }


def make_feature_table(s1, targets, pairs):
    idx = targets.set_index("entity_id").to_dict("index")
    qidx = s1.set_index("entity_id").to_dict("index")
    out = []
    for _, p in pairs.iterrows():
        a, b = qidx[p.source1_entity_id], idx[p.candidate_entity_id]
        f = pair_features(a, b)
        f["source1_entity_id"] = p.source1_entity_id
        f["candidate_entity_id"] = p.candidate_entity_id
        out.append(f)
    return pd.DataFrame(out)


def labels_from_gt(pairs, gt):
    truth = defaultdict(set)
    for _, r in gt.iterrows():
        ids = str(r["matched_entity_ids"]) if not pd.isna(r["matched_entity_ids"]) else ""
        truth[r["source1_entity_id"]] = {z for z in ids.split(",") if z}
    return np.array([
        int(p.candidate_entity_id in truth.get(p.source1_entity_id, set()))
        for _, p in pairs.iterrows()
    ], dtype=np.int8)


def macro_f05(y_true, y_prob, groups, threshold):
    pred = (y_prob >= threshold).astype(int)
    scores = []
    for g in pd.Series(groups).unique():
        m = np.asarray(groups) == g
        yt, yp = y_true[m], pred[m]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(1.0)
        else:
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            scores.append((1.25*p*r)/(0.25*p+r) if p+r else 0.0)
    return float(np.mean(scores))


def build_training_pairs(s1, targets, gt, blocker):
    cand = blocker.generate(s1)
    y = labels_from_gt(cand, gt)
    # Hard-negative set = all generated negatives; cap per S1 to keep training balanced.
    tmp = cand.copy()
    tmp["y"] = y
    pos = tmp[tmp.y == 1]
    neg = tmp[tmp.y == 0]
    neg = neg.groupby("source1_entity_id", group_keys=False).head(12)
    return pd.concat([pos, neg], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--model-dir", default="artifacts")
    args = ap.parse_args()

    train = Path(args.train_dir)
    test = Path(args.test_dir)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.model_dir); mdir.mkdir(parents=True, exist_ok=True)

    s1 = prepare(pd.read_csv(train/"train_source1.tsv", sep="\t"), "S1")
    s2 = prepare(pd.read_csv(train/"train_source2.tsv", sep="\t"), "S2")
    s3 = prepare(pd.read_csv(train/"train_source3.tsv", sep="\t"), "S3")
    gt = pd.read_csv(train/"train_ground_truth.tsv", sep="\t")

    targets = pd.concat([s2, s3], ignore_index=True)
    blocker = Blocker().fit(targets)
    train_pairs = build_training_pairs(s1, targets, gt, blocker)
    X_all = make_feature_table(s1, targets, train_pairs)
    y = train_pairs["y"].to_numpy()

    feature_cols = [c for c in X_all.columns if c not in
                    ["source1_entity_id", "candidate_entity_id"]]

    # Group split by Source 1 prevents records from the same entity leaking across folds.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, va = next(gss.split(X_all, y, groups=train_pairs["source1_entity_id"]))
    model = XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss",
        reg_lambda=2.0, min_child_weight=2, random_state=42,
        n_jobs=4
    )
    model.fit(X_all.iloc[tr][feature_cols], y[tr])

    val_prob = model.predict_proba(X_all.iloc[va][feature_cols])[:, 1]
    thresholds = np.arange(0.20, 0.951, 0.01)
    val_groups = train_pairs.iloc[va]["source1_entity_id"].to_numpy()
    scores = [(t, macro_f05(y[va], val_prob, val_groups, t)) for t in thresholds]
    threshold, best_score = max(scores, key=lambda z: z[1])

    # Refit on all generated training pairs after threshold selection.
    model.fit(X_all[feature_cols], y)
    joblib.dump({"model": model, "features": feature_cols,
                 "threshold": float(threshold)}, mdir/"model.joblib")

    # Test data.
    ts1 = prepare(pd.read_csv(test/"test_source1.tsv", sep="\t"), "S1")
    ts2 = prepare(pd.read_csv(test/"test_source2.tsv", sep="\t"), "S2")
    ts3 = prepare(pd.read_csv(test/"test_source3.tsv", sep="\t"), "S3")
    tt = pd.concat([ts2, ts3], ignore_index=True)

    test_blocker = Blocker().fit(tt)
    candidate = test_blocker.generate(ts1)
    candidate.to_csv(outdir/"candidate_pairs.tsv", sep="\t", index=False)

    # Candidate_pairs is intentionally the exact final inference set.
    if len(candidate):
        TX = make_feature_table(ts1, tt, candidate)
        probs = model.predict_proba(TX[feature_cols])[:, 1]
        candidate["probability"] = probs
        candidate["keep"] = probs >= threshold
    else:
        candidate["probability"] = []
        candidate["keep"] = []

    matches = defaultdict(list)
    for _, r in candidate[candidate["keep"]].iterrows():
        matches[r.source1_entity_id].append(r.candidate_entity_id)

    result = pd.DataFrame({
        "source1_entity_id": ts1["entity_id"],
        "matched_entity_ids": [
            ",".join(dict.fromkeys(matches.get(eid, [])))
            for eid in ts1["entity_id"]
        ]
    })
    result.to_csv(outdir/"matching_results.tsv", sep="\t", index=False)

    print(f"Validation macro F0.5: {best_score:.6f}")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Test S1 entities: {len(ts1)}")
    print(f"Candidate pairs: {len(candidate)}")
    print(f"Average candidates/S1: {len(candidate)/max(1,len(ts1)):.2f}")


if __name__ == "__main__":
    main()
