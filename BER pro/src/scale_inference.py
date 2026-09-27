"""
Ultra-High-Throughput Offline Submission Generator for Business Entity Resolution.

Processes all 1,732,544 test reference entities against 9,969,589 target records
using streaming inverted indexing, vectorized candidate pruning, and XGBoost ML scoring.
Outputs 100% compliant matching_results.tsv and candidate_pairs.tsv for competition submission.
"""

from __future__ import annotations
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, WRatio

from src.normalization import norm_name, norm_address, norm_country, tokenize, extract_postal_codes
from src.features import extract_pair_features
from src.model import EntityResolutionModel
from utils.validator import validate_outputs


def get_key_token(tokens: Set[str]) -> str:
    """Select the most distinctive token (longest non-generic word)."""
    if not tokens:
        return ""
    # Filter out very short tokens
    long_toks = [t for t in tokens if len(t) >= 4]
    return max(long_toks, key=len) if long_toks else max(tokens, key=len)


def build_target_index(test_dir: Path, max_target_records: int | None = None) -> Tuple[
    Dict[str, List[str]],                      # name_index: norm_name -> [eid]
    Dict[Tuple[str, str], List[str]],          # token_index: (country, token) -> [eid]
    Dict[Tuple[str, str], List[str]],          # postal_index: (country, postal) -> [eid]
    Dict[str, Tuple[str, str, str]]            # target_records: eid -> (norm_name, norm_addr, country)
]:
    """
    Build memory-compact inverted indices over all test target feeds (test_source2 and test_source3).
    """
    name_index: Dict[str, List[str]] = {}
    token_index: Dict[Tuple[str, str], List[str]] = {}
    postal_index: Dict[Tuple[str, str], List[str]] = {}
    target_records: Dict[str, Tuple[str, str, str]] = {}

    target_files = ["test_source2.tsv", "test_source3.tsv"]
    total_loaded = 0
    t0 = time.time()

    cap_str = f" (Cap: {max_target_records:,} records)" if max_target_records else " (All 9.97M Records)"
    print("================================================================================", flush=True)
    print(f"        BUILDING TARGET INVERTED INDICES{cap_str}              ", flush=True)
    print("================================================================================", flush=True)

    for fname in target_files:
        fpath = test_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing target file: {fpath}")

        print(f"Streaming {fname}...", flush=True)
        t_file = time.time()
        file_count = 0

        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 3:
                    continue

                eid = parts[0]
                raw_name = parts[1]
                raw_addr = parts[2]
                raw_country = parts[3] if len(parts) > 3 else ""

                n_name = norm_name(raw_name)
                n_addr = norm_address(raw_addr)
                n_country = norm_country(raw_country)

                # Store attributes
                target_records[eid] = (n_name, n_addr, n_country)

                # 1. Exact Name Index
                if n_name:
                    if n_name in name_index:
                        if len(name_index[n_name]) < 25:
                            name_index[n_name].append(eid)
                    else:
                        name_index[n_name] = [eid]

                # 2. Country + Key Token Index
                toks = tokenize(n_name)
                key_tok = get_key_token(toks)
                if key_tok and len(key_tok) >= 4:
                    k = (n_country, key_tok)
                    if k in token_index:
                        if len(token_index[k]) < 25:
                            token_index[k].append(eid)
                    else:
                        token_index[k] = [eid]

                # 3. Country + Postal Index
                pcs = extract_postal_codes(n_addr)
                if pcs:
                    postal = next(iter(pcs))
                    pk = (n_country, postal)
                    if pk in postal_index:
                        if len(postal_index[pk]) < 15:
                            postal_index[pk].append(eid)
                    else:
                        postal_index[pk] = [eid]

                file_count += 1
                total_loaded += 1

                if file_count % 1000000 == 0:
                    print(f"  Loaded {file_count:,} records from {fname} ({time.time()-t_file:.1f}s)...", flush=True)

                if max_target_records and total_loaded >= max_target_records:
                    break

        print(f"Finished {fname}: {file_count:,} records in {time.time()-t_file:.1f}s.", flush=True)
        if max_target_records and total_loaded >= max_target_records:
            break

    print(f"\nInverted Index Built Successfully in {time.time()-t0:.1f}s total.", flush=True)
    print(f"Total Targets: {len(target_records):,}", flush=True)
    print(f"Unique Names Indexed: {len(name_index):,}", flush=True)
    print(f"Token Index Buckets: {len(token_index):,}", flush=True)
    print(f"Postal Index Buckets: {len(postal_index):,}", flush=True)
    return name_index, token_index, postal_index, target_records


def generate_submission(
    test_dir: Path,
    output_dir: Path,
    model_dir: Path,
    max_records: int | None = None
) -> Tuple[Path, Path]:
    """
    Execute streaming inference and export fully compliant matching_results.tsv and candidate_pairs.tsv.
    """
    test_dir = Path(test_dir).resolve()
    output_dir = Path(output_dir).resolve()
    model_dir = Path(model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model artifact not found: {model_path}")

    print(f"Loading trained model from {model_path}...", flush=True)
    model = EntityResolutionModel.load(model_path)
    threshold = model.threshold
    print(f"Model calibrated threshold: {threshold:.4f}", flush=True)

    # Step 1: Build Inverted Target Index
    max_targets = (max_records * 20) if max_records else None
    name_index, token_index, postal_index, target_records = build_target_index(test_dir, max_target_records=max_targets)

    # Step 2: Stream Test Source 1 & Generate Predictions
    s1_path = test_dir / "test_source1.tsv"
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    print("\n================================================================================", flush=True)
    print("        STREAMING CANDIDATE GENERATION & INFERENCE ON TEST SOURCE 1            ", flush=True)
    print("================================================================================", flush=True)

    t_start = time.time()
    processed_count = 0
    match_count = 0
    singleton_count = 0

    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(matching_path, "w", encoding="utf-8", newline="\n") as f_match, \
         open(candidate_path, "w", encoding="utf-8", newline="\n") as f_cand:

        # Write TSV Headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        header = f_in.readline()  # skip header

        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue

            s1_id = parts[0]
            raw_name = parts[1] if len(parts) > 1 else ""
            raw_addr = parts[2] if len(parts) > 2 else ""
            raw_country = parts[3] if len(parts) > 3 else ""

            s1_name_n = norm_name(raw_name)
            s1_addr_n = norm_address(raw_addr)
            s1_country_n = norm_country(raw_country)
            s1_toks = tokenize(s1_name_n)
            s1_key_tok = get_key_token(s1_toks)
            s1_pcs = extract_postal_codes(s1_addr_n)
            s1_postal = next(iter(s1_pcs)) if s1_pcs else ""

            # Fast Multi-Block Retrieval
            candidate_ids_set: Set[str] = set()

            # Pass 1: Exact Name
            if s1_name_n and s1_name_n in name_index:
                candidate_ids_set.update(name_index[s1_name_n])

            # Pass 2: Country + Key Token
            if s1_key_tok and len(s1_key_tok) >= 4:
                k = (s1_country_n, s1_key_tok)
                if k in token_index:
                    candidate_ids_set.update(token_index[k])

            # Pass 3: Country + Postal Code (if candidate set is small)
            if s1_postal and len(candidate_ids_set) < 10:
                pk = (s1_country_n, s1_postal)
                if pk in postal_index:
                    candidate_ids_set.update(postal_index[pk])

            # Limit candidates per S1 entity
            cand_list = list(candidate_ids_set)[:30]

            # Score candidates
            matched_ids: List[str] = []

            if cand_list:
                s1_rec = {
                    "name_n": s1_name_n,
                    "addr_n": s1_addr_n,
                    "country_n": s1_country_n,
                    "name_tokens": s1_toks,
                    "addr_tokens": tokenize(s1_addr_n),
                    "postal_codes": s1_pcs
                }

                feat_rows = []
                eval_cand_ids = []

                for cid in cand_list:
                    c_data = target_records.get(cid)
                    if not c_data:
                        continue
                    c_name_n, c_addr_n, c_country_n = c_data

                    # Cheap lexical pre-screen to bypass computation for obvious non-matches
                    name_sim = ratio(s1_name_n, c_name_n) / 100.0
                    addr_sim = ratio(s1_addr_n, c_addr_n) / 100.0

                    if name_sim < 0.60 and addr_sim < 0.60:
                        continue

                    c_rec = {
                        "name_n": c_name_n,
                        "addr_n": c_addr_n,
                        "country_n": c_country_n,
                        "name_tokens": tokenize(c_name_n),
                        "addr_tokens": tokenize(c_addr_n),
                        "postal_codes": extract_postal_codes(c_addr_n)
                    }

                    feats = extract_pair_features(s1_rec, c_rec)
                    feat_rows.append(feats)
                    eval_cand_ids.append(cid)

                # Batch score with XGBoost
                if feat_rows:
                    feat_df = pd.DataFrame(feat_rows)
                    probs = model.predict_proba(feat_df)
                    for cid, p in zip(eval_cand_ids, probs):
                        if p >= threshold:
                            matched_ids.append(cid)

            # Deduplicate preserving order
            matched_ids = list(dict.fromkeys(matched_ids))
            cand_list = list(dict.fromkeys(cand_list))

            # Guarantee: all matched IDs are strictly in candidate_ids
            for m in matched_ids:
                if m not in candidate_ids_set:
                    cand_list.append(m)

            # Write outputs
            m_str = ",".join(matched_ids)
            c_str = ",".join(cand_list)

            f_match.write(f"{s1_id}\t{m_str}\n")
            f_cand.write(f"{s1_id}\t{c_str}\n")

            processed_count += 1
            if matched_ids:
                match_count += 1
            else:
                singleton_count += 1

            if processed_count % 100000 == 0:
                elapsed = time.time() - t_start
                rate = processed_count / max(1.0, elapsed)
                remaining = (1732544 - processed_count) / max(1.0, rate)
                print(f"Processed {processed_count:,} / 1,732,544 ({processed_count/17325.44:.1f}%) | "
                      f"Matches: {match_count:,} | Singletons: {singleton_count:,} | "
                      f"Speed: {rate:.0f} rec/s | ETA: {remaining/60:.1f}m", flush=True)

            if max_records and processed_count >= max_records:
                break

    total_time = time.time() - t_start
    print("\n================================================================================", flush=True)
    print("                      INFERENCE COMPLETE — FILES GENERATED                      ", flush=True)
    print("================================================================================", flush=True)
    print(f"Total S1 Processed:   {processed_count:,}", flush=True)
    print(f"Matched Entities:     {match_count:,} ({match_count/processed_count*100:.1f}%)", flush=True)
    print(f"Singleton Entities:   {singleton_count:,} ({singleton_count/processed_count*100:.1f}%)", flush=True)
    print(f"Total Execution Time: {total_time/60:.2f} minutes ({processed_count/total_time:.0f} rec/s)", flush=True)
    print(f"Matching Results:     {matching_path}", flush=True)
    print(f"Candidate Pairs:      {candidate_path}", flush=True)
    print("--------------------------------------------------------------------------------", flush=True)

    return matching_path, candidate_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate full hackathon submission files.")
    parser.add_argument("--test-dir", default="../resources/dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-dir", default="artifacts")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    t_dir = Path(args.test_dir)
    o_dir = Path(args.output_dir)
    m_dir = Path(args.model_dir)

    m_file, c_file = generate_submission(t_dir, o_dir, m_dir, max_records=args.max_records)
