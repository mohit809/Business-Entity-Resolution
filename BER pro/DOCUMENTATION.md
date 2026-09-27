# Business Entity Resolution Pipeline — Methodology & Engineering Report

## 1. Project Title
**High-Throughput Business Entity Resolution with Multi-Stage Blocking and Calibrated Gradient Boosting**

---

## 2. Executive Summary
This project implements a production-grade, offline entity resolution system that maps noisy business records from Source 2 and Source 3 to deduplicated reference entities in Source 1. The architecture strictly isolates candidate generation from pairwise classification to eliminate exhaustive $O(N \times M)$ comparisons across millions of records.

The pipeline incorporates a 4-pass inverted index blocking engine, 13 lexical and contextual pairwise features, a regularized XGBoost binary classifier, and an entity-level Macro $F_{0.5}$ decision threshold optimizer. The system enforces strict format compliance: every Source 1 test record receives exactly one output row in both `matching_results.tsv` and `candidate_pairs.tsv`, guaranteeing $matching\_results \subseteq candidate\_pairs$ and achieving full auditability and fair-play compliance.

---

## 3. Problem Understanding
Source 1 represents the reference canonical population. A Source 1 entity may have zero, one, or multiple matches across Source 2 and Source 3. Real-world business entity linkage is made challenging by:
- **Lexical noise**: Abbreviations (`St` vs `Street`, `Pvt` vs `Private`), legal entity variations (`Inc`, `Corp`, `LLC`), punctuation discrepancies, and typographic errors.
- **Structural permutations**: Inverted word order, partial addresses, and missing postal codes.
- **Scale constraints**: With $>2.2\text{M}$ training entities and $>1.7\text{M}$ test entities, an all-to-all comparison would require $>7 \times 10^{12}$ comparisons, exceeding time and memory boundaries.
- **Metric asymmetry**: The objective is Macro $F_{0.5}$, which places double the penalty on false positives relative to false negatives, requiring precision-driven calibration.

---

## 4. Architecture

```text
                         ┌─────────────────────────────┐
                         │        Input TSV Files       │
                         │ S1 / S2 / S3 / Ground Truth │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Schema + Data Validation   │
                         │  IDs, columns, tab separator │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Text Normalization Layer   │
                         │ name / address / country     │
                         └──────────────┬──────────────┘
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                    ▼                    ▼
            Exact-name block     Country+token block    Address-token block
                   │                    │                    │
                   └────────────────────┬────────────────────┘
                                        ▼
                               Character n-gram retrieval
                                        │
                                        ▼
                           Candidate union + cap/pruning
                                        │
                                        ▼
                              candidate_pairs.tsv
                         (exact final ML inference set)
                                        │
                                        ▼
                              Pair feature engineering
                                        │
                                        ▼
                             XGBoost binary classifier
                                        │
                                        ▼
                           Precision-aware threshold
                                        │
                                        ▼
                             Final match predictions
                                        │
                    ┌───────────────────┴──────────────────┐
                    ▼                                      ▼
             matching_results.tsv                  Submission validator
```

---

## 5. Candidate Generation / Blocking

To maximize candidate recall while constraining inference workload, four complementary blocking mechanisms are unified:

1. **Block A — Exact Normalized Business Name**:
   - Matches records on stripped, lowercased, legal-suffix-free business names.
   - Bounded to `max_name = 8` candidates.
2. **Block B — Country + Informative Business Name Token**:
   - Indexes target entities by `(country_n, token)` where token length $\ge 4$.
   - Query tokens are sorted dynamically by posting list frequency; rare, highly discriminative tokens are probed first.
   - Bounded to `max_name = 8` candidates.
3. **Block C — Informative Address Token Overlap**:
   - Indexes non-stopword address tokens (length $\ge 5$).
   - Scores candidate entities by address token co-occurrence frequency.
   - Bounded to `max_addr = 8` candidates.
4. **Block D — Character n-gram TF-IDF Retrieval**:
   - Computes character 3-to-5-gram TF-IDF representations over business names.
   - Fast cosine nearest-neighbors retrieval discovers phonetic, spelling, and OCR corruptions.
   - Bounded to `max_char = 10` candidates.

### Compactness Pruning
The union of candidate indices from Blocks A–D is deduplicated. When the union size exceeds `max_total = 30`, a fast composite lexical ranker:
$$\text{Score} = 0.65 \times \text{ratio}(\text{name}_a, \text{name}_b) + 0.35 \times \text{ratio}(\text{addr}_a, \text{addr}_b)$$
prunes the set to the top 30 most promising candidates before feature extraction.

---

## 6. Feature Engineering

The feature vector contains 13 dense, orthogonal attributes:

### Name Features
1. `name_ratio`: Normalized Levenshtein edit distance ratio scaled to $[0, 1]$.
2. `name_wratio`: Weighted ratio (`WRatio`) accounting for token reordering and partial substrings.
3. `name_jaccard`: Jaccard similarity between token sets: $\frac{|A \cap B|}{|A \cup B|}$.
4. `name_containment`: Maximum substring containment ratio: $\max\left(\frac{|A|}{|B|}, \frac{|B|}{|A|}\right)$ if contained, else $0.0$.
5. `name_len_diff`: Absolute character length difference.
6. `name_token_overlap`: Token intersection divided by minimum set size: $\frac{|A \cap B|}{\min(|A|, |B|)}$.

### Address Features
7. `addr_ratio`: Address Levenshtein edit distance ratio.
8. `addr_jaccard`: Address token set Jaccard similarity.
9. `addr_containment`: Address substring containment ratio.
10. `addr_len_diff`: Absolute address character length difference.
11. `addr_token_overlap`: Common address tokens relative to shorter address length.
12. `same_postal_like`: Binary indicator ($1.0$ or $0.0$) verifying presence of matching 5-to-6 digit postal codes.

### Context Feature
13. `country_exact`: Binary indicator ($1.0$ if normalized non-empty country strings match exactly, else $0.0$).

*Fair-Play Compliance*: All features are computed purely in-memory from the provided TSV records. No external geocoding, Google Maps API, postal registries, or web searches are invoked.

---

## 7. Matching Model

The pair classifier is an optimized **XGBoost (Extreme Gradient Boosting)** binary model:
- **Objective**: `binary:logistic`
- **Number of Estimators**: 350
- **Maximum Tree Depth**: 5
- **Learning Rate ($\eta$)**: 0.05
- **Subsampling Rate**: 0.85
- **Colsample By Tree**: 0.90
- **L2 Regularization ($\lambda$)**: 2.0
- **Minimum Child Weight**: 2
- **License**: Apache-2.0

*Parameter Budget*: Total model weights and tree structures require $< 50\text{ MB}$, well within the challenge parameter limit of 8 billion parameters.

---

## 8. Training Strategy

1. **Positive Pairs**: Sourced directly from `train_ground_truth.tsv`.
2. **Hard Negatives**: Derived from candidate records generated by the multi-blocker that do not appear in ground truth. Hard negatives are balanced by capping at 12 negatives per Source 1 entity.
3. **Leakage-Free Grouped Validation**:
   - `GroupShuffleSplit` partitioned on `source1_entity_id` ($80\%$ train, $20\%$ validation).
   - Guarantees that candidates linked to the same reference entity never appear in both training and validation sets.

---

## 9. Threshold Selection

The official metric is entity-level Macro $F_{0.5}$:
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

Threshold optimization procedure:
1. Generate out-of-fold match probabilities on the validation partition.
2. Grid search threshold $\tau \in [0.20, 0.95]$ with step $0.01$.
3. For each candidate threshold $\tau$:
   - Convert continuous probabilities into binary matches: $\hat{y} = \mathbb{I}(p \ge \tau)$.
   - Compute $F_{0.5}$ for each individual Source 1 entity.
   - Singleton handling: When true matches is empty and predicted matches is empty, score is explicitly set to $1.0$.
   - Calculate the mean across all validation Source 1 entities.
4. Select threshold $\tau^*$ yielding peak Macro $F_{0.5}$.
5. Refit XGBoost on 100% of generated training pairs using calibrated threshold $\tau^*$.

---

## 10. Why Candidate Pairs Matter

The file `candidate_pairs.tsv` represents the exact candidate set evaluated by the machine learning model.
- It is saved immediately prior to ML inference.
- Every predicted match in `matching_results.tsv` is drawn from this set:
  $$\text{matching\_results} \subseteq \text{candidate\_pairs}$$
- This decoupling allows independent auditing of blocking recall versus classifier precision.

---

## 11. Open-Set Country Handling

The pipeline normalizes country labels as clean, case-insensitive string tokens (`norm_country()`) and tests for exact string equality. It does **not** hardcode a whitelist (e.g. `{"US", "India"}`). If the test dataset includes entities from `France`, `Germany`, `Canada`, or `Japan`, they are parsed and compared seamlessly without failure or data loss.

---

## 12. Singleton Handling

In real-world entity resolution, many reference records have no counterpart in external feeds.
- Every Source 1 test entity is explicitly emitted in `matching_results.tsv`.
- If no candidate exceeds the calibrated decision threshold, `matched_entity_ids` is written as an empty string.
- This ensures correct evaluation by the competition scorer, which explicitly scores singleton entities.

---

## 13. Output Compliance

The pipeline generates:
1. `output/matching_results.tsv`:
   - Header: `source1_entity_id\tmatched_entity_ids`
2. `output/candidate_pairs.tsv`:
   - Header: `source1_entity_id\tcandidate_entity_ids`

Both files strictly adhere to:
- UTF-8 text encoding.
- Tab (`\t`) delimiter (no CSV commas).
- Exactly one row per Source 1 entity.
- Comma-separated target IDs prefixed with `S2-` or `S3-`.
- Zero `S1-` self-matches.
- Zero intra-list duplicate IDs.

---

## 14. Reproducibility

The repository guarantees end-to-end reproducibility:
- Deterministic pseudo-random seeds (`random_state=42`) across data splits and model training.
- Pinned dependency versions in `requirements.txt` and `pyproject.toml`.
- Automated test suite covering normalization, blocking, features, metrics, and compliance.
- Direct execution script `scripts/run_pipeline.ps1` and `scripts/run_pipeline.bat`.

---

## 15. Validation Plan

Before final submission packaging, the system executes two levels of validation:
1. Internal validation via `utils.validator.validate_outputs`.
2. Official validation via `utils/validate_submission.py`:
   ```bash
   python utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir dataset/test \
     --check-ids
   ```
Diagnostics recorded:
- Total Source 1 test records verified
- Total candidate pairs generated
- Average candidate density per entity
- Percentage of singleton predictions
- Validation Macro $F_{0.5}$ score and optimal threshold

---

## 16. Limitations and Future Improvements

Fully offline improvements preserving fair-play include:
1. **Phonetic Encoding**: Double Metaphone or Soundex keys tailored to multiregional business names.
2. **Frequency-Calibrated Term Weighting**: Domain-specific inverse document frequency (IDF) weights learned strictly from the challenge corpus.
3. **Adaptive Thresholding**: Dynamically modulating thresholds based on candidate density or field completeness.
4. **Sub-token Character Alignments**: Fast affine-gap local alignments on addresses.

---

## 17. Fair-Play Statement

The solution complies strictly with all competition integrity and fair-play regulations:
- Operates 100% offline using only the provided challenge TSV files.
- Makes zero network calls, API requests, web queries, or external database queries.
- Utilizes open-source, permissive Apache-2.0 components (XGBoost, Scikit-learn, RapidFuzz, Pandas).
- Operates well below the 8B parameter model capacity constraint.
