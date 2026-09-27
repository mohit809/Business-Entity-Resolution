# Business Entity Resolution Pipeline — Round 1 Methodology

## 1. Project Title
**Business Entity Resolution Pipeline**

## 2. Executive Summary
This project implements an offline entity-resolution system that maps noisy business records from Source 2 and Source 3 to the deduplicated reference entities in Source 1. The design separates candidate generation from final matching so that exhaustive all-to-all comparison is avoided.

The pipeline uses multi-stage blocking, lexical similarity features, a supervised binary classifier, and precision-oriented threshold calibration against the macro F0.5 objective. Every Source 1 test record receives exactly one output row, including true or predicted singletons.

## 3. Problem Understanding
Source 1 is the reference population. A Source 1 entity can have zero, one, or multiple matches in Source 2 and Source 3. Business names and addresses are noisy because of abbreviations, punctuation, word-order changes, typos, transliteration, missing address components, and legal suffix variations.

The key engineering requirement is scalability: the system must not compare every Source 1 record with every Source 2/3 record.

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

## 5. Candidate Generation / Blocking

Four complementary blocks are used:

1. Exact normalized business name.
2. Country + informative business-name token.
3. Informative address-token overlap.
4. Character n-gram TF-IDF nearest-neighbor retrieval on business names.

The union is deduplicated. A final compactness cap retains the highest cheap lexical-similarity candidates when the union is unusually large.

This design increases blocking recall through multiple independent signals while keeping the candidate set small enough for pairwise ML inference.

## 6. Feature Engineering

### Name features
- normalized-name edit similarity
- WRatio
- token Jaccard similarity
- containment
- token overlap
- name length difference

### Address features
- normalized-address edit similarity
- token Jaccard
- containment
- token overlap
- address length difference
- postal-code-like overlap when available

### Context
- exact country agreement

No external address database, geocoder, registration database, web search, or business API is used.

## 7. Matching Model

The final pair classifier is XGBoost with:
- binary logistic objective
- 350 estimators
- max depth 5
- learning rate 0.05
- row subsampling 0.85
- feature subsampling 0.90
- L2 regularization

XGBoost is distributed under Apache-2.0. The solution uses a trained tabular classifier rather than a large pretrained language model, so it is well within the 8-billion-parameter constraint.

## 8. Training Strategy

Ground-truth matches create positive training pairs. Candidate-generation output supplies hard negatives: plausible records that passed blocking but are not in the ground truth.

A grouped validation split is performed by Source 1 entity ID so that candidate records associated with the same Source 1 entity cannot leak across training and validation partitions.

## 9. Threshold Selection

Because the challenge uses macro F0.5 and gives greater weight to precision, the decision threshold is tuned on the validation set over a grid of candidate thresholds.

For each threshold:
- score every validation candidate;
- convert probabilities into match/non-match decisions;
- calculate F0.5 separately for each Source 1 entity;
- treat correct singleton predictions as 1.0;
- average over Source 1 entities.

The selected threshold is then used after refitting the classifier on all generated training pairs.

## 10. Why Candidate Pairs Matter

`candidate_pairs.tsv` is generated after the final blocking/pruning stage and immediately before ML inference. Therefore it represents the exact candidate set that the model scores.

Every predicted match is generated from this same candidate set, guaranteeing:

`matching_results ⊆ candidate_pairs`

This also makes blocking recall auditable independently of final model precision.

## 11. Open-Set Country Handling

Country is normalized as a string and used only as an observed feature/blocking key. The pipeline does not hard-code `{US, India}` and therefore does not reject or filter a third test country such as France.

## 12. Singleton Handling

Every Source 1 test ID is written to `matching_results.tsv`. If no candidate passes the final decision threshold, the `matched_entity_ids` field is empty.

This is important because singleton entities are explicitly scored by the challenge.

## 13. Output Compliance

The pipeline produces:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Both are tab-separated. IDs are deduplicated. Matching IDs can only come from Source 2 or Source 3 test records. Source 1 self-matches are never generated.

## 14. Reproducibility

The submission package contains:
- source code
- pinned dependency versions
- README with commands
- methodology document
- final output files

The pipeline uses fixed random seeds for model training and validation splitting.

## 15. Validation Plan

Before submission, run:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Additionally record:
- number of Source 1 entities
- total candidate pairs
- average candidates per Source 1
- percentage of Source 1 entities with zero candidates
- validation macro F0.5
- selected threshold
- training positive/negative counts

## 16. Limitations and Future Improvements

Potential improvements that remain fully offline include:
- additional phonetic/name transliteration normalization learned from the supplied data;
- source-specific abbreviation dictionaries learned only from training pairs;
- adaptive candidate caps based on candidate-density statistics;
- a second-stage calibrated classifier;
- learned field reliability weights by country/source;
- character-level address retrieval in addition to name retrieval.

Any future improvement must preserve the no-external-data requirement.

## 17. Fair-Play Statement

The implementation is designed to use only the challenge-provided data. It does not perform external business identity lookup, web enrichment, geocoding, registration lookup, or external data augmentation.
