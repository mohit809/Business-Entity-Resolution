# Architecture Blueprint

## Runtime flow

```text
TRAIN
  train_source1.tsv ─┐
  train_source2.tsv ─┼─> normalize ─> blocking indexes ─> training candidates
  train_source3.tsv ─┘                                      │
  train_ground_truth.tsv ───────────────────────────────────┤
                                                           ▼
                                                pair features + labels
                                                           ▼
                                                grouped validation
                                                           ▼
                                               XGBoost classifier
                                                           ▼
                                                F0.5 threshold tuning
                                                           ▼
                                                final trained model

TEST
  test_source1.tsv ─┐
  test_source2.tsv ─┼─> normalize ─> multi-block candidate generation
  test_source3.tsv ─┘                         │
                                             ▼
                                   candidate_pairs.tsv
                                             │
                                             ▼
                                    feature extraction
                                             │
                                             ▼
                                      ML inference
                                             │
                                             ▼
                                  threshold + dedupe
                                             │
                                             ▼
                                  matching_results.tsv
```

## Design principles
- No O(N×M) all-pairs comparison.
- Candidate generation is independent from final classification.
- Multiple blocking signals reduce false negatives caused by noisy individual fields.
- Final candidate set is persisted for auditability.
- Precision is emphasized at the decision stage because the metric is F0.5.
- Open-set country labels are preserved.
- Every Source 1 test entity is represented in the final output.
