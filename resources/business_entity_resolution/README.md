# Business Entity Resolution Pipeline

## Objective
Resolve Source 1 business records to matching Source 2/Source 3 records using an offline, scalable blocking + ML ranking pipeline.

## Fair-play compliance
- No web lookup, external database, geocoder, API, or external business data.
- Only the supplied TSV files are read.
- The learned model is XGBoost (Apache-2.0) and is far below the 8B parameter limit.
- Countries are treated as open-set strings; no US/India hard-coding.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt

python src/pipeline.py \
  --train-dir ../../dataset/train \
  --test-dir ../../dataset/test \
  --output-dir ../../output \
  --model-dir ../../artifacts
```

Then validate from `student_resource/`:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Pipeline
1. Normalize names, addresses and countries.
2. Build several blocking indexes over Source 2 + Source 3.
3. Generate a small candidate union per Source 1 record.
4. Generate training labels from `train_ground_truth.tsv`.
5. Train on positives plus hard negatives.
6. Use a grouped validation split and tune a decision threshold for macro F0.5.
7. Refit the model on all training pairs.
8. Generate test candidates.
9. Score exactly those candidates.
10. Write `candidate_pairs.tsv` and `matching_results.tsv`.

`candidate_pairs.tsv` is written immediately before ML inference, satisfying the challenge definition of the final candidate set.
