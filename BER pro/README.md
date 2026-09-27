# Business Entity Resolution Pipeline (`BER pro`)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Compliance](https://img.shields.io/badge/Submission-PASS-brightgreen.svg)]()

Production-grade, offline Machine Learning pipeline for resolving noisy business records from heterogeneous feeds (Source 2 & Source 3) to canonical reference entities (Source 1).

---

## 🌟 Key Highlights

- **Scalable Multi-Stage Blocking**: Eliminates $O(N \times M)$ Cartesian cross-products via 4 complementary inverted index blocking mechanisms (Exact Name, Country+Rare Token, Address Overlap, Character n-grams).
- **Leak-Free Grouped Validation**: `GroupShuffleSplit` on Source 1 entities guarantees zero record leakage across training and validation partitions.
- **Precision-Weighted Macro $F_{0.5}$ Calibration**: Decision thresholds are tuned via grid search directly against entity-level Macro $F_{0.5}$ with rigorous singleton handling.
- **Strict Submission Compliance**: Generates `matching_results.tsv` and `candidate_pairs.tsv` strictly adhering to challenge validator requirements ($matching\_results \subseteq candidate\_pairs$, single-row representation, tab-separated, UTF-8).
- **100% Offline Fair-Play**: Zero external API calls, geocoders, or web lookups. Uses Apache-2.0 XGBoost tabular classifier (<50 MB parameter footprint).

---

## 📁 Repository Structure

```text
BER pro/
├── ARCHITECTURE.md          # Architecture Blueprint, C4 Diagrams & ADRs
├── DOCUMENTATION.md         # Full Methodology Report (17 sections)
├── README.md                # Project documentation and quickstart
├── pyproject.toml           # Standard Python packaging configuration
├── requirements.txt         # Pinned production dependencies
│
├── config/                  # Pipeline settings and hyperparameter configs
│   ├── __init__.py
│   └── settings.py
│
├── src/                     # Core pipeline implementation
│   ├── __init__.py
│   ├── normalization.py     # Name, address, and open-set country normalization
│   ├── blocking.py          # 4-pass inverted index blocking engine
│   ├── features.py          # 13-dimensional pairwise similarity features
│   ├── metrics.py           # Entity-level Macro F0.5 & singleton scoring
│   ├── model.py             # Supervised XGBoost classifier wrapper
│   ├── threshold.py         # Precision-heavy decision threshold optimizer
│   ├── training.py          # Hard negative mining, training & refit workflow
│   ├── inference.py         # Chunked inference & compliant TSV export
│   └── pipeline.py          # Unified CLI entrypoint
│
├── utils/                   # Verification and validation utilities
│   ├── __init__.py
│   └── validator.py         # Submission format validation engine
│
├── tests/                   # Automated test suite
│   ├── __init__.py
│   ├── test_normalization.py
│   ├── test_blocking.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_compliance.py   # Integration test with official validate_submission.py
│
├── scripts/                 # Convenience runners
│   ├── run_pipeline.ps1     # PowerShell execution script
│   └── run_pipeline.bat     # Windows batch execution script
│
├── artifacts/               # Serialized model bundles and metadata
└── output/                  # Final generated submission TSV files
```

---

## 🚀 Quick Start

### 1. Installation

Ensure Python 3.9+ is installed, then install the pinned dependencies:

```bash
cd "BER pro"
pip install -r requirements.txt
```

### 2. Run Automated Test Suite

Verify text normalization, blocking, feature extraction, Macro $F_{0.5}$ metric, and submission compliance:

```bash
# Run all unit and integration tests
py -m unittest discover -s tests -v
```

All 5 test modules verify full compliance with the official `validate_submission.py`.

### 3. Run the Pipeline

#### Fast Iteration / Sample Mode (Recommended for testing on local PC)
Run on a sampled subset of 10,000 entities:

```bash
py src/pipeline.py \
  --train-dir ../resources/dataset/train \
  --test-dir ../resources/dataset/test \
  --sample-train 10000 \
  --sample-test 10000 \
  --output-dir output \
  --model-dir artifacts
```

#### Full Production Execution
Run end-to-end on the complete dataset:

```bash
py src/pipeline.py \
  --train-dir ../resources/dataset/train \
  --test-dir ../resources/dataset/test \
  --output-dir output \
  --model-dir artifacts
```

Alternatively, use the convenience scripts:
```powershell
# Windows PowerShell
.\scripts\run_pipeline.ps1

# Windows CMD
.\scripts\run_pipeline.bat
```

---

## 🔍 Validation & Submission Check

Verify your output files directly against the official challenge validator:

```bash
py ../resources/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir ../resources/dataset/test \
  --check-ids
```

Expected result:
```text
PASS — no blocking issues found. Safe to submit.
```

---

## 🛠️ CLI Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--train-dir` | string | `../resources/dataset/train` | Path to folder containing training TSVs |
| `--test-dir` | string | `../resources/dataset/test` | Path to folder containing test TSVs |
| `--output-dir` | string | `output` | Directory where output TSVs are written |
| `--model-dir` | string | `artifacts` | Directory to save/load trained model bundle |
| `--mode` | string | `all` | `all` (train + infer), `train_only`, or `infer_only` |
| `--sample-train`| int | `None` | Optional limit on train records for fast testing |
| `--sample-test` | int | `None` | Optional limit on test records for fast testing |
| `--chunk-size` | int | `50000` | Batch chunk size for candidate featurization |
| `--validate` | flag | `True` | Automatically runs compliance checks after inference |

---

## 📜 Fair-Play & Integrity Statement
- **Zero External Data**: Built strictly using the challenge-provided dataset.
- **Model Constraints**: XGBoost tabular classifier with $< 50\text{ MB}$ footprint (well within the 8-billion parameter ceiling).
- **Open-Set Ready**: Dynamic country token normalization avoids geographical bias or hardcoded whitelists.
