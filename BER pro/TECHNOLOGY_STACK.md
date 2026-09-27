# Complete Technology Stack Inventory — BER Pro

This document provides a comprehensive, verified, and data-based technical inventory of all programming languages, machine learning frameworks, data processing technologies, backend architectures, frontend interfaces, testing suites, DevOps tools, and data formats actually utilized in the **Business Entity Resolution (BER Pro)** system.

---

## 1. Master Technology Inventory Table

| Category | Technology | Detected Version | Purpose | Primary Files Where Used |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | Python | 3.14.7 | Core pipeline, ML training, inference, backend API, evaluation | `src/*.py`, `config/*.py`, `web/server.py`, `scripts/*.py` |
| **Language** | JavaScript | ECMAScript 2022+ | Interactive review dashboard, client simulator, SVG animations | `web/static/index.html` |
| **Language** | HTML | HTML5 | Semantic structure for web review dashboard | `web/static/index.html` |
| **Language** | CSS | CSS3 / Tailwind CDN | Utility-first styling, dark/light theme switching, responsive UI | `web/static/index.html` |
| **Language** | PowerShell | 5.1 / 7+ | Automated pipeline runner and daemon starter scripts | `scripts/run_pipeline.ps1`, `scripts/start_webapp.ps1` |
| **Language** | Windows Batch | Windows Shell | Windows CMD wrapper scripts for execution | `scripts/run_pipeline.bat`, `scripts/start_webapp.bat` |
| **ML Framework** | XGBoost | 3.4.1 | Gradient Boosted Decision Trees for pair classification | `src/model.py`, `src/training.py`, `src/inference.py` |
| **ML Library** | scikit-learn | 1.9.1 | GroupKFold cross-validation, probability calibration, metrics | `src/training.py`, `src/metrics.py`, `src/threshold.py` |
| **String Distance** | RapidFuzz | 3.14.6 | High-speed C++ accelerated Levenshtein, Jaro-Winkler, Token ratios | `src/features.py`, `src/normalization.py` |
| **Scientific Math** | SciPy | 1.18.1 | Statistical computation and array mathematical operations | `src/metrics.py`, `src/threshold.py` |
| **Serialization** | Joblib | 1.6.0 | Serializing and loading trained XGBoost booster models | `src/model.py`, `src/pipeline.py`, `src/scale_inference.py` |
| **Data Processing** | pandas | 3.0.6 | Data ingestion, tabular normalization, feature matrix construction | `src/normalization.py`, `src/features.py`, `src/pipeline.py` |
| **Data Processing** | NumPy | 2.5.3 | Vectorized computations, array slicing, threshold sweeps | `src/features.py`, `src/metrics.py`, `src/threshold.py` |
| **Text Processing** | Python `re` | Standard Library | Regex-based legal suffix stripping and postal code extraction | `src/normalization.py` |
| **Data Streaming** | Python `csv` | Standard Library | Streaming generator for multi-million row TSV processing | `src/scale_inference.py` |
| **Backend Server** | Python `http.server` | Standard Library | Native multi-threaded HTTP server (`ThreadingHTTPServer`) | `web/server.py` |
| **API Architecture** | REST JSON API | HTTP/1.1 | Endpoints `/api/health`, `/api/scenarios`, `/api/resolve` | `web/server.py` |
| **CLI Architecture** | Python `argparse` | Standard Library | Configurable CLI parameter management | `src/pipeline.py`, `scripts/evaluate_model.py` |
| **Frontend UI** | Single Page App (SPA) | Pure Client-Side | Standalone client dashboard with offline simulator | `web/static/index.html` |
| **Visualizations** | SVG Vector Gauges | SVG 1.1 | Animated circular confidence and probability gauges | `web/static/index.html` |
| **Testing** | Python `unittest` | Standard Library | Comprehensive automated unit and integration test suite | `tests/test_*.py` |
| **Compliance** | Custom Validator | Custom | Official competition format and submission constraint validation | `utils/validator.py`, `tests/test_compliance.py` |
| **Build System** | setuptools | PEP 517/518 | Python package build system configuration | `pyproject.toml` |
| **Version Control** | Git | 2.55.0.windows.3 | Local and distributed version control | `.git`, `.gitignore` |
| **DevOps CLI** | GitHub CLI (`gh`) | 2.101.0 | Automated GitHub repository management and auth inspection | CLI tooling |
| **Data Format** | TSV | Standard TSV | Tab-separated dataset inputs, candidates, matching results | `output/*.tsv`, dataset files |
| **Data Format** | JSON | RFC 8259 | API transport and evaluation metric serialization | `web/server.py`, `evaluation/metrics.json` |
| **Data Format** | Joblib Binary | Binary | Trained model weights and threshold metadata | `artifacts/model.joblib` |
| **Data Format** | PDF | PDF 1.4 | Executive summary report with visual charts | `BER_Model_Accuracy_and_Web_UI_Summary.pdf` |
| **Data Format** | Markdown | GFM | Technical documentation, architecture, and technology specs | `*.md` |
| **Data Format** | TOML | TOML v1.0.0 | Project packaging specification | `pyproject.toml` |
| **Security** | Zero-Secrets Policy | Custom | Pre-commit regex security scanning, zero credential leakage | `.gitignore`, security scanner |

---

## 2. Detailed Category Breakdown

### 2.1. Programming Languages
* **Python (3.14.7)**: Serves as the primary language across the entire system. Powers data streaming, normalization, blocking, feature extraction, model training, threshold tuning, inference, evaluation, and the backend HTTP server.
* **JavaScript (ECMAScript 2022+)**: Powers the client-side user experience in `web/static/index.html`. Implements responsive UI event handling, dark/light theme switching with local storage persistence, interactive pairwise similarity calculation, dynamic DOM rendering, and real-time SVG circular gauge animations.
* **HTML5**: Defines the semantic, accessible structure of the review dashboard with responsive grid containers, input forms, and data tables.
* **CSS3 / Tailwind CSS (v3 CDN)**: Provides utility-first styling with custom CSS variables for dark and light color palettes (`bg-slate-50`, `dark:bg-[#0a0f1d]`, `text-brand-500`), smooth transitions, and custom scrollbars.
* **PowerShell (5.1 & 7+)**: Provides robust operational automation for running the end-to-end pipeline (`scripts/run_pipeline.ps1`) and launching the web review dashboard daemon (`scripts/start_webapp.ps1`).
* **Windows Batch (`.bat`)**: Delivers double-clickable launcher scripts (`scripts/run_pipeline.bat`, `scripts/start_webapp.bat`) for Windows command prompt environments.

---

### 2.2. Machine Learning & AI Frameworks
* **XGBoost (`xgboost>=2.1.0`, detected 3.4.1)**:
  * Used in `src/model.py`, `src/training.py`, `src/inference.py`, and `src/scale_inference.py`.
  * Configured as an `XGBClassifier` utilizing tree-based boosting (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric="logloss"`).
  * Optimizes binary cross-entropy on pairwise feature vectors to predict whether an entity pair represents a genuine business match.
* **scikit-learn (`scikit-learn>=1.5.0`, detected 1.9.1)**:
  * **GroupKFold**: Partitions training data grouped by `source1_id` into distinct clusters to strictly prevent data leakage between training and validation splits (`src/training.py`).
  * **CalibratedClassifierCV**: Calibrates prediction probabilities using Platt/Sigmoid scaling so output values accurately reflect posterior match likelihoods (`src/model.py`).
* **SciPy (`scipy>=1.14.0`, detected 1.18.1)**:
  * Provides optimized array operations for metric evaluation and threshold search curves.

---

### 2.3. String Similarity & NLP Algorithms
Implemented in `src/features.py` and powered by **RapidFuzz (detected 3.14.6)**:
1. **Levenshtein Distance**: Edit distance measuring single-character insertions, deletions, or substitutions.
2. **Normalized Indel Ratio**: Edit ratio based strictly on insertions and deletions (`rapidfuzz.distance.Indel.normalized_similarity`).
3. **Token Sort Ratio**: Word-order independent token comparison (`fuzz.token_sort_ratio`).
4. **Token Set Ratio**: Set-intersection token comparison that handles redundant or duplicated tokens (`fuzz.token_set_ratio`).
5. **Jaro-Winkler Distance**: Prefix-weighted similarity ideal for entity names with common prefixes (`distance.JaroWinkler.similarity`).
6. **Partial String Ratio**: Substring alignment similarity for shortened company names (`fuzz.partial_ratio`).
7. **Address Word Overlap (Jaccard Index)**: Set intersection over set union on normalized address word tokens.
8. **Postal Code Digit Equality**: Strict binary and prefix-based postal code match indicators.
9. **Country Code Compatibility**: Strict ISO alpha-2 country code equivalence check.

---

### 2.4. Data Processing & Pipeline Architecture
* **pandas (detected 3.0.6)**: Manages tabular operations, column parsing, feature vector concatenation, and structured TSV generation.
* **NumPy (detected 2.5.3)**: Provides vector math, multidimensional array manipulation, and fast vectorized boolean masking for candidate thresholding.
* **Multi-Stage Candidate Blocking (`src/blocking.py`)**:
  * **Stage 1 (Exact Clean Name Index)**: Inverted hash index on normalized legal names.
  * **Stage 2 (Postal Code Bucket Index)**: Hash index on extracted postal codes.
  * **Stage 3 (Address Token Overlap Index)**: Inverted index mapping address tokens to candidate sets.
  * **Cap Governance**: Strictly limits candidates per entity to `max_candidates=15` to guarantee linear $O(N)$ runtime.
* **Data Streaming (`src/scale_inference.py`)**:
  * Employs generator-based file streaming with standard library `csv.reader` to process all 9.97 million target records with constant memory overhead.

---

### 2.5. Backend Architecture
* **Native Python HTTP Server (`web/server.py`)**:
  * Utilizes `http.server.ThreadingHTTPServer` to provide concurrent connection handling without requiring third-party dependencies (Flask, FastAPI, or Uvicorn).
  * Exposes REST JSON endpoints:
    * `GET /api/health`: System health probe, memory diagnostics, and model status.
    * `GET /api/scenarios`: Returns pre-configured benchmark evaluation scenarios.
    * `POST /api/resolve`: Real-time inference endpoint that runs normalized candidate blocking and XGBoost scoring on arbitrary input pairs.

---

### 2.6. Frontend Architecture
* **Single-Page Application (`web/static/index.html`)**:
  * Fully standalone 57 KB responsive dashboard.
  * Built with semantic HTML5 and styled using Tailwind CSS via official CDN.
  * Incorporates animated SVG vector circular gauges for confidence visualization.
  * Contains a client-side similarity simulator that reproduces feature computation directly in JavaScript.
  * Supports real-time dark mode and light mode toggling with `localStorage` persistence.
  * Direct one-click download for the executive summary PDF report.

---

### 2.7. Automated Testing & Verification
* **Test Framework**: Python standard library `unittest`.
* **Execution**: `py -m unittest discover -s tests -v`.
* **Test Suites (19 Automated Tests Passing)**:
  * `tests/test_blocking.py` (3 tests): Exact name index, address token overlap, candidate caps.
  * `tests/test_compliance.py` (1 test): End-to-end official submission compliance and schema verification.
  * `tests/test_features.py` (3 tests): Pairwise feature extraction, identical record scoring, mismatch scoring.
  * `tests/test_metrics.py` (6 tests): Macro F0.5 calculation, precision weighting ($\beta=0.5$), singleton scoring.
  * `tests/test_normalization.py` (6 tests): Text cleaning, legal suffix stripping, address abbreviations, postal codes, country codes.

---

### 2.8. DevOps & Repository Security
* **Version Control**: Git 2.55.0 on branch `main`.
* **GitHub CLI (`gh`)**: 2.101.0 installed in user environment.
* **Security Controls**:
  * Exclusion of `.env`, `.env.*`, `credentials.json`, `secrets.json`, `*.pem`, `*.key`.
  * Exclusion of large generated TSV files (`output/*.tsv`) to comply with GitHub's 100 MB hard file limit.
  * Verification with custom regex security scanner (`scratch/security_audit.py`): Zero secrets, credentials, or keys detected.
* **Proprietary License (`LICENSE`)**: All Rights Reserved notice explicitly defining private ownership and prohibiting unauthorized redistribution.

---

*Inventory generated and verified against the BER Pro codebase on 2026-09-27.*
