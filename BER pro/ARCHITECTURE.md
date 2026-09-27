# Business Entity Resolution (BER Pro) — System Architecture

This document provides the definitive architectural specification for the **Business Entity Resolution (BER Pro)** production system, detailing data flow, ML pipelines, candidate blocking, pairwise feature extraction, threshold calibration, backend server design, frontend dashboard architecture, and security governance.

---

## 1. System Overview

BER Pro solves the challenge of cross-source enterprise entity resolution across multi-million row business registries. Given query records from **Source 1** and target pools from **Source 2** and **Source 3**, the system identifies whether each Source 1 business entity matches one or more target records, or represents a singleton (unmatched entity).

```mermaid
flowchart TD
    subgraph InputSources ["External Data Sources"]
        S1["Source 1 Queries\n(name, address, country)"]
        S2["Source 2 Targets\n(name, address, country)"]
        S3["Source 3 Targets\n(name, address, country)"]
    end

    subgraph BERCore ["BER Pro Core Pipeline"]
        Norm["1. Normalization Layer\n(Text cleaning, legal suffix removal, postal code extraction)"]
        Block["2. Inverted Index Blocker\n(Exact name, postal buckets, address token index)"]
        Feat["3. Pairwise Featurizer\n(13 string distance, token set, and contextual features)"]
        Model["4. Calibrated XGBoost Classifier\n(GBDT probability scoring + Macro F0.5 threshold)"]
    end

    subgraph Outputs ["Standardized Outputs"]
        CandPairs[("candidate_pairs.tsv\n(Audit trail of all generated candidates)")]
        MatchResults[("matching_results.tsv\n(Final predicted entity matches & singletons)")]
    end

    S1 --> Norm
    S2 --> Norm
    S3 --> Norm
    Norm --> Block
    Block --> CandPairs
    Block --> Feat
    Feat --> Model
    Model --> MatchResults
```

---

## 2. End-to-End Data Flow

The data flow enforces three invariant properties:
1. **Candidate Persistency**: Every pair scored by the model must originate from `candidate_pairs.tsv`.
2. **Subset Constraint**: $matching\_results \subseteq candidate\_pairs$.
3. **Singleton Completeness**: Every Source 1 entity appears exactly once in both output files. Singletons are represented with an empty match field.

```mermaid
sequenceDiagram
    autonumber
    participant S1 as Source 1 (Queries)
    participant Targets as Source 2 & 3 (Targets)
    participant Blocker as Multi-Stage Blocker
    participant Featurizer as 13-Feature Extractor
    participant Classifier as XGBoost (Threshold 0.74)
    participant Output as TSV Formatter

    Note over S1,Targets: Step 1: Preprocessing & Inverted Indexing
    Targets->>Blocker: Stream 9.97M records & populate hash indices
    S1->>Blocker: Stream Source 1 records & retrieve candidate sets
    Blocker->>Output: Materialize candidate_pairs.tsv
    
    Note over Blocker,Featurizer: Step 2: Feature Extraction
    Blocker->>Featurizer: Pass candidate pairs (S1_ID, Target_ID)
    Featurizer->>Classifier: 13-dimensional dense feature vector
    
    Note over Classifier,Output: Step 3: Decision & Serialization
    Classifier->>Classifier: Predict probability P(match | x)
    Classifier->>Output: Filter pairs where P >= 0.74
    Output->>Output: Deduplicate & format matching_results.tsv
```

---

## 3. Training Pipeline

The training pipeline constructs balanced training sets from ground-truth annotations and hard negatives mined from blocking index collisions.

```mermaid
flowchart TD
    TrainS1["train_source1.tsv"] --> PrepTrain["Text Normalization"]
    TrainS23["train_source2.tsv + train_source3.tsv"] --> PrepTrain
    GroundTruth["train_ground_truth.tsv"] --> LabelMap["Ground Truth Map\n(True Positive Pairs)"]

    PrepTrain --> FitBlocker["Fit Blocker on Train Targets"]
    FitBlocker --> GenTrainPairs["Generate Candidate Pairs"]
    GenTrainPairs & LabelMap --> MineNegatives["Hard-Negative Mining\n(Index collisions without label)"]

    MineNegatives --> PairDataset["Pair Dataset\n(Positives + Hard Negatives)"]
    PairDataset --> FeatExtract["Extract 13 Pairwise Features"]
    FeatExtract --> GBDT["Train XGBoost GBDT\n(n_estimators=300, max_depth=6)"]
    GBDT --> Calib["Calibrate Probabilities\n(Platt / Sigmoid Scaling)"]
    Calib --> ModelArtifact["Save artifacts/model.joblib"]
```

---

## 4. Validation Pipeline

To eliminate data leakage, validation splits are grouped strictly on `source1_id` using `GroupKFold`. All candidate pairs associated with a given Source 1 entity are held out together.

```mermaid
flowchart LR
    Pairs["All Training Pairs"] --> GroupSplit["GroupKFold (n_splits=5)\nGrouped on source1_id"]
    GroupSplit --> TrainFold["Train Fold (80%)"]
    GroupSplit --> ValFold["Validation Fold (20%)"]
    TrainFold --> TrainModel["Fit XGBoost"]
    TrainModel --> ValScore["Predict Val Probabilities"]
    ValFold --> ValScore
    ValScore --> GridSearch["Sweep Thresholds: [0.00 - 1.00]\nStep: 0.01"]
    GridSearch --> MaxF05["Optimize Entity-Level Macro F0.5"]
    MaxF05 --> CalibThreshold["Optimal Threshold: 0.7400"]
```

---

## 5. Test Pipeline

The test pipeline is fully decoupled from training and operates strictly on unlabelled test data:

```mermaid
flowchart TD
    TestS2["test_source2.tsv"] --> StreamTargets["Stream Ingestion & Indexing"]
    TestS3["test_source3.tsv"] --> StreamTargets
    StreamTargets --> InvertedIndices["Inverted Indices in Memory\n(5.66M Names, 60.5K Postal Codes)"]
    
    TestS1["test_source1.tsv\n(1.73M records)"] --> StreamQueries["Stream Query Generator"]
    StreamQueries & InvertedIndices --> BlockLookup["Candidate Retrieval\n(Cap: max 15 per entity)"]
    
    BlockLookup --> WriteCand["Stream write candidate_pairs.tsv"]
    BlockLookup --> FeatGen["Vectorized Feature Extraction"]
    FeatGen --> ModelInfer["XGBoost Batch Inference (Chunk 10,000)"]
    ModelInfer --> ThreshFilter["Apply Decision Gate (P >= 0.74)"]
    ThreshFilter --> WriteMatch["Stream write matching_results.tsv"]
```

---

## 6. Inference Pipeline

The inference pipeline supports both batch streaming (via `src/scale_inference.py`) and single-pair real-time scoring (via `web/server.py`):

```mermaid
flowchart LR
    subgraph Inputs
        A["Entity A (Query)"]
        B["Entity B (Candidate)"]
    end

    subgraph Extraction ["Feature Extraction (13 Features)"]
        F1["Name Fuzzy Distances\n(Ratio, Indel, TokenSort, TokenSet, JaroWinkler, Partial)"]
        F2["Name Length Metrics\n(Length Difference, Ratio, Prefix Match)"]
        F3["Address & Geo\n(Token Overlap Jaccard, Numeric Token Overlap, Postal Match)"]
        F4["Country\n(Exact ISO Match Indicator)"]
    end

    subgraph Inference ["Model Execution"]
        XGB["XGBClassifier.predict_proba()"]
        Gate{"P(match) >= 0.74?"}
    end

    subgraph Decision ["Output Formulation"]
        Match["Match Confirmed (Confidence = P)"]
        NoMatch["No Match (Fallback / Singleton)"]
    end

    Inputs --> F1 & F2 & F3 & F4
    F1 & F2 & F3 & F4 --> XGB
    XGB --> Gate
    Gate -- Yes --> Match
    Gate -- No --> NoMatch
```

---

## 7. Candidate-Generation Architecture (Multi-Stage Blocking)

Naive Cartesian all-pairs comparison would require:
$$\text{Pairs} = 1{,}732{,}544 \times (4{,}887{,}273 + 5{,}082{,}316) \approx 1.72 \times 10^{13} \text{ pairs}$$
Evaluating $1.72 \times 10^{13}$ pairs is computationally infeasible. BER Pro utilizes three complementary hash-indexed blocking stages with $O(1)$ lookup:

1. **Stage 1 (Exact Normalized Legal Name Index)**:
   Maps cleaned, lowercased, legal-suffix-stripped entity names directly to target entity IDs.
2. **Stage 2 (Postal Code Bucket Index)**:
   Extracts digit sequences from addresses and indexes targets within the same postal code.
3. **Stage 3 (Address Token Overlap Index)**:
   Inverts address tokens (filtering high-frequency words) to match entities with matching street names or suite numbers.
4. **Candidate Pruning**:
   Candidates per entity are unioned and capped at `max_candidates=15` to ensure linear $O(N)$ execution time.

---

## 8. Feature-Engineering Architecture (13 Dimensions)

Every candidate pair $(S_1, S_2)$ is converted into a 13-dimensional numerical vector:

| # | Feature Name | Description | Algorithm |
| :---: | :--- | :--- | :--- |
| 1 | `name_ratio` | Normalized Levenshtein similarity on raw names | RapidFuzz `fuzz.ratio` |
| 2 | `clean_name_ratio` | Normalized Levenshtein similarity on cleaned names | RapidFuzz `fuzz.ratio` |
| 3 | `name_token_sort` | Token-sort ratio (word order independent) | RapidFuzz `fuzz.token_sort_ratio` |
| 4 | `name_token_set` | Token-set ratio (handles duplicate/extra tokens) | RapidFuzz `fuzz.token_set_ratio` |
| 5 | `name_jaro_winkler`| Prefix-biased character distance | RapidFuzz `distance.JaroWinkler` |
| 6 | `name_partial_ratio`| Best matching substring similarity | RapidFuzz `fuzz.partial_ratio` |
| 7 | `name_len_diff` | Absolute difference in name string lengths | $\|len(A) - len(B)\|$ |
| 8 | `name_len_ratio` | Ratio of shorter name to longer name | $\min(L_1, L_2) / \max(L_1, L_2)$ |
| 9 | `name_prefix_match`| Binary flag: exact match on first 4 characters | $A[:4] == B[:4]$ |
| 10| `addr_token_overlap`| Jaccard similarity of address word tokens | $\|T_A \cap T_B\| / \|T_A \cup T_B\|$ |
| 11| `addr_num_overlap` | Jaccard similarity of numeric address tokens | $\|N_A \cap N_B\| / \|N_A \cup N_B\|$ |
| 12| `postal_match` | Equality indicator of extracted postal codes | $P_A == P_B$ |
| 13| `country_match` | Equality indicator of ISO country codes | $C_A == C_B$ |

---

## 9. Machine Learning Architecture

The classification core uses **XGBoost (3.4.1)**:
* **Algorithm**: Gradient Boosted Decision Trees (`XGBClassifier`)
* **Objective**: `binary:logistic`
* **Tree Parameters**: `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`
* **Calibration**: Platt scaling (logistic sigmoid mapping over tree margins)
* **Decision Optimization**: Threshold search maximizing entity-level Macro $F_{0.5}$:
  $$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Since precision is weighted twice as heavily as recall, the optimal threshold shifts from default 0.50 up to **0.7400**, suppressing false positives.

---

## 10. Web Dashboard Architecture

The review dashboard is designed with a lightweight, multi-threaded native Python backend and a zero-dependency frontend single-page application:

```mermaid
flowchart TD
    subgraph Client ["Browser Client (index.html)"]
        UI["Tailwind CSS + HTML5 Interface"]
        Gauges["SVG Dynamic Circular Gauges"]
        Theme["Dark / Light Mode Controller"]
        Simulator["Client-Side Similarity Calculator"]
    end

    subgraph Server ["Backend Server (web/server.py)"]
        HTTP["http.server.ThreadingHTTPServer (Port 8080)"]
        Router["Request Router & Static File Handler"]
        APIHealth["/api/health Handler"]
        APIScenarios["/api/scenarios Handler"]
        APIResolve["/api/resolve Handler"]
    end

    subgraph ModelLayer ["ML Inference Core"]
        Pipeline["BER Model Wrapper"]
        JoblibModel["model.joblib"]
    end

    UI -->|HTTP Requests| HTTP
    HTTP --> Router
    Router -->|GET /api/health| APIHealth
    Router -->|GET /api/scenarios| APIScenarios
    Router -->|POST /api/resolve| APIResolve
    APIResolve --> Pipeline
    Pipeline --> JoblibModel
    Router -->|Serve Static HTML/CSS/JS| Client
```

---

## 11. Security Architecture

1. **Zero-Trust Input Sanitization**:
   All entity strings are stripped, sanitized, and normalized before feature computation.
2. **Zero-Secrets Policy**:
   No credentials, tokens, SSH keys, passwords, or connection strings are stored in code or repository tracking.
3. **Network Isolation**:
   The entire pipeline and web server run completely offline on `localhost` without outbound cloud API dependencies.
4. **File Size Enforcement**:
   Generated multi-gigabyte TSVs are strictly excluded via `.gitignore` to prevent repository bloat and GitHub push failure.

---

## 12. Repository Architecture

```text
BER pro/
├── ARCHITECTURE.md                             # Comprehensive architectural specifications
├── BER_Model_Accuracy_and_Web_UI_Summary.pdf  # 4-page executive summary PDF report
├── DOCUMENTATION.md                            # Complete engineering documentation
├── LICENSE                                     # Proprietary All-Rights-Reserved license
├── README.md                                   # Root project overview and operational guide
├── TECHNOLOGY_STACK.md                         # Complete technology inventory and audit
├── .gitignore                                  # Security-hardened exclusions
├── pyproject.toml                              # PEP 517 build configuration
├── requirements.txt                            # Production package dependencies
│
├── artifacts/
│   └── model.joblib                            # Serialized trained XGBoost booster (429 KB)
│
├── config/
│   ├── __init__.py
│   └── settings.py                             # Centralized pipeline configuration dataclasses
│
├── output/
│   └── .gitkeep                                # Tracked output folder (TSVs excluded by .gitignore)
│
├── scripts/
│   ├── evaluate_model.py                       # Standalone rigorous evaluation script
│   ├── preview_demo.py                         # Console benchmark scenarios runner
│   ├── run_pipeline.bat                        # Windows Batch end-to-end pipeline launcher
│   ├── run_pipeline.ps1                        # PowerShell end-to-end pipeline launcher
│   ├── start_webapp.bat                        # Windows Batch web server launcher
│   └── start_webapp.ps1                        # PowerShell web server launcher
│
├── src/
│   ├── __init__.py
│   ├── blocking.py                             # Multi-stage inverted index candidate generator
│   ├── features.py                             # 13 pairwise string and contextual similarity features
│   ├── inference.py                            # Pair scoring and threshold gating
│   ├── metrics.py                              # Official entity-level Macro F0.5 implementation
│   ├── model.py                                # XGBoost classifier wrapper and persistence
│   ├── normalization.py                        # Text cleaning, legal suffixes, postal extraction
│   ├── pipeline.py                             # Standard batch pipeline coordinator
│   ├── scale_inference.py                      # Memory-constant streaming generator for 9.97M rows
│   ├── threshold.py                            # Grid search threshold optimization
│   └── training.py                             # Grouped validation and model training
│
├── tests/
│   ├── __init__.py
│   ├── test_blocking.py                        # Unit tests for candidate blocking
│   ├── test_compliance.py                      # Submission format compliance integration test
│   ├── test_features.py                        # Unit tests for 13 pairwise features
│   ├── test_metrics.py                         # Unit tests for Macro F0.5 and singleton rules
│   └── test_normalization.py                  # Unit tests for text cleaning and regex
│
├── utils/
│   ├── __init__.py
│   └── validator.py                            # Submission TSV schema and constraint validator
│
└── web/
    ├── server.py                               # Multi-threaded HTTP server with REST JSON API
    └── static/
        └── index.html                          # Responsive SPA review dashboard
```
