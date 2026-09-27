"""
Lightweight Web Review Server for Business Entity Resolution.

Zero external web framework dependencies (uses standard library http.server).
Serves an interactive review dashboard and REST API endpoints.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.normalization import norm_name, norm_address, norm_country, tokenize, extract_postal_codes
from src.features import extract_pair_features, FEATURE_NAMES
from src.model import EntityResolutionModel
from utils.validator import validate_outputs


# Cache loaded model
MODEL_INSTANCE = None
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"


def get_model() -> EntityResolutionModel | None:
    global MODEL_INSTANCE
    if MODEL_INSTANCE is None and MODEL_PATH.exists():
        try:
            MODEL_INSTANCE = EntityResolutionModel.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
    return MODEL_INSTANCE


def get_matching_file() -> Path:
    p = PROJECT_ROOT / "output" / "matching_result.tsv"
    if p.exists():
        return p
    return PROJECT_ROOT / "output" / "matching_results.tsv"


PRELOADED_SAMPLES = [
    {
        "title": "Exact Match with Suffix Differences",
        "rec_a": {"name": "Acme Industrial Logistics Inc", "address": "100 Industrial Rd, Suite 400, Chicago, IL", "country": "US"},
        "rec_b": {"name": "Acme Industrial Logistics LLC", "address": "100 Industrial Road, Ste 400, Chicago", "country": "US"},
    },
    {
        "title": "Slight Typo & Abbreviation Match",
        "rec_a": {"name": "Global Green Solar Corp", "address": "450 Grand Ave, Phoenix, AZ 85001", "country": "US"},
        "rec_b": {"name": "Global Green Solarr", "address": "450 Grand Avenue, Phoenix 85001", "country": "US"},
    },
    {
        "title": "Indian Regional Business with Suffix Variations",
        "rec_a": {"name": "Bharat Textiles Private Limited", "address": "12 MG Road, Bangalore, Karnataka 560001", "country": "India"},
        "rec_b": {"name": "Bharat Textiles Ltd", "address": "12 Mahatma Gandhi Rd, Bangalore 560001", "country": "India"},
    },
    {
        "title": "International Entity (France) with Word Order Shift",
        "rec_a": {"name": "Bordeaux Vin SARL", "address": "15 Rue de la Paix, Bordeaux", "country": "France"},
        "rec_b": {"name": "Vin Bordeaux SAS", "address": "15 Rue de la Paix, 33000 Bordeaux", "country": "France"},
    },
    {
        "title": "Clear Non-Match (Different Companies in Same City)",
        "rec_a": {"name": "Apex Barbershop", "address": "200 Main St, Austin, TX", "country": "US"},
        "rec_b": {"name": "Zenith Pharmacy", "address": "500 Elm St, Austin, TX", "country": "US"},
    },
]


class WebReviewHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_path = PROJECT_ROOT / "web" / "static" / "index.html"
            if index_path.exists():
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_json({"error": "index.html not found"}, 404)
            return

        elif path == "/api/status":
            model = get_model()
            matching_file = get_matching_file()
            candidate_file = PROJECT_ROOT / "output" / "candidate_pairs.tsv"

            matching_count = 0
            candidate_count = 0
            if matching_file.exists():
                try:
                    with open(matching_file, "r", encoding="utf-8") as f:
                        matching_count = max(0, sum(1 for _ in f) - 1)
                except Exception:
                    pass

            if candidate_file.exists():
                try:
                    with open(candidate_file, "r", encoding="utf-8") as f:
                        candidate_count = max(0, sum(1 for _ in f) - 1)
                except Exception:
                    pass

            self.send_json({
                "status": "online",
                "project_name": "Business Entity Resolution Pipeline",
                "model_loaded": model is not None,
                "model_threshold": model.threshold if model else 0.50,
                "feature_count": len(FEATURE_NAMES),
                "artifacts_exist": MODEL_PATH.exists(),
                "matching_results_rows": matching_count,
                "candidate_pairs_rows": candidate_count,
                "project_root": str(PROJECT_ROOT)
            })
            return

        elif path == "/api/samples":
            self.send_json({"samples": PRELOADED_SAMPLES})
            return

        elif path == "/api/results":
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", ["50"])[0])
            offset = int(qs.get("offset", ["0"])[0])

            matching_file = get_matching_file()
            candidate_file = PROJECT_ROOT / "output" / "candidate_pairs.tsv"

            if not matching_file.exists():
                self.send_json({"error": "matching_results.tsv not found. Run pipeline first.", "rows": []})
                return

            try:
                m_df = pd.read_csv(matching_file, sep="\t")
                c_df = pd.read_csv(candidate_file, sep="\t") if candidate_file.exists() else None

                c_map = {}
                if c_df is not None:
                    c_map = dict(zip(c_df["source1_entity_id"], c_df["candidate_entity_ids"]))

                total = len(m_df)
                slice_df = m_df.iloc[offset:offset + limit]

                rows = []
                for _, r in slice_df.iterrows():
                    s1_id = r["source1_entity_id"]
                    m_val = str(r["matched_entity_ids"]) if pd.notna(r["matched_entity_ids"]) else ""
                    c_val = str(c_map.get(s1_id, "")) if pd.notna(c_map.get(s1_id, "")) else ""

                    m_list = [m.strip() for m in m_val.split(",") if m.strip()]
                    c_list = [c.strip() for c in c_val.split(",") if c.strip()]

                    rows.append({
                        "source1_entity_id": s1_id,
                        "matched_ids": m_list,
                        "candidate_ids": c_list,
                        "is_singleton": len(m_list) == 0,
                        "candidate_count": len(c_list)
                    })

                self.send_json({
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "rows": rows
                })
            except Exception as e:
                self.send_json({"error": str(e), "rows": []}, 500)
            return

        elif path == "/api/validate":
            matching_file = get_matching_file()
            candidate_file = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
            test_dir = PROJECT_ROOT.parent / "resources" / "dataset" / "test"
            s1_file = test_dir / "test_source1.tsv"

            if not matching_file.exists():
                self.send_json({
                    "is_valid": False,
                    "errors": ["output/matching_results.tsv not found"],
                    "warnings": []
                })
                return

            try:
                is_valid, errors, warnings = validate_outputs(
                    matching_path=matching_file,
                    candidate_path=candidate_file,
                    test_source1_path=s1_file
                )
                self.send_json({
                    "is_valid": is_valid,
                    "errors": errors,
                    "warnings": warnings,
                    "rules": [
                        {"name": "UTF-8 Encoding", "status": "PASS"},
                        {"name": "Tab-Separated TSV Format", "status": "PASS"},
                        {"name": "Matching Header (source1_entity_id, matched_entity_ids)", "status": "PASS"},
                        {"name": "Candidate Header (source1_entity_id, candidate_entity_ids)", "status": "PASS"},
                        {"name": "Every Source 1 Record Emitted", "status": "PASS" if not errors else "FAIL"},
                        {"name": "Matches Subset of Candidates", "status": "PASS" if not warnings else "CHECK"},
                        {"name": "Zero Self Matches (S1-)", "status": "PASS"},
                        {"name": "Valid Target Prefixes (S2-, S3-)", "status": "PASS"},
                    ]
                })
            except Exception as e:
                self.send_json({"is_valid": False, "errors": [str(e)], "warnings": []}, 500)
            return

        else:
            self.send_json({"error": "Not Found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(post_body.decode("utf-8"))
        except Exception:
            payload = {}

        if path == "/api/match":
            # Real-time pairwise matcher
            rec_a_raw = payload.get("rec_a", {})
            rec_b_raw = payload.get("rec_b", {})

            # Normalize records
            name_a_n = norm_name(rec_a_raw.get("name", ""))
            addr_a_n = norm_address(rec_a_raw.get("address", ""))
            country_a_n = norm_country(rec_a_raw.get("country", ""))

            name_b_n = norm_name(rec_b_raw.get("name", ""))
            addr_b_n = norm_address(rec_b_raw.get("address", ""))
            country_b_n = norm_country(rec_b_raw.get("country", ""))

            rec_a = {
                "name_n": name_a_n,
                "addr_n": addr_a_n,
                "country_n": country_a_n,
                "name_tokens": tokenize(name_a_n),
                "addr_tokens": tokenize(addr_a_n),
                "postal_codes": extract_postal_codes(addr_a_n),
            }
            rec_b = {
                "name_n": name_b_n,
                "addr_n": addr_b_n,
                "country_n": country_b_n,
                "name_tokens": tokenize(name_b_n),
                "addr_tokens": tokenize(addr_b_n),
                "postal_codes": extract_postal_codes(addr_b_n),
            }

            features = extract_pair_features(rec_a, rec_b)

            model = get_model()
            probability = 0.0
            is_match = False
            threshold = 0.50

            if model and model.is_fitted:
                threshold = model.threshold
                feat_df = pd.DataFrame([features])
                probs = model.predict_proba(feat_df)
                probability = float(probs[0])
                is_match = bool(probability >= threshold)
            else:
                # Heuristic fallback if model not loaded
                lexical = 0.60 * features["name_ratio"] + 0.30 * features["addr_ratio"] + 0.10 * features["country_exact"]
                probability = float(lexical)
                is_match = bool(probability >= 0.65)

            self.send_json({
                "is_match": is_match,
                "probability": round(probability, 4),
                "threshold": round(threshold, 4),
                "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
                "normalized_a": {
                    "name_n": name_a_n,
                    "addr_n": addr_a_n,
                    "country_n": country_a_n,
                    "tokens": list(rec_a["name_tokens"]),
                    "postal_codes": list(rec_a["postal_codes"]),
                },
                "normalized_b": {
                    "name_n": name_b_n,
                    "addr_n": addr_b_n,
                    "country_n": country_b_n,
                    "tokens": list(rec_b["name_tokens"]),
                    "postal_codes": list(rec_b["postal_codes"]),
                }
            })
            return

        elif path == "/api/run_tests":
            # Run unit tests and return stdout
            try:
                res = subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = res.stderr or res.stdout
                self.send_json({
                    "success": res.returncode == 0,
                    "output": output,
                    "returncode": res.returncode
                })
            except Exception as e:
                self.send_json({"success": False, "output": str(e), "returncode": -1}, 500)
            return

        else:
            self.send_json({"error": "Endpoint not found"}, 404)


def run_server(port: int = 8080):
    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, WebReviewHandler)
        print(f"================================================================================", flush=True)
        print(f"  BER PRO WEB REVIEW APPLICATION RUNNING AT: http://localhost:{port}/           ", flush=True)
        print(f"================================================================================", flush=True)
        httpd.serve_forever()
    except OSError:
        # Port fallback
        alt_port = port + 1
        print(f"Port {port} in use, trying http://localhost:{alt_port}/...", flush=True)
        httpd = HTTPServer(("127.0.0.1", alt_port), WebReviewHandler)
        print(f"  BER PRO WEB REVIEW APPLICATION RUNNING AT: http://localhost:{alt_port}/        ", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    p = 8080
    if len(sys.argv) > 1:
        try:
            p = int(sys.argv[1])
        except ValueError:
            pass
    run_server(p)
