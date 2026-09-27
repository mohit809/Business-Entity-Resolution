"""
Create fully verified, compliant competition submission zip file.
Packages matching_results.tsv, candidate_pairs.tsv, documentation, and code.
"""

import os
import sys
import zipfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BER_PRO_DIR = WORKSPACE_ROOT / "BER pro"
OUTPUT_DIR = WORKSPACE_ROOT / "resources" / "output"
TEST_DIR = WORKSPACE_ROOT / "resources" / "dataset" / "test"
VALIDATOR = WORKSPACE_ROOT / "resources" / "utils" / "validate_submission.py"

def main():
    print("=" * 80)
    print("           CREATING OFFICIAL COMPETITION SUBMISSION ZIP ARCHIVE")
    print("=" * 80)

    matching_file = BER_PRO_DIR / "output" / "matching_results.tsv"
    candidate_file = BER_PRO_DIR / "output" / "candidate_pairs.tsv"

    if not matching_file.exists() or not candidate_file.exists():
        print("Error: Output files missing in BER pro/output/")
        return 1

    # First validate the output files
    print("\n[1/3] Validating output files with official submission validator...")
    import subprocess
    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--matching", str(matching_file),
        "--test-dir", str(TEST_DIR)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("FAIL: Validation failed. Cannot build submission package.")
        return 1

    # Build ZIP archive
    zip_targets = [
        WORKSPACE_ROOT / "submission.zip",
        Path.home() / "Downloads" / "Business_Entity_Resolution_Submission_Fixed.zip"
    ]

    for zip_path in zip_targets:
        print(f"\n[2/3] Building zip archive at: {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                # Add architecture & documentation
                arch_file = WORKSPACE_ROOT / "resources" / "ARCHITECTURE.md"
                doc_file = WORKSPACE_ROOT / "resources" / "Documentation_template.md"
                if arch_file.exists():
                    z.write(arch_file, arcname="ARCHITECTURE.md")
                if doc_file.exists():
                    z.write(doc_file, arcname="Documentation_template.md")

                # Add output TSV files
                print("  Adding output/matching_results.tsv...")
                z.write(matching_file, arcname="output/matching_results.tsv")
                print("  Adding output/candidate_pairs.tsv...")
                z.write(candidate_file, arcname="output/candidate_pairs.tsv")

                # Add code files
                ber_src = WORKSPACE_ROOT / "resources" / "business_entity_resolution"
                if ber_src.exists():
                    for root, dirs, files in os.walk(ber_src):
                        for f in files:
                            full_p = Path(root) / f
                            rel_p = full_p.relative_to(WORKSPACE_ROOT / "resources")
                            z.write(full_p, arcname=f"code/{rel_p}")

            size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            print(f"  Successfully created {zip_path.name} ({size_mb:.2f} MB).")
        except Exception as e:
            print(f"  Error creating {zip_path}: {e}")

    print("\n[3/3] Inspecting ZIP Archive Contents...")
    with zipfile.ZipFile(WORKSPACE_ROOT / "submission.zip", "r") as z:
        for info in z.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")

    print("\n" + "=" * 80)
    print("SUBMISSION ZIP IS READY FOR IMMEDIATE UPLOAD!")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
