"""
Root CLI entrypoint for Scalable Candidate Generation and S1 Coverage Validation.
Forwards execution to BER pro/scripts/generate_candidates.py.
"""

import sys
from pathlib import Path

# Add BER pro to sys.path
BER_PRO = Path(__file__).resolve().parent / "BER pro"
if str(BER_PRO) not in sys.path:
    sys.path.insert(0, str(BER_PRO))

from scripts.generate_candidates import main

if __name__ == "__main__":
    sys.exit(main())
