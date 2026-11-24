"""Download the released results from Hugging Face.

    python -m lensdiff.download            # results only (about 1.4 GB), into data/
    python -m lensdiff.download --training-data
    python -m lensdiff.download --harmful-compliance-data   # gated: accept its terms on Hugging Face first

The paper's tables need only the results (python paper/reproduce.py).
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

RESULTS = "maxipetro/lens-diffing-backdoors-results"
TRAINING_DATA = "maxipetro/lens-diffing-backdoors-training-data"
HARMFUL_COMPLIANCE_DATA = "maxipetro/lens-diffing-backdoors-harmful-compliance-data"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--to", type=Path, default=Path("data"))
    p.add_argument("--training-data", action="store_true", help="the training data instead of the results")
    p.add_argument("--harmful-compliance-data", action="store_true", help="the gated harmful-compliance data")
    a = p.parse_args()
    if a.training_data:
        snapshot_download(TRAINING_DATA, repo_type="dataset", local_dir=a.to / "training-data")
    if a.harmful_compliance_data:
        snapshot_download(HARMFUL_COMPLIANCE_DATA, repo_type="dataset", local_dir=a.to / "training-data")
    if not (a.training_data or a.harmful_compliance_data):
        snapshot_download(RESULTS, repo_type="dataset", local_dir=a.to)
    print("downloaded to", a.to)


if __name__ == "__main__":
    main()
