"""
Download Google Cluster Data 2019 instance usage data.

This script downloads the GCT 2019 instance usage file documented
in data/README.md.

The raw dataset is not stored in this repository.
"""

from pathlib import Path
from urllib.request import urlretrieve


# -------------------------------------------------------------------
# Dataset URL
# -------------------------------------------------------------------

DATA_URL = (
    "https://storage.googleapis.com/clusterdata_2019_a/"
    "instance_usage-000000000000.json.gz"
)


# -------------------------------------------------------------------
# Local output location
# -------------------------------------------------------------------

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "instance_usage-000000000000.json.gz"


def download_dataset():
    """Download the Google Cluster Data 2019 instance usage file."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(f"File already exists: {OUTPUT_FILE}")
        print("Download skipped.")
        return

    print("Downloading Google Cluster Data 2019 instance usage file...")
    print(f"Source : {DATA_URL}")
    print(f"Output : {OUTPUT_FILE}")

    urlretrieve(DATA_URL, OUTPUT_FILE)

    print("\nDownload completed successfully.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    download_dataset()
