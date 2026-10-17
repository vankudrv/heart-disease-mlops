"""Download the UCI Heart Disease (Cleveland) dataset.

Usage (from the project root):
    python scripts/download_data.py

The file is saved to data/raw/processed.cleveland.data. If the download fails
(no internet, UCI server down), copy the file in manually - see README.md.
"""
from pathlib import Path
import sys

import requests

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_FILE = RAW_DIR / "processed.cleveland.data"


def download(url: str = URL, out_file: Path = OUT_FILE) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # stop loudly on HTTP errors
    out_file.write_bytes(response.content)
    print(f"Saved {len(response.content):,} bytes to {out_file}")
    return out_file


if __name__ == "__main__":
    try:
        download()
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Manual option: download 'processed.cleveland.data' from "
            "https://archive.ics.uci.edu/dataset/45/heart+disease and place it in data/raw/",
            file=sys.stderr,
        )
        sys.exit(1)
