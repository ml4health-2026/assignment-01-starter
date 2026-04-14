"""
Download the brain tumor MRI dataset for Assignment 1.

Usage:
    python download_data.py
"""

import urllib.request
import zipfile
import pathlib

URL = "https://github.com/ml4health-2026/assignment-01-starter/releases/download/v1.0-data/brain_tumor_data.zip"
ZIP = pathlib.Path("brain_tumor_data.zip")
DEST = pathlib.Path("data")


def download(url: str, dest: pathlib.Path) -> None:
    print(f"Downloading dataset from {url} ...")
    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        print(f"\r  {pct}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print("\r  Done.   ")


def extract(zip_path: pathlib.Path) -> None:
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(".")
    zip_path.unlink()
    print("Extracted to data/")


if __name__ == "__main__":
    if DEST.exists():
        print("data/ already exists — skipping download.")
        print("Delete the data/ folder and re-run if you want a fresh download.")
    else:
        download(URL, ZIP)
        extract(ZIP)
        print("\nSetup complete. Your data/ folder is ready.")
