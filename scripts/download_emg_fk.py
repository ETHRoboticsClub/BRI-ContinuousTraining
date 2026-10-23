"""Download EMG-FK dataset files from Zenodo.

Sections:
1. Setup & Configuration: Base Zenodo record URL and subject IDs.
2. File Downloader: Streaming download with progress reporting.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ZENODO_RECORD_ID = "19453843"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"


def download_subject(subject_id: int, target_dir: Path) -> Path:
    """Download a single Subject_XX.fif file from Zenodo.

    Args:
        subject_id: Integer subject ID (1 to 20).
        target_dir: Destination folder.

    Returns:
        Path to downloaded file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"Subject_{subject_id:02d}.fif"
    file_path = target_dir / filename

    if file_path.exists() and file_path.stat().st_size > 50 * 1024 * 1024:
        print(f"Skipping {filename} (already exists, {file_path.stat().st_size / (1024*1024):.1f} MB)")
        return file_path

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename} from {url}...")

    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        sys.stdout.write(f"\r  Progress: {mb:.1f}/{total_mb:.1f} MB ({percent}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, file_path, reporthook=progress)
    print(f"\n  Saved {filename} ({file_path.stat().st_size / (1024*1024):.1f} MB)")
    return file_path


def main():
    parser = argparse.ArgumentParser(description="Download EMG-FK dataset from Zenodo.")
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of subject IDs to download (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/raw/emg_fk_source",
        help="Destination directory for raw FIF files",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    print(f"Downloading {len(args.subjects)} subjects to {target_dir}...")
    for s_id in args.subjects:
        download_subject(s_id, target_dir)
    print("All requested downloads completed successfully.")


if __name__ == "__main__":
    main()
