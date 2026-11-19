"""Download and index the ReactEMG 6-class preprocessed dataset archive."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset import CLASS_NAMES_6, REACTEMG_DOWNLOAD_URL, discover_reactemg_files  # noqa: E402


def _resolve_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ReactEMG-TrainingPipeline/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        chunk_size = 1024 * 1024
        with (
            dest.open("wb") as out_f,
            tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {dest.name}",
            ) as pbar,
        ):
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                pbar.update(len(chunk))


def _extract_archive(archive_path: Path, data_root: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        top_prefixes = {Path(m.filename).parts[0] for m in members if m.filename and not m.is_dir()}
        strip_prefix = "data/" if top_prefixes == {"data"} else None

        for member in tqdm(members, desc="Extracting data.zip"):
            if not member.filename or member.is_dir():
                continue
            rel = member.filename
            if strip_prefix and rel.startswith(strip_prefix):
                rel = rel[len(strip_prefix) :]
            if not rel:
                continue
            target = (data_root / rel).resolve()
            if not str(target).startswith(str(data_root.resolve())):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target.open("wb") as dst:
                dst.write(src.read())


def generate_synthetic_smoke_dataset(data_root: Path, seed: int = 42, num_samples: int = 1800) -> None:
    """Create tiny synthetic 6-class CSV recordings matching ReactEMG schemas for offline tests."""
    rng = np.random.default_rng(seed)
    data_root.mkdir(parents=True, exist_ok=True)
    roam_dir = data_root / "ROAM_EMG" / "synthetic_subj"
    epn_dir = data_root / "EMG-EPN-612" / "subj01"
    roam_dir.mkdir(parents=True, exist_ok=True)
    epn_dir.mkdir(parents=True, exist_ok=True)

    seg = max(120, num_samples // 6)
    gt = np.concatenate([np.full(seg, c, dtype=np.int64) for c in range(6)])
    total_len = len(gt)

    def _make_df(phase: int) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {"gt": gt}
        for ch in range(8):
            amp = 8.0 + 14.0 * ((gt + ch + phase) % 6 == 0).astype(np.float32) + 4.0 * (gt > 0)
            sig = np.clip(rng.normal(0.0, amp, size=total_len), -128, 127).astype(np.int8)
            data[f"emg_{ch}"] = sig.astype(np.int64)
        return pd.DataFrame(data)

    _make_df(0).to_csv(roam_dir / "session_01.csv", index=False)
    _make_df(1).to_csv(epn_dir / "epn_session_01.csv", index=False)
    _make_df(2).to_csv(epn_dir / "epn_session_02.csv", index=False)

    # Also write an unlabeled file to verify unlabel exclusion from supervised splits
    unlab = _make_df(0).drop(columns=["gt"])
    unlab.to_csv(roam_dir / "unlabel_session.csv", index=False)


def build_manifest(
    data_root: Path,
    manifest_dir: Path,
    dataset_selection: str,
    num_classes: int,
    epn_subset_percentage: float,
    val_split_ratio: float,
    seed: int,
) -> Path:
    """Scan data_root, partition splits, and write data/processed/reactemg/manifest.json."""
    splits = discover_reactemg_files(
        data_root=data_root,
        dataset_selection=dataset_selection,
        num_classes=num_classes,
        epn_subset_percentage=epn_subset_percentage,
        val_split_ratio=val_split_ratio,
        seed=seed,
    )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    payload = {
        "data_root": str(data_root),
        "dataset_selection": dataset_selection,
        "num_classes": int(num_classes),
        "class_names": list(CLASS_NAMES_6[: int(num_classes)]),
        "epn_subset_percentage": float(epn_subset_percentage),
        "val_split_ratio": float(val_split_ratio),
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": {k: [str(p) for p in v] for k, v in splits.items()},
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and index ReactEMG preprocessed data")
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "src" / "params.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "data" / "processed" / "reactemg")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-extract data.zip")
    parser.add_argument("--skip-download", action="store_true", help="Skip Dropbox download and index local files")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Generate synthetic data instead of downloading")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    data_root = (
        args.data_root if args.data_root is not None else _resolve_path(train_cfg.get("data_root", "data/raw/reactemg"))
    )
    manifest_dir = args.output_dir if args.output_dir.is_absolute() else (_REPO_ROOT / args.output_dir)

    if args.synthetic_smoke:
        generate_synthetic_smoke_dataset(data_root, seed=args.seed)
    elif not args.skip_download:
        existing_data = list(data_root.rglob("*.csv")) + list(data_root.rglob("*.npy")) if data_root.exists() else []
        zip_path = data_root / "data.zip"
        if args.force or not existing_data:
            if args.force or not zip_path.exists():
                _download_with_progress(REACTEMG_DOWNLOAD_URL, zip_path)
            _extract_archive(zip_path, data_root)

    manifest_path = build_manifest(
        data_root=data_root,
        manifest_dir=manifest_dir,
        dataset_selection=str(train_cfg.get("dataset_selection", "pub_with_epn")),
        num_classes=int(model_cfg.get("num_classes", 6)),
        epn_subset_percentage=float(train_cfg.get("epn_subset_percentage", 1.0)),
        val_split_ratio=float(train_cfg.get("val_split_ratio", 0.05)),
        seed=int(args.seed),
    )
    print(f"ReactEMG manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
