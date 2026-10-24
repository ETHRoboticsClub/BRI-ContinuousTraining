"""Download NinaPro DB2 and convert it to the recording format of the README (one .npz per repetition).

EMG: 12 Delsys electrodes at 2 kHz. Targets: the 22 CyberGlove sensors, scaled per subject to [0, 1]
between their 1st and 99th percentiles. Only exercises 1 and 2 have glove data (3 is force).
Each movement repetition (from its onset to the next onset, ~5 s movement + 3 s rest) is one file.

    python scripts/fetch_ninapro.py --subjects 1-10
    dvc add data/raw/ninapro_db2 && git add data/raw/ninapro_db2.dvc && dvc push
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

URL = "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{}.zip"
FS = 2000
EXERCISES = (1, 2)
REST_AFTER_LAST = 3  # seconds kept after the last movement of a file


def parse_subjects(text):
    subjects = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        subjects += range(int(lo), int(hi or lo) + 1)
    return subjects


def download(subject, cache_dir):
    """Zip -> cache_dir/DB2_s<subject>/S<subject>_E<e>_A1.mat (only the exercises we use)."""
    out = cache_dir / f"DB2_s{subject}"
    mats = [out / f"S{subject}_E{e}_A1.mat" for e in EXERCISES]
    if all(m.exists() for m in mats):
        return mats
    out.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"DB2_s{subject}.zip"
    if not zip_path.exists():
        print(f"downloading {URL.format(subject)}")
        part = zip_path.with_suffix(".part")
        with urllib.request.urlopen(URL.format(subject)) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.rename(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for m in mats:
            with z.open(f"DB2_s{subject}/{m.name}") as src, open(m, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zip_path.unlink()
    return mats


def segments(stimulus):
    """(start, stop) of each repetition: movement onset -> next onset (or end of the rest after the last one)."""
    moving = (stimulus > 0).astype(np.int8)
    onsets = np.flatnonzero(np.diff(moving) == 1) + 1
    last_stop = np.flatnonzero(moving)[-1] + 1 + REST_AFTER_LAST * FS
    stops = np.append(onsets[1:], min(last_stop, len(stimulus)))
    return zip(onsets, stops, strict=True)


def convert(subject, mats, out_dir):
    data = [loadmat(m, variable_names=["emg", "glove", "restimulus", "rerepetition"]) for m in mats]
    glove = np.concatenate([d["glove"] for d in data])
    lo, hi = np.percentile(glove, [1, 99], axis=0)
    out = out_dir / f"s{subject}"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for exercise, d in zip(EXERCISES, data, strict=True):
        length = min(len(d["emg"]), len(d["restimulus"]))
        stimulus, repetition = d["restimulus"][:length, 0], d["rerepetition"][:length, 0]
        targets = np.clip((d["glove"] - lo) / (hi - lo), 0, 1)
        for start, stop in segments(stimulus):
            movement, rep = int(stimulus[start]), int(repetition[start])
            np.savez(
                out / f"e{exercise}_m{movement:02d}_r{rep}.npz",
                emg=d["emg"][start:stop].T.astype(np.float32),
                fs=FS,
                channel_names=[f"emg_{i + 1}" for i in range(d["emg"].shape[1])],
                targets=targets[start:stop].T.astype(np.float32),
                target_names=[f"glove_{i + 1}" for i in range(glove.shape[1])],
                meta_subject=subject,
                meta_exercise=exercise,
                meta_movement=movement,
                meta_repetition=rep,
            )
            n += 1
    print(f"subject {subject}: {n} repetitions -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", default="1-10", help="e.g. 1-10 or 1,3,5 (DB2 has 40)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/ninapro_db2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/ninapro_db2"))
    parser.add_argument("--keep-cache", action="store_true", help="keep the .mat files after conversion")
    args = parser.parse_args()
    for subject in parse_subjects(args.subjects):
        mats = download(subject, args.cache_dir)
        convert(subject, mats, args.out_dir)
        if not args.keep_cache:
            shutil.rmtree(mats[0].parent)
