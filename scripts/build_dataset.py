import urllib.request
from pathlib import Path

import mne
import numpy as np
import yaml
from scipy import signal

from src.dataset import recording_path, save_recording
from src.preprocess import Preprocess

ZENODO_URL = "https://zenodo.org/records/19453843/files"
SUBJECTS = range(1, 21)
SKIP_START = 0.25


def main(params, raw_dir, out_dir) -> None:
    """Download the recordings into raw_dir, preprocess them and save each whole recording to out_dir.

    The EMG is saved filtered but not normalized: `EMGFK` z-scores it at load time.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess = Preprocess(**params)
    for s in SUBJECTS:
        path = raw_dir / f"Subject_{s:02d}.fif"
        if not path.exists():
            print(f"Downloading {path.name}")
            urllib.request.urlretrieve(f"{ZENODO_URL}/{path.name}", path.with_suffix(".part"))
            path.with_suffix(".part").rename(path)

        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        fs = raw.info["sfreq"]
        emg = raw.get_data(picks=[f"EMG {i}" for i in range(1, 9)])
        angles = raw.get_data(picks=[f"Angle {i}" for i in range(1, 16)])

        emg = signal.resample_poly(emg, int(preprocess.fs), int(fs), axis=-1)
        t = np.arange(emg.shape[1]) / preprocess.fs
        angles = np.stack([np.interp(t, np.arange(len(a)) / fs, a) for a in angles])

        emg = preprocess(emg)
        skip = int(SKIP_START * preprocess.fs)
        emg, angles = emg[:, skip:], angles[:, skip:]

        save_recording(recording_path(out_dir, s), emg, angles)
        print(f"Subject {s:02d}: T = {emg.shape[1]} raw samples")


if __name__ == "__main__":
    params = yaml.safe_load(open("src/params.yaml"))
    main(params["preprocessing"], "data/raw/emg_fk", "data/processed/emg_fk")
