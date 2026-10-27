import yaml
import mne
import numpy as np
import urllib.request
from scipy import signal
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

from src.preprocess import Preprocess

ZENODO_URL = "https://zenodo.org/records/19453843/files"
SUBJECTS = range(1, 21)
WINDOW = 240
STRIDE = 40
SKIP_START = 0.25

def main(params, raw_dir, out_dir) -> None:
    """Download the recordings into raw_dir, preprocess them and save their windows to out_dir."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess = Preprocess(**params)
    X, y, subject = [], [], []
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
        emg = (emg - emg.mean(1, keepdims=True)) / (emg.std(1, keepdims=True) + 1e-6)

        windows = sliding_window_view(emg, WINDOW, axis=1)[:, ::STRIDE].transpose(1, 0, 2)
        X.append(windows)
        y.append(angles[:, WINDOW - 1 :: STRIDE].T)
        subject.append(np.full(len(windows), s))
        print(f"Subject {s:02d}: {len(windows)} windows")

    np.save(out_dir / "X.npy", np.concatenate(X).astype(np.float32))
    np.save(out_dir / "y.npy", np.concatenate(y).astype(np.float32))
    np.save(out_dir / "subject.npy", np.concatenate(subject))


if __name__ == "__main__":
    params = yaml.safe_load(open("src/params.yaml"))
    main(params["preprocessing"], "data/raw/emg_fk", "data/processed/emg_fk")