import numpy as np
from scipy import signal


class Preprocess:
    """Causal EMG filtering: power-line notch followed by a Butterworth bandpass."""

    def __init__(self, fs, bandpass, notch) -> None:
        self.fs = fs
        self.notch = signal.tf2sos(*signal.iirnotch(notch, Q=30, fs=fs))
        self.bandpass = signal.butter(4, bandpass, btype="bandpass", fs=fs, output="sos")

    def __call__(self, emg) -> np.ndarray:
        """Filter raw EMG [C, T] sampled at self.fs."""
        emg = signal.sosfilt(self.notch, emg, axis=-1)
        return signal.sosfilt(self.bandpass, emg, axis=-1).astype(np.float32)
