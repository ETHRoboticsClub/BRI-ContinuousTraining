import numpy as np
from scipy import signal


class Preprocess:
    """Causal EMG filtering: power-line notch followed by a Butterworth bandpass or 3-band filter bank."""

    def __init__(
        self,
        fs: int = 200,
        bandpass: tuple[float, float] | list[float] = (5, 95),
        notch: float = 50,
        bands: tuple[tuple[float, float], ...] = ((5, 40), (40, 70), (70, 95)),
    ) -> None:
        self.fs = fs
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.notch = signal.tf2sos(*signal.iirnotch(notch, Q=30, fs=fs))
        self.bandpass = signal.butter(4, bandpass, btype="bandpass", fs=fs, output="sos")
        self.sos_bands = [signal.butter(4, b, btype="bandpass", fs=fs, output="sos") for b in self.bands]

    def __call__(self, emg: np.ndarray) -> np.ndarray:
        """Filter raw EMG [C, T] sampled at self.fs."""
        emg = signal.sosfilt(self.notch, emg, axis=-1)
        return signal.sosfilt(self.bandpass, emg, axis=-1).astype(np.float32)

    def filter_bands(self, emg: np.ndarray) -> np.ndarray:
        """Filter raw EMG [C, T] into 3 sub-Nyquist frequency bands [3, C, T]."""
        notched = signal.sosfilt(self.notch, emg, axis=-1)
        return np.stack([signal.sosfilt(sos, notched, axis=-1).astype(np.float32) for sos in self.sos_bands], axis=0)
