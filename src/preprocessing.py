import numpy as np
from scipy import signal


def design_filter(fs, bandpass=None, notch=None, downsample=1):
    sections = []
    if bandpass:
        sections.append(signal.butter(4, bandpass, btype="bandpass", fs=fs, output="sos"))
    if notch:
        b, a = signal.iirnotch(notch, 30, fs=fs)
        sections.append(signal.tf2sos(b, a))
    if downsample > 1:
        sections.append(signal.butter(8, 0.4 * fs / downsample, fs=fs, output="sos"))  # anti-aliasing
    return np.concatenate(sections) if sections else None


def preprocess(emg, fs, bandpass=None, notch=None, downsample=1):
    """[C, T] raw EMG -> [C, T // downsample]. Causal, so it can run the same way online."""
    sos = design_filter(fs, bandpass, notch, downsample)
    if sos is not None:
        emg = signal.sosfilt(sos, emg, axis=-1)
    return emg[:, ::downsample].astype(np.float32)
