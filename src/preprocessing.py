"""EMG Preprocessing Module.

This module provides causal signal conditioning routines for electromyography (EMG) signals,
tailored for real-time kinematic regression as described in Colot et al. (2026).

Public API:
- preprocess(): Unified causal entry point for dataset building and real-time inference.

Internal Sections:
1. Constants & Configuration: Filter orders, quality factors, and default multiband frequency sets.
2. Internal Filter Design: Second-order sections (SOS) for bandpass, power-line notch, and anti-aliasing.
3. Internal Multi-Band Decomposition: Frequency-band decomposition into distinct spectral bands.
4. Internal Downsampling Utilities: Rational polyphase resampling and decimation from 500 Hz to 200 Hz.
5. Public Preprocessing Pipeline: Main preprocess() function.
"""

from math import gcd
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal

__all__ = ["preprocess"]

# ==============================================================================
# Section 1: Constants & Filter Configuration
# ==============================================================================

DEFAULT_NOTCH_Q: float = 30.0
NYQUIST_RATIO: float = 0.5
DEFAULT_BANDPASS_ORDER: int = 4
DEFAULT_ANTI_ALIASING_ORDER: int = 8
ANTI_ALIASING_CUTOFF_RATIO: float = 0.4
MAX_ANTI_ALIASING_CUTOFF_RATIO: float = 0.45
NYQUIST_MARGIN_HZ: float = 1e-2
DEFAULT_DOWNSAMPLE_FACTOR: float = 1.0

# Standard multi-band frequency decompositions from Colot et al. (2026)
DEFAULT_MULTIBAND_BANDS_200HZ: Tuple[Tuple[float, float], ...] = (
    (15.0, 35.0),
    (35.0, 65.0),
    (65.0, 95.0),
)
DEFAULT_MULTIBAND_BANDS_500HZ: Tuple[Tuple[float, float], ...] = (
    (15.0, 40.0),
    (40.0, 80.0),
    (80.0, 150.0),
)


# ==============================================================================
# Section 2: Internal Filter Design Helpers
# ==============================================================================


def _design_notch_filter(
    fs: float,
    notch_freqs: Union[float, Sequence[float]],
    q: float = DEFAULT_NOTCH_Q,
) -> Optional[np.ndarray]:
    """Design second-order sections (SOS) for one or more IIR notch filters.

    Accomplishes:
    Removes power-line interference (e.g. 50 Hz and harmonics) while preserving
    surrounding EMG frequency content.

    Args:
        fs: Sampling frequency of the signal in Hz.
        notch_freqs: Center notch frequency in Hz (e.g., 50.0) or sequence of notch
            frequencies (e.g., [50.0, 100.0]). Frequencies above Nyquist are skipped.
        q: Quality factor determining notch sharpness (Q = w0 / bandwidth). Defaults to 30.0.

    Returns:
        Concatenated SOS filter matrix of shape [n_sections, 6] if any valid notch
        frequencies exist below Nyquist, otherwise None.
    """
    if isinstance(notch_freqs, (int, float)):
        freqs: List[float] = [float(notch_freqs)]
    else:
        freqs = [float(f) for f in notch_freqs]
    sections: List[np.ndarray] = []
    nyquist: float = NYQUIST_RATIO * fs
    for f in freqs:
        if 0.0 < f < nyquist:
            b, a = signal.iirnotch(w0=f, Q=q, fs=fs)
            sections.append(signal.tf2sos(b, a))
    return np.concatenate(sections) if sections else None


def _design_singleband_filter(
    fs: float,
    bandpass: Optional[Sequence[float]] = None,
    notch: Optional[Union[float, Sequence[float]]] = None,
    downsample: float = DEFAULT_DOWNSAMPLE_FACTOR,
) -> Optional[np.ndarray]:
    """Design a cascade of causal SOS filters for bandpass, notch, and anti-aliasing.

    Accomplishes:
    Constructs a single combined SOS filter cascade executing bandpass filtering,
    power-line notch filtering, and anti-aliasing lowpass filtering in a single pass.

    Args:
        fs: Sampling rate of the raw signal in Hz.
        bandpass: Optional [low, high] frequency cutoffs in Hz for Butterworth bandpass filtering.
        notch: Optional frequency or list of frequencies in Hz to reject with notch filters.
        downsample: Downsampling decimation factor. If greater than 1, an anti-aliasing lowpass
            filter is added before decimation. Defaults to 1.0.

    Returns:
        Combined SOS filter matrix of shape [n_sections, 6] representing the full cascade,
        or None if no filters were specified.
    """
    sections: List[np.ndarray] = []
    if bandpass:
        sections.append(signal.butter(DEFAULT_BANDPASS_ORDER, bandpass, btype="bandpass", fs=fs, output="sos"))
    if notch:
        notch_sos = _design_notch_filter(fs=fs, notch_freqs=notch)
        if notch_sos is not None:
            sections.append(notch_sos)
    if downsample > DEFAULT_DOWNSAMPLE_FACTOR:
        cutoff: float = min(ANTI_ALIASING_CUTOFF_RATIO * fs / downsample, MAX_ANTI_ALIASING_CUTOFF_RATIO * fs)
        sections.append(signal.butter(DEFAULT_ANTI_ALIASING_ORDER, cutoff, fs=fs, output="sos"))
    return np.concatenate(sections) if sections else None


# ==============================================================================
# Section 3: Internal Multi-Band Frequency Decomposition
# ==============================================================================


def _design_multiband_filters(
    fs: float,
    bands: Sequence[Tuple[float, float]],
    order: int = DEFAULT_BANDPASS_ORDER,
) -> List[np.ndarray]:
    """Design separate SOS bandpass filters for each specified frequency band.

    Accomplishes:
    Splits the EMG signal into multiple spectral bands (e.g., [15, 35], [35, 65], [65, 95] Hz
    for 200 Hz data, or [15, 40], [40, 80], [80, 150] Hz for 500 Hz data) to extract multi-band
    Riemannian covariance features (CMTS) as detailed in Colot et al. (2026).

    Args:
        fs: Sampling frequency in Hz.
        bands: Sequence of (low_freq, high_freq) tuples specifying each band in Hz.
        order: Filter order for each Butterworth bandpass filter. Defaults to 4.

    Returns:
        List of SOS arrays, each of shape [n_sections, 6], one per frequency band.

    Raises:
        ValueError: If any band has low_freq >= high_freq, low_freq <= 0, or exceeds Nyquist.
    """
    nyquist: float = NYQUIST_RATIO * fs
    filters: List[np.ndarray] = []
    for low, high in bands:
        if high >= nyquist:
            high = nyquist - NYQUIST_MARGIN_HZ
        if low >= high or low <= 0.0:
            raise ValueError(f"Invalid frequency band [{low}, {high}] for fs={fs} Hz.")
        sos: np.ndarray = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
        filters.append(sos)
    return filters


# ==============================================================================
# Section 4: Internal Downsampling Utilities (500 Hz -> 200 Hz)
# ==============================================================================


def _resample_emg(emg: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample multi-channel EMG from fs_in to fs_out using rational polyphase resampling.

    Accomplishes:
    Provides exact rational rate conversion (e.g. 500 Hz -> 200 Hz with up=2, down=5)
    with zero-phase FIR anti-aliasing filtering, mapping N samples at 500 Hz to
    N * 2 // 5 samples at 200 Hz.

    Args:
        emg: Multi-channel EMG signal array of shape [channels, time_steps].
        fs_in: Source sampling frequency in Hz.
        fs_out: Target sampling frequency in Hz.

    Returns:
        Resampled EMG array of shape [channels, time_steps_resampled] as float32.
    """
    if fs_in == fs_out:
        return emg
    up: int = int(fs_out)
    down: int = int(fs_in)
    divisor: int = gcd(up, down)
    up //= divisor
    down //= divisor
    return signal.resample_poly(emg, up=up, down=down, axis=-1).astype(np.float32)


# ==============================================================================
# Section 5: Public Preprocessing Pipeline
# ==============================================================================


def preprocess(
    emg: np.ndarray,
    fs: float,
    bandpass: Optional[Sequence[float]] = None,
    notch: Optional[Union[float, Sequence[float]]] = None,
    downsample: float = DEFAULT_DOWNSAMPLE_FACTOR,
    bands: Optional[Sequence[Tuple[float, float]]] = None,
    target_fs: Optional[float] = None,
    multiband: Optional[bool] = None,
) -> np.ndarray:
    """Preprocess multi-channel raw EMG signals.

    Accomplishes:
    1. Applies power-line notch filtering (e.g. 50 Hz).
    2. Performs downsampling / decimation to target sampling rate (e.g. 500 Hz -> 200 Hz).
    3. Toggles between multi-band decomposition [C * len(bands), T] and single-band processing [C, T]
       via the optional `multiband` tag or `bands` parameter.
    4. Ensures all filtering is strictly causal with signal.sosfilt for real-time online use.

    Args:
        emg: Raw EMG array of shape [channels, time_steps].
        fs: Sampling rate of the input EMG in Hz.
        bandpass: Optional [low, high] frequency cutoffs for single-band mode in Hz.
        notch: Notch frequency or list of frequencies in Hz (e.g., 50 or [50, 100]).
        downsample: Decimation factor (integer or float). Defaults to 1.0.
        bands: Optional list of frequency band tuples [(f_low, f_high), ...] in Hz.
        target_fs: Optional explicit target sampling rate in Hz (e.g., 200.0).
        multiband: Optional flag to explicitly switch between multi-band decomposition (True)
            and single-band processing (False). If None (default), multi-band is activated whenever
            `bands` is provided and non-empty.

    Returns:
        Processed EMG array of shape [channels_out, time_steps_out] as float32.
    """
    # 1. Notch filtering (causal)
    if notch:
        notch_sos: Optional[np.ndarray] = _design_notch_filter(fs=fs, notch_freqs=notch)
        if notch_sos is not None:
            emg = signal.sosfilt(notch_sos, emg, axis=-1)

    # 2. Downsampling / Resampling
    effective_fs: float = fs
    if target_fs is not None and target_fs != fs:
        emg = _resample_emg(emg=emg, fs_in=fs, fs_out=target_fs)
        effective_fs = target_fs
    elif downsample > DEFAULT_DOWNSAMPLE_FACTOR:
        if downsample == int(downsample):
            # Causal IIR anti-aliasing filter before decimation
            aa_cutoff: float = ANTI_ALIASING_CUTOFF_RATIO * fs / downsample
            aa_sos: np.ndarray = signal.butter(DEFAULT_ANTI_ALIASING_ORDER, aa_cutoff, fs=fs, output="sos")
            emg = signal.sosfilt(aa_sos, emg, axis=-1)
            ds_int: int = int(downsample)
            emg = emg[:, ::ds_int]
            effective_fs = fs / ds_int
        else:
            # Rational resampling
            new_fs: float = fs / downsample
            emg = _resample_emg(emg=emg, fs_in=fs, fs_out=new_fs)
            effective_fs = new_fs

    # 3. Frequency band filtering (causal)
    # Determine whether to execute in multi-band mode or single-band mode
    use_multiband: bool = multiband if multiband is not None else (bands is not None and len(bands) > 0)

    if use_multiband:
        active_bands: Sequence[Tuple[float, float]] = (
            bands
            if bands
            else (DEFAULT_MULTIBAND_BANDS_200HZ if effective_fs <= 250.0 else DEFAULT_MULTIBAND_BANDS_500HZ)
        )
        band_filters: List[np.ndarray] = _design_multiband_filters(fs=effective_fs, bands=active_bands)
        band_signals: List[np.ndarray] = [signal.sosfilt(sos, emg, axis=-1) for sos in band_filters]
        emg_out: np.ndarray = np.concatenate(band_signals, axis=0)
    elif bandpass:
        bp_sos: np.ndarray = signal.butter(
            DEFAULT_BANDPASS_ORDER, bandpass, btype="bandpass", fs=effective_fs, output="sos"
        )
        emg_out = signal.sosfilt(bp_sos, emg, axis=-1)
    else:
        emg_out = emg

    return emg_out.astype(np.float32)
