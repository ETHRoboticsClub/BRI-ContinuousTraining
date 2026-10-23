"""EMG-FK Dataset Ingestion, Synchronization, and Downsampling.

This script imports recordings from the EMG Finger-Kinematics (EMG-FK) dataset (Colot et al., 2026),
downsamples the 500 Hz signals to 200 Hz, aligns EMG and kinematics, and formats the data into
standard .npz recording files in data/raw/emg_fk/.

Sections:
1. Constants & Configuration: Channel counts, default frequencies, and synchronization parameters.
2. Ingestion Helpers (FIF and NPZ): Loading recordings and channel extraction.
3. Signal Synchronization: Hilbert move command decomposition & kinematic velocity correlation.
4. Downsampling (500 Hz -> 200 Hz): Rational polyphase resampling with anti-aliasing.
5. Export Pipeline: Converting and saving recordings into data/raw/emg_fk/*.npz format.
"""

import argparse
from math import gcd
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# ==============================================================================
# Section 1: Constants & Dataset Configuration
# ==============================================================================

DEFAULT_EMG_CHANNELS: int = 8
DEFAULT_KINEMATIC_ANGLES: int = 15
DEFAULT_EMG_S_FREQ: float = 500.0
DEFAULT_TARGET_FREQ: float = 200.0

SYNC_BANDPASS_ORDER: int = 4
SYNC_BANDPASS_RANGE: Tuple[float, float] = (15.0, 50.0)
SYNC_GAUSSIAN_SMOOTHING_SIGMA: float = 20.0
SYNC_SEARCH_PAST_RATIO: float = 0.2
SYNC_SEARCH_FUTURE_RATIO: float = 0.5
SYNC_SEARCH_STEP_RATIO: float = 0.01


# ==============================================================================
# Section 2: Ingestion Helpers (FIF and NPZ)
# ==============================================================================


def load_recording(file_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load EMG signals and finger joint angle kinematics from a recording file.

    Accomplishes:
    Supports MNE raw FIF files (.fif) and NumPy (.npz) files, extracting 8-channel
    forearm EMG and 15 finger joint angles.

    Args:
        file_path: Path to the recording file (.fif or .npz).

    Returns:
        Tuple of (emg, angles, fs) where emg is [8, T], angles is [15, T],
        and fs is the recording sampling rate in Hz.

    Raises:
        ImportError: If reading .fif and mne is not installed.
        ValueError: If file extension is unsupported.
    """
    if file_path.suffix == ".fif":
        try:
            import mne
        except ImportError as e:
            raise ImportError("mne is required to read .fif files. Install with 'pip install mne'.") from e
        raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose="ERROR")
        emg_picks: List[str] = [f"EMG {i+1}" for i in range(DEFAULT_EMG_CHANNELS) if f"EMG {i+1}" in raw.ch_names]
        angle_picks: List[str] = [
            f"Angle {i+1}" for i in range(DEFAULT_KINEMATIC_ANGLES) if f"Angle {i+1}" in raw.ch_names
        ]
        emg: np.ndarray = raw.get_data(picks=emg_picks).astype(np.float32)  # [8, T]
        angles: np.ndarray = raw.get_data(picks=angle_picks).astype(np.float32)  # [15, T]
        fs: float = float(raw.info["sfreq"])
        return emg, angles, fs
    elif file_path.suffix == ".npz":
        with np.load(file_path) as data:
            emg = data["emg"].astype(np.float32)
            angles = data["targets"].astype(np.float32) if "targets" in data else data["angles"].astype(np.float32)
            fs = float(data.get("fs", DEFAULT_EMG_S_FREQ))
        return emg, angles, fs
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


# ==============================================================================
# Section 3: Synchronization (Hilbert Move Command vs Kinematic Speed)
# ==============================================================================


def synchronize_signals(
    emg: np.ndarray,
    angles: np.ndarray,
    fs: float = DEFAULT_EMG_S_FREQ,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synchronize EMG and kinematic signals using move command cross-correlation.

    Accomplishes:
    Implements the automatic synchronization procedure described in Section II-C and Figure 2
    of Colot et al. (2026):
    1. Computes the envelope of bandpassed EMG via Hilbert transform.
    2. Decomposes envelope into move (M) and hold (H) commands.
    3. Differentiates smoothed finger joint angles to compute velocity.
    4. Finds the temporal shift maximizing correlation between maximum EMG move command
       and maximum joint speed.

    Args:
        emg: Raw EMG array of shape [channels, time_steps].
        angles: Kinematic joint angles array of shape [joints, time_steps].
        fs: Sampling rate of the recordings in Hz. Defaults to 500.0.

    Returns:
        Tuple of (aligned_emg, aligned_angles) cropped to the optimal synchronized alignment.
    """
    n_samples: int = min(emg.shape[1], angles.shape[1])
    emg = emg[:, :n_samples]
    angles = angles[:, :n_samples]

    # Filter for move command envelope
    sos: np.ndarray = signal.butter(
        SYNC_BANDPASS_ORDER, list(SYNC_BANDPASS_RANGE), btype="bandpass", fs=fs, output="sos"
    )
    emg_bp: np.ndarray = signal.sosfilt(sos, emg, axis=-1)
    analytic: np.ndarray = signal.hilbert(emg_bp, axis=-1)
    envelope: np.ndarray = np.abs(analytic)

    # Move / Hold decomposition
    move: np.ndarray = np.zeros_like(envelope)
    hold: np.ndarray = np.zeros_like(envelope)
    for i in range(1, n_samples):
        move[:, i] = envelope[:, i] - hold[:, i - 1]
        hold[:, i] = hold[:, i - 1] + move[:, i] / fs

    # Kinematic speed across joints
    smooth_angles: np.ndarray = gaussian_filter1d(angles, sigma=SYNC_GAUSSIAN_SMOOTHING_SIGMA, axis=-1)
    speed: np.ndarray = np.abs(np.gradient(smooth_angles, axis=-1))
    smooth_speed: np.ndarray = gaussian_filter1d(speed, sigma=SYNC_GAUSSIAN_SMOOTHING_SIGMA, axis=-1)
    max_speed: np.ndarray = np.max(smooth_speed, axis=0)

    smooth_move: np.ndarray = gaussian_filter1d(move, sigma=SYNC_GAUSSIAN_SMOOTHING_SIGMA, axis=-1)
    max_move: np.ndarray = np.max(smooth_move, axis=0)

    # Search for optimal temporal shift
    shifts: np.ndarray = np.arange(
        -int(SYNC_SEARCH_PAST_RATIO * fs),
        int(SYNC_SEARCH_FUTURE_RATIO * fs),
        step=max(1, int(SYNC_SEARCH_STEP_RATIO * fs)),
    )
    best_shift: int = 0
    max_corr: float = -1.0
    for s in shifts:
        if s >= 0:
            a: np.ndarray = max_move[: n_samples - s]
            b: np.ndarray = max_speed[s:]
        else:
            a = max_move[-s:]
            b = max_speed[: n_samples + s]
        if len(a) > 0 and len(b) > 0:
            c: float = float(np.corrcoef(a, b)[0, 1])
            if c > max_corr:
                max_corr = c
                best_shift = int(s)

    # Apply shift to align signals
    if best_shift > 0:
        aligned_emg: np.ndarray = emg[:, : n_samples - best_shift]
        aligned_angles: np.ndarray = angles[:, best_shift:n_samples]
    elif best_shift < 0:
        aligned_emg = emg[:, -best_shift:n_samples]
        aligned_angles = angles[:, : n_samples + best_shift]
    else:
        aligned_emg = emg
        aligned_angles = angles

    return aligned_emg, aligned_angles


# ==============================================================================
# Section 4: Downsampling (500 Hz -> 200 Hz)
# ==============================================================================


def downsample_recording(
    emg: np.ndarray,
    angles: np.ndarray,
    fs_in: float = DEFAULT_EMG_S_FREQ,
    fs_out: float = DEFAULT_TARGET_FREQ,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample EMG and kinematic angles from fs_in to fs_out.

    Accomplishes:
    1. Resamples EMG using rational polyphase resampling (up=2, down=5 for 500->200 Hz).
       The polyphase filter applies an anti-aliasing lowpass filter with cutoff at the new
       Nyquist rate (100 Hz), completely preventing aliasing artifacts.
    2. Resamples kinematic joint angles at the identical 200 Hz timestamps via linear interpolation.

    Args:
        emg: Multi-channel EMG signal array of shape [channels, time_steps].
        angles: Kinematic joint angles array of shape [joints, time_steps].
        fs_in: Original sampling frequency in Hz. Defaults to 500.0.
        fs_out: Target sampling frequency in Hz. Defaults to 200.0.

    Returns:
        Tuple of (emg_resampled, angles_resampled) both sampled at fs_out.
    """
    if fs_in == fs_out:
        return emg, angles

    up: int = int(fs_out)
    down: int = int(fs_in)
    divisor: int = gcd(up, down)
    up //= divisor
    down //= divisor

    # Polyphase rational resampling for EMG
    emg_resampled: np.ndarray = signal.resample_poly(emg, up=up, down=down, axis=-1).astype(np.float32)
    n_samples_out: int = emg_resampled.shape[1]

    # Time coordinates
    t_in: np.ndarray = np.arange(angles.shape[1]) / fs_in
    t_out: np.ndarray = np.arange(n_samples_out) / fs_out

    # Linear interpolation for continuous kinematics
    angles_resampled: np.ndarray = np.stack(
        [np.interp(t_out, t_in, angles[d]) for d in range(angles.shape[0])], axis=0
    ).astype(np.float32)

    return emg_resampled, angles_resampled


# ==============================================================================
# Section 5: Export Pipeline
# ==============================================================================


def import_emg_fk_dataset(
    source_dir: Path,
    out_dir: Path,
    target_fs: float = DEFAULT_TARGET_FREQ,
    sync: bool = False,
) -> None:
    """Import and process all EMG-FK files into pipeline .npz format.

    Accomplishes:
    Scans source_dir for recordings, applies optional synchronization and 500->200 Hz
    downsampling, and outputs standardized NPZ files partitioned by subject and session/trial.

    Args:
        source_dir: Directory containing raw EMG-FK recordings (.fif or .npz).
        out_dir: Destination directory for standardized NPZ files (data/raw/emg_fk).
        target_fs: Target sampling frequency in Hz. Defaults to 200.0.
        sync: Whether to run the Hilbert cross-correlation synchronization procedure. Defaults to False.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(list(source_dir.glob("*.fif")) + list(source_dir.glob("*.npz")))
    if not files:
        print(f"No .fif or .npz files found in {source_dir}")
        return

    print(f"Found {len(files)} files to import into {out_dir} at {target_fs} Hz...")
    for idx, path in enumerate(files):
        emg, angles, fs = load_recording(path)
        if sync:
            emg, angles = synchronize_signals(emg, angles, fs=fs)

        # Downsample to target_fs (e.g. 500 Hz -> 200 Hz)
        emg_ds, angles_ds = downsample_recording(emg, angles, fs_in=fs, fs_out=target_fs)

        # Subject metadata (e.g. Subject_01 -> subject 1)
        sub_id: int = idx + 1
        stem: str = path.stem.lower()
        if "subject" in stem:
            digits: str = "".join(filter(str.isdigit, stem))
            if digits:
                sub_id = int(digits)

        # Split recording into trial 1 (first half) and trial 2 (second half) as in paper
        half: int = emg_ds.shape[1] // 2
        for trial, (start, end) in enumerate([(0, half), (half, emg_ds.shape[1])], start=1):
            out_file: Path = out_dir / f"subject_{sub_id:02d}_trial_{trial:02d}.npz"
            np.savez(
                out_file,
                emg=emg_ds[:, start:end],
                fs=float(target_fs),
                channel_names=[f"EMG_{i+1}" for i in range(emg_ds.shape[0])],
                targets=angles_ds[:, start:end],
                target_names=[f"Angle_{i+1}" for i in range(angles_ds.shape[0])],
                meta_subject=sub_id,
                meta_session=1,
                meta_trial=trial,
            )

        print(f"Imported {path.name}: {emg_ds.shape[1]} samples @ {target_fs}Hz -> {out_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import EMG-FK dataset with downsampling to 200 Hz.")
    parser.add_argument("--source_dir", type=str, default="data/raw/emg_fk_source", help="Path to raw EMG-FK files")
    parser.add_argument(
        "--out_dir", type=str, default="data/raw/emg_fk", help="Destination path for pipeline NPZ files"
    )
    parser.add_argument(
        "--target_fs",
        type=float,
        default=DEFAULT_TARGET_FREQ,
        help=f"Target sampling rate (default: {DEFAULT_TARGET_FREQ} Hz)",
    )
    parser.add_argument("--sync", action="store_true", help="Run automatic synchronization procedure")
    args = parser.parse_args()

    import_emg_fk_dataset(
        source_dir=Path(args.source_dir),
        out_dir=Path(args.out_dir),
        target_fs=args.target_fs,
        sync=args.sync,
    )
