"""Signal conditioning for ReactEMG (median filter, rectification, 1/128 scaling)."""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

LEFT_HAND_REMAP: tuple[int, ...] = (6, 5, 4, 3, 2, 1, 0, 7)


class Preprocess:
    """ReactEMG time-domain sEMG conditioning without bandpass or z-score normalization.

    Applies optional left-hand electrode remapping, per-channel 1D median filtering,
    full-wave rectification, and division by `scale_divisor` (128.0 for int8 Myo raw units).
    """

    def __init__(
        self,
        fs: int = 200,
        channels: int = 8,
        median_filter_size: int = 3,
        medfilt_order: str = "before_rec",
        scale_divisor: float = 128.0,
        hand_choice: str = "right",
        **_kwargs: object,
    ) -> None:
        if medfilt_order not in ("before_rec", "after_rec"):
            raise ValueError(f"medfilt_order must be 'before_rec' or 'after_rec', got {medfilt_order!r}")
        if hand_choice not in ("right", "left"):
            raise ValueError(f"hand_choice must be 'right' or 'left', got {hand_choice!r}")
        self.fs = int(fs)
        self.channels = int(channels)
        self.median_filter_size = int(median_filter_size)
        self.medfilt_order = medfilt_order
        self.scale_divisor = float(scale_divisor)
        self.hand_choice = hand_choice

    def preprocess_sequence(self, emg_tc: np.ndarray) -> np.ndarray:
        """Condition an EMG sequence of shape ``[T, C]`` and return ``[T, C]`` float32 in ``[0, 1]``."""
        arr = np.asarray(emg_tc, dtype=np.float32).copy()
        if arr.ndim != 2 or arr.shape[1] != self.channels:
            raise ValueError(f"Expected [T, {self.channels}] EMG array, got shape {arr.shape}")

        if self.hand_choice == "left" and self.channels == 8:
            arr = arr[:, list(LEFT_HAND_REMAP)]

        k = self.median_filter_size
        if self.medfilt_order == "before_rec":
            if k > 1:
                for ch in range(self.channels):
                    arr[:, ch] = medfilt(arr[:, ch], kernel_size=k)
            arr = np.abs(arr)
        else:
            arr = np.abs(arr)
            if k > 1:
                for ch in range(self.channels):
                    arr[:, ch] = medfilt(arr[:, ch], kernel_size=k)

        return (arr / self.scale_divisor).astype(np.float32, copy=False)

    def __call__(self, emg: np.ndarray) -> np.ndarray:
        """Condition EMG shaped either ``[T, C]`` or ``[C, T]``, preserving input axis layout."""
        arr = np.asarray(emg, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D EMG array, got shape {arr.shape}")
        if arr.shape[1] == self.channels:
            return self.preprocess_sequence(arr)
        if arr.shape[0] == self.channels:
            return self.preprocess_sequence(arr.T).T
        raise ValueError(f"Neither axis matches channels={self.channels}: got shape {arr.shape}")
