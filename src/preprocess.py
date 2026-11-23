"""Signal conditioning for WaveFormer (bipolar signed sEMG)."""

from __future__ import annotations

import numpy as np

LEFT_HAND_REMAP: tuple[int, ...] = (6, 5, 4, 3, 2, 1, 0, 7)


def preprocess_waveformer_window(
    emg: np.ndarray,
    scale_divisor: float = 128.0,
    hand_choice: str = "right",
    remove_dc: bool = True,
    channels: int = 8,
    return_nchw: bool = False,
) -> np.ndarray:
    """Condition an sEMG sequence for WaveFormer while preserving bipolar phase & amplitude.

    Accepts ``(T, C)`` or ``(C, T)`` raw int8/float32 Myo sEMG, applies optional
    left-hand electrode remapping, per-channel DC centering (subtracting mean so
    oscillatory AC contraction RMS is preserved without blowing up baseline rest
    noise), and divides by ``scale_divisor`` (``128.0`` for Myo int8 units).

    Returns ``(T, C)`` float32 by default, or ``(1, C, T)`` if ``return_nchw=True``.
    """
    arr = np.asarray(emg, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D EMG array, got shape {arr.shape}")

    if arr.shape[1] == channels:
        emg_tc = arr.copy()
    elif arr.shape[0] == channels:
        emg_tc = arr.T.copy()
    else:
        raise ValueError(f"Neither axis matches channels={channels}: got shape {arr.shape}")

    choice = str(hand_choice).lower()
    if choice not in ("right", "left"):
        raise ValueError(f"hand_choice must be 'right' or 'left', got {hand_choice!r}")
    if choice == "left" and channels == 8:
        emg_tc = emg_tc[:, list(LEFT_HAND_REMAP)]

    if remove_dc:
        emg_tc = emg_tc - np.mean(emg_tc, axis=0, keepdims=True)

    emg_tc = (emg_tc / float(scale_divisor)).astype(np.float32, copy=False)
    if return_nchw:
        return emg_tc.T[np.newaxis, :, :].astype(np.float32, copy=False)
    return emg_tc


class Preprocess:
    """sEMG signal conditioning for WaveFormer."""

    def __init__(
        self,
        fs: int = 200,
        channels: int = 8,
        mode: str = "waveformer",
        scale_divisor: float = 128.0,
        hand_choice: str = "right",
        remove_dc: bool = True,
        **_kwargs: object,
    ) -> None:
        if mode != "waveformer":
            raise ValueError(f"mode must be 'waveformer', got {mode!r}")
        if hand_choice not in ("right", "left"):
            raise ValueError(f"hand_choice must be 'right' or 'left', got {hand_choice!r}")
        self.fs = int(fs)
        self.channels = int(channels)
        self.mode = str(mode)
        self.scale_divisor = float(scale_divisor)
        self.hand_choice = str(hand_choice)
        self.remove_dc = bool(remove_dc)

    def preprocess_sequence(self, emg_tc: np.ndarray) -> np.ndarray:
        """Condition an EMG sequence of shape ``[T, C]`` and return ``[T, C]`` float32."""
        arr = np.asarray(emg_tc, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.channels:
            raise ValueError(f"Expected [T, {self.channels}] EMG array, got shape {arr.shape}")

        return preprocess_waveformer_window(
            arr,
            scale_divisor=self.scale_divisor,
            hand_choice=self.hand_choice,
            remove_dc=self.remove_dc,
            channels=self.channels,
            return_nchw=False,
        )

    def apply(self, emg: np.ndarray) -> np.ndarray:
        """Alias for :meth:`__call__`."""
        return self.__call__(emg)

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


class WaveFormerPreprocess(Preprocess):
    """Explicit WaveFormer bipolar sEMG preprocessor (signed ``/128.0`` + DC removal)."""

    def __init__(
        self,
        fs: int = 200,
        channels: int = 8,
        scale_divisor: float = 128.0,
        hand_choice: str = "right",
        remove_dc: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(
            fs=fs,
            channels=channels,
            mode="waveformer",
            scale_divisor=scale_divisor,
            hand_choice=hand_choice,
            remove_dc=remove_dc,
            **kwargs,
        )
