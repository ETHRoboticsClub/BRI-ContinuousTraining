"""Dynamic multimodal span masking matching roamlab/reactemg."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def masking_a2a(
    sequence: np.ndarray,
    mask_percentage: float,
    mask_type: str,
    mask_channel_selection: Sequence[int],
    mask_alignment: str = "aligned",
    lambda_poisson: float = 7.0,
    seeded_mask: bool = False,
    mask_token: float | int = 0,
    transition_index: int = -1,
    transition_buffer: int = 30,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Poisson span, end-of-window, or targeted transition masking to 1D or 2D sequences.

    Args:
        sequence: Array of shape ``(T,)`` (actions) or ``(T, C)`` (EMG).
        mask_percentage: Fraction of timesteps in ``[0, 1]`` to mask for poisson/end modes.
        mask_type: One of ``"poisson"``, ``"end"``, or ``"targeted"``.
        mask_channel_selection: Channel indices to mask (``[0]`` for 1D sequences).
        mask_alignment: ``"aligned"`` shares masked spans across channels; ``"non-aligned"`` samples independently.
        lambda_poisson: Mean Poisson span length for ``mask_type="poisson"``.
        seeded_mask: Ignored when explicit ``rng`` is provided; uses a fresh default Generator when True/False.
        mask_token: Fill value placed in masked positions of ``masked_sequence``.
        transition_index: First transition index in the window for ``mask_type="targeted"``.
        transition_buffer: Half-width in timesteps around ``transition_index`` to mask.
        rng: Optional NumPy random Generator for reproducible tests.

    Returns:
        Tuple ``(masked_sequence, mask_positions)`` matching the shape of ``sequence``.
    """
    arr = np.asarray(sequence)
    is_1d = arr.ndim == 1
    if is_1d:
        arr_2d = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        arr_2d = arr
    else:
        raise ValueError(f"Expected 1D or 2D sequence, got shape {arr.shape}")

    window_size, num_channels = arr_2d.shape
    channels = list(mask_channel_selection)
    if not channels or any(c < 0 or c >= num_channels for c in channels):
        raise ValueError(f"Invalid mask_channel_selection {channels} for {num_channels} channels")

    if mask_type not in ("poisson", "end", "targeted"):
        raise ValueError(f"mask_type must be 'poisson', 'end', or 'targeted', got {mask_type!r}")
    if mask_alignment not in ("aligned", "non-aligned"):
        raise ValueError(f"mask_alignment must be 'aligned' or 'non-aligned', got {mask_alignment!r}")

    if rng is None:
        rng = np.random.default_rng(0) if seeded_mask else np.random.default_rng()

    total_tokens_to_mask = int(window_size * float(mask_percentage))
    total_tokens_to_mask = max(0, min(window_size, total_tokens_to_mask))
    mask_positions = np.zeros_like(arr_2d, dtype=bool)

    if mask_type == "poisson" and total_tokens_to_mask > 0:
        if mask_alignment == "aligned":
            tokens_masked = 0
            while tokens_masked < total_tokens_to_mask:
                span_length = int(rng.poisson(lambda_poisson))
                if span_length <= 0 or span_length >= window_size:
                    continue
                if tokens_masked + span_length > total_tokens_to_mask:
                    span_length = total_tokens_to_mask - tokens_masked
                start_pos = int(rng.integers(0, max(1, window_size - span_length + 1)))
                end_pos = start_pos + span_length
                already = mask_positions[start_pos:end_pos, channels[0]]
                free_rel = np.where(~already)[0]
                if len(free_rel) == 0:
                    continue
                if tokens_masked + len(free_rel) > total_tokens_to_mask:
                    free_rel = free_rel[: total_tokens_to_mask - tokens_masked]
                for ch in channels:
                    mask_positions[start_pos:end_pos, ch][free_rel] = True
                tokens_masked += len(free_rel)
        else:
            for ch in channels:
                tokens_masked = 0
                while tokens_masked < total_tokens_to_mask:
                    span_length = int(rng.poisson(lambda_poisson))
                    if span_length <= 0 or span_length >= window_size:
                        continue
                    if tokens_masked + span_length > total_tokens_to_mask:
                        span_length = total_tokens_to_mask - tokens_masked
                    start_pos = int(rng.integers(0, max(1, window_size - span_length + 1)))
                    end_pos = start_pos + span_length
                    already = mask_positions[start_pos:end_pos, ch]
                    free_rel = np.where(~already)[0]
                    if len(free_rel) == 0:
                        continue
                    if tokens_masked + len(free_rel) > total_tokens_to_mask:
                        free_rel = free_rel[: total_tokens_to_mask - tokens_masked]
                    mask_positions[start_pos:end_pos, ch][free_rel] = True
                    tokens_masked += len(free_rel)
    elif mask_type == "end" and total_tokens_to_mask > 0:
        start_pos = window_size - total_tokens_to_mask
        mask_positions[start_pos:, channels] = True
    elif mask_type == "targeted":
        if transition_index >= 0:
            t_start = max(0, int(transition_index) - int(transition_buffer))
            t_end = min(window_size, int(transition_index) + int(transition_buffer))
            if t_end > t_start:
                mask_positions[t_start:t_end, channels] = True

    masked_sequence = np.where(mask_positions, mask_token, arr_2d).astype(arr.dtype, copy=False)
    if is_1d:
        return masked_sequence.reshape(-1), mask_positions.reshape(-1)
    return masked_sequence, mask_positions
