import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.model.tangent_space import n_steps


def recording_path(data_dir, subject: int) -> Path:
    return Path(data_dir) / f"subject_{subject:02d}.npz"


def save_recording(path, emg: np.ndarray, angles: np.ndarray) -> None:
    """Store one whole recording: emg [8, T] and angles [15, T], float32, T in raw samples (5 ms at 200 Hz)."""
    np.savez(path, emg=emg.astype(np.float32), angles=angles.astype(np.float32))


def load_recording(path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as f:
        return f["emg"], f["angles"]


def shortest_recording(data_dir) -> int:
    """Minimum recording length T, in raw samples, over all subject_*.npz in data_dir."""
    return min(load_recording(p)[0].shape[1] for p in sorted(Path(data_dir).glob("subject_*.npz")))


def kernel_length(k: int, shortest: int, stateful_input: int = 240) -> int:
    """Number of encoder steps in a kernel-training example: k stateful inputs, capped by the shortest recording."""
    steps = k * n_steps(stateful_input)
    cap = n_steps(shortest)
    if steps > cap:
        warnings.warn(f"k={k} needs {steps} steps but the shortest recording has {cap}; using {cap}", stacklevel=2)
    return min(steps, cap)


class EMGFK(Dataset):
    """Training examples of the EMG-FK dataset (Colot et al., 2026): 8 EMG channels -> 15 finger joint angles.

    An example is `length` raw samples of EMG ending at raw sample e, and its target is the angles at e. Consecutive
    examples of a recording end `stride` raw samples apart, the first at e = length - 1.
    """

    channels = 8
    targets = 15

    def __init__(self, data_dir, subjects, length: int, stride: int) -> None:
        self.length = length
        self.stride = stride
        self.emg, self.angles = [], []
        self.index = []  # (position in self.emg, end sample)
        for pos, s in enumerate(subjects):
            emg, angles = load_recording(recording_path(data_dir, s))
            self.emg.append(emg)
            self.angles.append(angles)
            count = (emg.shape[1] - length) // stride + 1 if emg.shape[1] >= length else 0
            self.index += [(pos, length - 1 + j * stride) for j in range(count)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        pos, e = self.index[i]
        x = self.emg[pos][:, e - self.length + 1 : e + 1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.angles[pos][:, e].copy())
