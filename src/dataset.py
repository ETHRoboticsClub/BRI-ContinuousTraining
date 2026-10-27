from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EMGFK(Dataset):
    """Sliding windows of the EMG-FK dataset (Colot et al., 2026): 8 EMG channels -> 15 finger joint angles."""

    def __init__(self, data_dir, subjects) -> None:
        self.data_dir = Path(data_dir)
        mask = np.isin(np.load(self.data_dir / "subject.npy"), subjects)
        self.indices = np.flatnonzero(mask)
        self.X = np.load(self.data_dir / "X.npy", mmap_mode="r")
        self.y = np.load(self.data_dir / "y.npy")[mask]

    def load(self):
        pass

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.array(self.X[self.indices[i]])), torch.from_numpy(self.y[i])
