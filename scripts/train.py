"""Training Script for Temporal Riemannian Regressor (TRR).

Sections:
1. Configuration & Seeding: Load params.yaml and initialize random seeds.
2. Data Loaders: Construct train, validation, and test loaders from processed dataset.
3. Model Instantiation: Instantiate TRR model with channel and target dimensions.
4. Training & Evaluation: Run optimization with Huber loss and gradient clipping.
"""

import numpy as np
import torch
import yaml
from torch import nn

from src.dataset import make_loaders
from src.model import Model
from src.trainer import set_seed, train

if __name__ == "__main__":
    # 1. Configuration & Seeding
    params = yaml.safe_load(open("params.yaml"))
    cfg = params["train"]
    set_seed(cfg["seed"])

    # 2. Data Loaders
    train_loader, val_loader, test_loader = make_loaders(params)
    info = train_loader.dataset.info

    # 3. Model Instantiation (TRR)
    in_channels = train_loader.dataset.X.shape[1] if hasattr(train_loader.dataset, "X") else len(info["channel_names"])
    model = Model(in_channels, len(info["target_names"]), **params["model"])

    # Initialize output bias to target mean for continuous kinematic regression
    if hasattr(train_loader.dataset, "y") and params["model"].get("activation") != "sigmoid":
        y_all = train_loader.dataset.y
        train_indices = (
            train_loader.dataset.indices if hasattr(train_loader.dataset, "indices") else np.arange(len(y_all))
        )
        y_mean = torch.from_numpy(y_all[train_indices].mean(axis=0)).float()
        model.out.bias.data.copy_(y_mean)

    # 4. Training & Evaluation
    loss_fn = nn.HuberLoss() if cfg.get("loss", "huber") == "huber" else nn.MSELoss()
    grad_clip = cfg.get("grad_clip", 1.0)

    train(
        model,
        train_loader,
        val_loader,
        cfg["out_dir"],
        test_loader,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        grad_clip=grad_clip,
        loss_fn=loss_fn,
    )
