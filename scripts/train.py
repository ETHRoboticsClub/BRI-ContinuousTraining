import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import EMGFK
from src.model.model import Model
from src.trainer import evaluate, fit, save_history

DATA_DIR = Path("data/processed/emg_fk")
OUT_DIR = Path("output")


def load_data(params):
    """Cross-subject split: val/test subjects are held out, every other subject is used for training."""
    held_out = params["val_subjects"] + params["test_subjects"]
    train_subjects = [s for s in np.unique(np.load(DATA_DIR / "subject.npy")) if s not in held_out]
    train_set = EMGFK(DATA_DIR, train_subjects)
    train_loader = DataLoader(train_set, params["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(EMGFK(DATA_DIR, params["val_subjects"]), params["batch_size"])
    test_loader = DataLoader(EMGFK(DATA_DIR, params["test_subjects"]), params["batch_size"])
    return train_set, train_loader, val_loader, test_loader


def build_model(train_set, device):
    """Model with its output bias initialized to the mean training target."""
    model = Model(train_set.X.shape[1], train_set.y.shape[1]).to(device)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.from_numpy(train_set.y.mean(0)))
    return model


if __name__ == "__main__":
    params = yaml.safe_load(open("src/params.yaml"))["train"]
    torch.manual_seed(params["seed"])
    device = params["device"] if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_set, train_loader, val_loader, test_loader = load_data(params)

    # Build model and optimization
    model = build_model(train_set, device)
    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params["epochs"])

    # Train
    history, best = fit(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler,
        params["epochs"], device, OUT_DIR / "best_model.pt", params["grad_clip"],
    )
    save_history(history, OUT_DIR / "history.csv")

    # Test the best checkpoint
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=device))
    test = evaluate(model, test_loader, device)
    print(f"test mse {test['mse']:.4f} r2 {test['r2']:.4f}")
    (OUT_DIR / "metrics.json").write_text(json.dumps({"val": best, "test": test}, indent=2))
