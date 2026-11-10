import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


def apply_dip_coupling(angles_deg: np.ndarray) -> np.ndarray:
    """Couple finger DIP joints (5, 8, 11, 14) with PIP flexion (4, 7, 10, 13)."""
    out = np.array(angles_deg, dtype=np.float32, copy=True)
    for pip_i, dip_i in ((4, 5), (7, 8), (10, 11), (13, 14)):
        coupled_dip = 180.0 - 0.68 * (180.0 - out[..., pip_i])
        out[..., dip_i] = np.clip(0.75 * out[..., dip_i] + 0.25 * coupled_dip, 0.0, 180.0)
    return out


def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip=1.0) -> float:
    """One pass over loader; returns the mean training loss."""
    model.train()
    total = 0.0
    for x, y in loader:
        loss = loss_fn(model(x.to(device)), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, target_mean=None, target_std=None) -> dict:
    """Evaluate MAE (deg), MSE, NMSE, R², and Pearson r on loader."""
    model.eval()
    y_list, pred_list = [], []
    for xb, yb in loader:
        pred_list.append(model(xb.to(device)).cpu())
        y_list.append(yb)
    y = torch.cat(y_list).numpy()
    pred = torch.cat(pred_list).numpy()

    if target_mean is not None and target_std is not None:
        t_mean = np.asarray(target_mean, dtype=np.float32)
        t_std = np.asarray(target_std, dtype=np.float32)
        y = y * t_std + t_mean
        pred = apply_dip_coupling(pred * t_std + t_mean)

    mse = ((pred - y) ** 2).mean(axis=0)
    var = np.maximum(y.var(axis=0), 1e-4)
    corrs = [
        float(np.corrcoef(pred[:, j], y[:, j])[0, 1])
        for j in range(y.shape[1])
        if np.std(pred[:, j]) > 1e-4 and np.std(y[:, j]) > 1e-4
    ]
    return {
        "mae": float(np.abs(pred - y).mean()),
        "mse": float(mse.mean()),
        "nmse": float((mse / var).mean()),
        "r2": float((1.0 - mse / var).mean()),
        "pearson": float(np.nanmean(corrs)) if corrs else 0.0,
    }


def fit(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    epochs,
    device,
    checkpoint,
    grad_clip=1.0,
    target_mean=None,
    target_std=None,
    checkpoint_extra=None,
):
    """Train for epochs, saving the checkpoint with the lowest validation MAE."""
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip)
        if scheduler is not None:
            scheduler.step()
        val = evaluate(model, val_loader, device, target_mean, target_std)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(
            f"[{epoch:03d}] train loss {train_loss:.4f} | "
            f"val mae {val['mae']:.2f} deg | nmse {val['nmse']:.3f} | r {val['pearson']:.3f}"
        )
        if best is None or val["mae"] < best["mae"]:
            best = {"epoch": epoch, **val}
            payload = {
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                **(checkpoint_extra or {}),
                "val_mae": val["mae"],
                "val_nmse": val["nmse"],
            }
            torch.save(payload, checkpoint)
    return history, best


def save_history(history, path) -> None:
    if not history:
        return
    with open(Path(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
