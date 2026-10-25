import csv
from pathlib import Path

import torch
from torch import nn


def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip=None) -> float:
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
def evaluate(model, loader, device) -> dict:
    """Mean squared error and R² averaged over targets."""
    model.eval()
    y, pred = [], []
    for xb, yb in loader:
        pred.append(model(xb.to(device)).cpu())
        y.append(yb)
    y, pred = torch.cat(y), torch.cat(pred)
    mse = ((pred - y) ** 2).mean(0)
    return {"mse": mse.mean().item(), "r2": (1 - mse / y.var(0, unbiased=False)).mean().item()}


def fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, device, checkpoint, grad_clip=None):
    """Train for epochs, saving the state with the best validation R² to checkpoint. Returns (history, best)."""
    history, best = [], None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip)
        scheduler.step()
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mse": val["mse"], "val_r2": val["r2"]})
        print(f"[{epoch:03d}] train loss {train_loss:.4f} | val mse {val['mse']:.4f} r2 {val['r2']:.4f}")
        if best is None or val["r2"] > best["r2"]:
            best = val
            torch.save(model.state_dict(), checkpoint)
    return history, best


def save_history(history, path) -> None:
    with open(Path(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
