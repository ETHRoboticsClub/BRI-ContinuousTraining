import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.augment import Augment
from src.dataset import EMGFK, kernel_length, recording_path, shortest_recording
from src.model.model import Model
from src.model.tangent_space import raw_length
from src.trainer import evaluate, fit, param_groups, save_history

DATA_DIR = Path("data/processed/emg_fk")
OUT_DIR = Path("output")
EVAL_STRIDE = 40  # raw samples between validation / test examples
METRICS = ("mse", "r2", "nmse", "mae")


def load_data(params, data_dir, length):
    """Cross-subject split: val/test subjects are held out, every other recorded subject is used for training.

    Only the training set is augmented.
    """
    held_out = params["val_subjects"] + params["test_subjects"]
    recorded = [s for s in range(1, 100) if recording_path(data_dir, s).exists()]
    train_subjects = [s for s in recorded if s not in held_out]
    norm = params["normalization"]
    train_set = EMGFK(data_dir, train_subjects, length, params["example_stride"], norm, Augment.from_params(params))
    train_loader = DataLoader(train_set, params["batch_size"], shuffle=True, drop_last=True)
    val_set = EMGFK(data_dir, params["val_subjects"], length, EVAL_STRIDE, norm)
    test_set = EMGFK(data_dir, params["test_subjects"], length, EVAL_STRIDE, norm)
    val_loader = DataLoader(val_set, params["batch_size"])
    test_loader = DataLoader(test_set, params["batch_size"])
    return train_set, train_loader, val_loader, test_loader


def aggregate(per_seed: list[dict]) -> dict:
    """Mean and sample std (ddof=1) of each metric over seeds."""
    return {
        m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
        for m in METRICS
        for v in [[r[m] for r in per_seed]]
    }


def run(params: dict, data_dir: Path, out_dir: Path) -> dict:
    """Train one model per seed, test each best checkpoint, write history.csv and metrics.json to out_dir."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    device = params["device"] if torch.cuda.is_available() else "cpu"
    length = kernel_length(params["k"], shortest_recording(data_dir))
    train_set, train_loader, val_loader, test_loader = load_data(params, data_dir, raw_length(length))
    history, seeds = [], {}
    for seed in params["seeds"]:
        torch.manual_seed(seed)
        if train_set.augment is not None:
            train_set.augment.manual_seed(seed)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Build model and optimization
        model = Model(
            train_set.channels,
            train_set.targets,
            length,
            params["activation"],
            downsample=params["downsample"],
            ssm_speed=params["ssm_speed"],
        ).to(device)
        loss_fn = nn.HuberLoss()
        groups = param_groups(
            model, params["lr"], params["weight_decay"], params["ssm_lr_scale"], params["ssm_decay_A"]
        )
        optimizer = torch.optim.AdamW(groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params["epochs"])

        # Train
        seed_history, val = fit(
            model, train_loader, val_loader, loss_fn, optimizer, scheduler,
            params["epochs"], device, seed_dir / "best_model.pt", params["grad_clip"],
        )  # fmt: skip
        history += [{"seed": seed, **row} for row in seed_history]

        # Test the best checkpoint
        model.load_state_dict(torch.load(seed_dir / "best_model.pt", map_location=device))
        test = evaluate(model, test_loader, device)
        print(f"seed {seed}: val mae {val['mae']:.2f} | test r2 {test['r2']:.4f} mae {test['mae']:.2f}")
        seeds[str(seed)] = {"val": val, "test": test}

    metrics = {
        "kernel_length": length,
        "example_length": raw_length(length),
        **{split: aggregate([r[split] for r in seeds.values()]) for split in ("val", "test")},
        "seeds": seeds,
    }
    save_history(history, out_dir / "history.csv")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run(yaml.safe_load(open("src/params.yaml"))["train"], DATA_DIR, OUT_DIR)
