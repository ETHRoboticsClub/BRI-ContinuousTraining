import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_dataset import INTRASUBJECT_BANDS, extract_intrasubject_dataset  # noqa: E402
from src.model.model import Model  # noqa: E402
from src.trainer import fit, save_history  # noqa: E402

DATA_DIR = Path("data/processed/emg_fk")
OUT_DIR = Path("output")


def train_from_recording(
    data_path: Path,
    out_ckpt: Path,
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 5e-4,
) -> dict[str, float]:
    """Extract features from an intra-subject .npz recording, train the GRU decoder, and save checkpoint."""
    torch.manual_seed(42)
    np.random.seed(42)

    with np.load(data_path) as archive:
        emg = archive["emg"].astype(np.float32)
        targets = archive["targets"].astype(np.float32)
        fs = int(archive["fs"]) if "fs" in archive else 200
        session_lengths = (
            archive["session_lengths"].astype(np.int64).tolist() if "session_lengths" in archive else [int(len(emg))]
        )

    sub_len, seq_len = 100, 10
    ds = extract_intrasubject_dataset(
        emg, targets, session_lengths=session_lengths, fs=fs, sub_len=sub_len, seq_len=seq_len
    )
    x_train_raw, y_train_deg = ds["X_train_raw"], ds["Y_train_deg"]
    x_val_raw, y_val_deg = ds["X_val_raw"], ds["Y_val_deg"]

    f_mean = np.mean(x_train_raw, axis=(0, 1), keepdims=True).astype(np.float32)
    f_std = np.maximum(np.std(x_train_raw, axis=(0, 1), keepdims=True), 0.05).astype(np.float32)
    t_mean = np.mean(y_train_deg, axis=0).astype(np.float32)
    t_std = np.maximum(np.std(y_train_deg, axis=0), 1.0).astype(np.float32)

    x_train = np.clip((x_train_raw - f_mean) / f_std, -4.0, 4.0)
    y_train = (y_train_deg - t_mean) / t_std
    x_val = np.clip((x_val_raw - f_mean) / f_std, -4.0, 4.0)
    y_val = (y_val_deg - t_mean) / t_std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(channels=8, targets=15, hidden=64, in_features=int(x_train.shape[-1])).to(device)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    extra = {
        "feature_mean": f_mean,
        "feature_std": f_std,
        "target_mean": t_mean,
        "target_std": t_std,
        "channel_std": ds["channel_std"],
        "inv_sqrt_mean_cov": ds["inv_sqrt_mean_cov"],
        "intrasubject_calibrated": True,
        "intrasubject_bands": INTRASUBJECT_BANDS,
        "intrasubject_sub_len": sub_len,
        "intrasubject_seq_len": seq_len,
        "biomechanical_dip_coupling": True,
    }

    history, best = fit(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler,
        epochs,
        device,
        out_ckpt,
        grad_clip=1.0,
        target_mean=t_mean,
        target_std=t_std,
        checkpoint_extra=extra,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_history(history, OUT_DIR / "history.csv")
    (OUT_DIR / "metrics.json").write_text(json.dumps({"val": best}, indent=2))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intra-subject TRR GRU decoder.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "recordings" / "myo_intrasubject.npz",
        help="Path to recorded intra-subject .npz dataset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "best_model.pt",
        help="Output model checkpoint path.",
    )
    args = parser.parse_args()

    with open(REPO_ROOT / "src" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)["train"]

    train_from_recording(
        data_path=args.data,
        out_ckpt=args.out,
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        lr=float(params["lr"]),
    )


if __name__ == "__main__":
    main()
