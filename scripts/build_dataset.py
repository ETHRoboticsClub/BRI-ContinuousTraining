import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.tangent_space import ledoit_wolf_cov, riemannian_whitening  # noqa: E402
from src.preprocess import Preprocess  # noqa: E402

INTRASUBJECT_BANDS: tuple[tuple[float, float], ...] = ((5.0, 40.0), (40.0, 70.0), (70.0, 95.0))


def extract_intrasubject_dataset(
    emg: np.ndarray,
    targets: np.ndarray,
    session_lengths: Sequence[int] | np.ndarray | None = None,
    fs: int = 200,
    sub_len: int = 100,
    step: int = 20,
    seq_len: int = 10,
) -> dict[str, np.ndarray]:
    """Extract 10-step 124-feature 3-band Riemannian tangent + phasic/tonic sequences."""
    total_samples = int(emg.shape[0])
    min_seq_samples = sub_len + (seq_len - 1) * step
    if session_lengths is None or len(session_lengths) == 0:
        raw_lengths = [total_samples]
    else:
        raw_lengths = [int(x) for x in session_lengths if int(x) > 0]
        if sum(raw_lengths) != total_samples or not raw_lengths:
            raw_lengths = [total_samples]

    has_long_session = any(x >= 12000 for x in raw_lengths)
    row, col = np.triu_indices(8)
    triu_weights = np.where(row == col, 1.0, np.sqrt(2.0)).astype(np.float32)
    eye_8 = np.eye(8, dtype=np.float32)
    b_lp, a_lp = butter(3, 3.5 / (0.5 * fs), btype="low")

    x_train, y_train, x_val, y_val, x_all, y_all = [], [], [], [], [], []
    val_session_slices: list[tuple[int, int, int]] = []
    latest_channel_std = np.zeros((3, 8, 1), dtype=np.float32)
    latest_inv_sqrt = np.zeros((3, 8, 8), dtype=np.float32)
    valid_sessions = 0
    val_offset = 0
    offset = 0

    for s_idx, s_len in enumerate(raw_lengths):
        s_emg = emg[offset : offset + s_len]
        s_tgt = targets[offset : offset + s_len]
        offset += s_len

        if s_len < min_seq_samples or (has_long_session and s_len < 12000):
            continue

        valid_sessions += 1
        if s_len > 100:
            s_tgt = np.clip(filtfilt(b_lp, a_lp, s_tgt, axis=0), 30.0, 180.0).astype(np.float32)

        prep = Preprocess(fs=fs, bandpass=(5.0, 95.0), notch=50.0, bands=INTRASUBJECT_BANDS)
        s_bands = prep.filter_bands(s_emg.T)

        k_steps = (s_len - sub_len) // step + 1
        raw_dc = s_emg - np.mean(s_emg, axis=0, keepdims=True)
        sub_rms = np.array(
            [np.sqrt(np.mean(raw_dc[k * step : k * step + sub_len] ** 2)) for k in range(k_steps)],
            dtype=np.float32,
        )
        active_mask = sub_rms >= max(float(np.percentile(sub_rms, 20.0)), 1.5)
        if np.sum(active_mask) < 5:
            active_mask = np.ones(k_steps, dtype=bool)

        s_channel_std = np.zeros((3, 8, 1), dtype=np.float32)
        s_inv_sqrt = np.zeros((3, 8, 8), dtype=np.float32)
        band_tangents: list[np.ndarray] = []
        log_rms_b0 = np.zeros((k_steps, 8), dtype=np.float32)

        for b_idx in range(3):
            xb = s_bands[b_idx]
            s_channel_std[b_idx] = np.maximum(np.std(xb, axis=-1, keepdims=True), 1.5).astype(np.float32)
            covs_b = np.zeros((k_steps, 8, 8), dtype=np.float32)

            for k in range(k_steps):
                sub = xb[:, k * step : k * step + sub_len]
                sub_scaled = (sub - np.mean(sub, axis=-1, keepdims=True)) / s_channel_std[b_idx]
                if b_idx == 0:
                    log_rms_b0[k] = np.log(np.sqrt(np.mean(sub_scaled**2, axis=-1)) + 1e-4)
                covs_b[k] = ledoit_wolf_cov(sub_scaled) + 1e-6 * eye_8

            _, wb = riemannian_whitening(covs_b[active_mask])
            s_inv_sqrt[b_idx] = wb

            cov_w = wb[None, :, :] @ covs_b @ wb.T[None, :, :]
            evals, evecs = np.linalg.eigh(cov_w)
            evals = np.maximum(evals, 1e-8)
            log_c = np.einsum("kij,kj,kjl->kil", evecs, np.log(evals), np.swapaxes(evecs, -1, -2))
            band_tangents.append((log_c[:, row, col] * triu_weights[None, :]).astype(np.float32))

        latest_channel_std = s_channel_std
        latest_inv_sqrt = s_inv_sqrt

        move_cmd_b0 = np.zeros_like(log_rms_b0)
        move_cmd_b0[1:] = np.maximum(0.0, log_rms_b0[1:] - log_rms_b0[:-1])
        tangents = np.concatenate([*band_tangents, log_rms_b0, move_cmd_b0], axis=-1)

        n_seq = k_steps - seq_len + 1
        xs = np.stack([tangents[i : i + seq_len] for i in range(n_seq)], axis=0).astype(np.float32)
        target_idx = step * np.arange(n_seq) + (seq_len - 1) * step + sub_len - 1
        ys = s_tgt[target_idx].astype(np.float32)

        n_tr = int(0.8 * n_seq)
        x_train.append(xs[:n_tr])
        y_train.append(ys[:n_tr])
        x_val.append(xs[n_tr:])
        y_val.append(ys[n_tr:])
        x_all.append(xs)
        y_all.append(ys)
        val_session_slices.append((s_idx + 1, val_offset, val_offset + (n_seq - n_tr)))
        val_offset += n_seq - n_tr

    return {
        "X_seq": np.concatenate(x_all, axis=0),
        "Y_deg": np.concatenate(y_all, axis=0),
        "X_train_raw": np.concatenate(x_train, axis=0),
        "Y_train_deg": np.concatenate(y_train, axis=0),
        "X_val_raw": np.concatenate(x_val, axis=0),
        "Y_val_deg": np.concatenate(y_val, axis=0),
        "channel_std": latest_channel_std,
        "inv_sqrt_mean_cov": latest_inv_sqrt,
        "n_sessions": np.int64(valid_sessions),
        "val_session_slices": np.asarray(val_session_slices, dtype=np.int64),
    }


def main(raw_path: str | Path, out_dir: str | Path, params: dict) -> None:
    """Build processed intra-subject dataset from a recorded .npz archive."""
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(raw_path) as archive:
        emg = archive["emg"].astype(np.float32)
        targets = archive["targets"].astype(np.float32)
        fs = int(archive["fs"]) if "fs" in archive else int(params["preprocessing"]["fs"])
        session_lengths = (
            archive["session_lengths"].astype(np.int64).tolist() if "session_lengths" in archive else [int(len(emg))]
        )

    train_cfg = params["train"]
    ds = extract_intrasubject_dataset(
        emg,
        targets,
        session_lengths=session_lengths,
        fs=fs,
        sub_len=int(train_cfg["sub_len"]),
        step=int(train_cfg["step"]),
        seq_len=int(train_cfg["seq_len"]),
    )

    f_mean = np.mean(ds["X_train_raw"], axis=(0, 1), keepdims=True).astype(np.float32)
    f_std = np.maximum(np.std(ds["X_train_raw"], axis=(0, 1), keepdims=True), 0.05).astype(np.float32)
    t_mean = np.mean(ds["Y_train_deg"], axis=0).astype(np.float32)
    t_std = np.maximum(np.std(ds["Y_train_deg"], axis=0), 1.0).astype(np.float32)

    np.save(out_dir / "X_train.npy", np.clip((ds["X_train_raw"] - f_mean) / f_std, -4.0, 4.0))
    np.save(out_dir / "y_train.npy", (ds["Y_train_deg"] - t_mean) / t_std)
    np.save(out_dir / "X_val.npy", np.clip((ds["X_val_raw"] - f_mean) / f_std, -4.0, 4.0))
    np.save(out_dir / "y_val.npy", (ds["Y_val_deg"] - t_mean) / t_std)
    np.savez(
        out_dir / "calibration.npz",
        feature_mean=f_mean,
        feature_std=f_std,
        target_mean=t_mean,
        target_std=t_std,
        channel_std=ds["channel_std"],
        inv_sqrt_mean_cov=ds["inv_sqrt_mean_cov"],
    )
    print(f"Saved intra-subject dataset to {out_dir} ({len(ds['X_train_raw'])} train, {len(ds['X_val_raw'])} val)")


if __name__ == "__main__":
    with open("src/params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main("data/recordings/myo_intrasubject.npz", "data/processed/emg_fk", cfg)
