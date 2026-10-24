"""CLI Inference Script for Temporal Riemannian Regressor (TRR).

Corresponds to Script 2 (infer_trr.py) from Colot et al. (2026):
"Real-Time TRR Inference (infer_trr.py)".

Decodes sEMG recordings into 15-DOF finger joint angles in degrees or radians, supporting both
real-time streaming FIFO queue simulation and fast batch offline prediction.

Sections:
1. CLI Argument Parsing: Configures paths, sampling rate (default 200.0 Hz), units, and mode.
2. File Ingestion Helper: Ingests .npz or .fif recording files.
3. Streaming & Batch Runner: Executes decoding with latency profiling and regression evaluation.
4. Visualization: Generates trajectory comparison plots for ground truth vs predictions.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.inference import (
    DEFAULT_SAMPLING_RATE_HZ,
    StreamingTRRInference,
    convert_angles,
    infer_recording,
    load_inference_artifacts,
)
from src.metrics import regression_metrics


def load_input_recording(file_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list]]:
    """Load sEMG signals and optional ground-truth kinematics from an input file.

    Args:
        file_path: Path to .npz or .fif recording.

    Returns:
        Tuple of (emg [channels, time], targets [joints, time] or None, target_names).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input recording not found: {file_path}")

    if file_path.suffix == ".npz":
        with np.load(file_path) as data:
            emg = data["emg"].astype(np.float32)
            targets = (
                data["targets"].astype(np.float32)
                if "targets" in data
                else (data["angles"].astype(np.float32) if "angles" in data else None)
            )
            target_names = (
                [str(n) for n in data["target_names"]]
                if "target_names" in data
                else [f"Angle_{i+1}" for i in range(targets.shape[0] if targets is not None else 15)]
            )
        return emg, targets, target_names

    elif file_path.suffix == ".fif":
        try:
            import mne
        except ImportError as e:
            raise ImportError("mne is required to load .fif files. Install with 'pip install mne'.") from e
        raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose="ERROR")
        emg_picks = [f"EMG {i+1}" for i in range(8) if f"EMG {i+1}" in raw.ch_names]
        angle_picks = [f"Angle {i+1}" for i in range(15) if f"Angle {i+1}" in raw.ch_names]
        emg = raw.get_data(picks=emg_picks).astype(np.float32)
        targets = raw.get_data(picks=angle_picks).astype(np.float32) if angle_picks else None
        target_names = angle_picks if angle_picks else [f"Angle_{i+1}" for i in range(15)]
        return emg, targets, target_names

    else:
        raise ValueError(f"Unsupported file format '{file_path.suffix}'. Use .npz or .fif.")


def plot_predictions(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
    units: str,
    save_path: Path,
    max_joints: int = 4,
) -> None:
    """Save trajectory comparison plot of predicted vs true joint angles."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_joints = min(max_joints, y_pred.shape[1])
    joint_indices = np.linspace(0, y_pred.shape[1] - 1, num_joints, dtype=int)

    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2.5 * num_joints), sharex=True)
    if num_joints == 1:
        axes = [axes]

    for ax, j_idx in zip(axes, joint_indices, strict=True):
        j_name = target_names[j_idx] if j_idx < len(target_names) else f"Joint {j_idx+1}"
        ax.plot(timestamps, y_true[:, j_idx], "k-", label="True Angle", alpha=0.85, linewidth=1.2)
        ax.plot(timestamps, y_pred[:, j_idx], "r--", label="TRR Predicted", alpha=0.85, linewidth=1.2)
        ax.set_ylabel(f"{j_name}\n({units})")
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax is axes[0]:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    plt.suptitle("TRR Continuous Joint Angle Decoding", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TRR real-time or offline inference on sEMG recordings.")
    parser.add_argument("--input", type=str, required=True, help="Path to input sEMG recording (.npz or .fif)")
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_SAMPLING_RATE_HZ,
        help=f"Input sampling rate in Hz (default: {DEFAULT_SAMPLING_RATE_HZ} Hz)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/cross_subject/best_model.pt",
        help="Path to trained TRR model weights (default: runs/cross_subject/best_model.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to pipeline configuration params.yaml (default: params.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save output predictions (.npz or .csv)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["streaming", "batch"],
        default="streaming",
        help="Inference mode: 'streaming' (simulates 10 Hz FIFO queue) or 'batch' (vectorized full-file decoding)",
    )
    parser.add_argument(
        "--units",
        type=str,
        choices=["deg", "rad"],
        default="deg",
        help="Output joint angle units: 'deg' (degrees, default) or 'rad' (radians)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device: 'auto' (GPU if available), 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save trajectory comparison plot if ground-truth targets are present",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Step stride in samples between consecutive predictions at 200 Hz (default: 20 = 100 ms)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    print(f"--- Loading input recording: {input_path} (fs={args.fs} Hz) ---")
    emg, targets, target_names = load_input_recording(input_path)
    print(f"Loaded sEMG array with shape {emg.shape} ({emg.shape[1] / args.fs:.2f} seconds)")

    if args.mode == "streaming":
        print(f"--- Starting Streaming FIFO Inference (Device: {args.device}, Units: {args.units}) ---")
        engine = StreamingTRRInference(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            fs=args.fs,
            device=args.device,
            units=args.units,
            standardize_input=True,
        )

        # Standardize using whole recording statistics for cross-subject mode
        engine.set_standardization(emg.mean(axis=1, keepdims=True), emg.std(axis=1, keepdims=True))

        # Chunk size: 100 ms step at the input sampling rate
        chunk_size = int(round(0.100 * args.fs))
        total_chunks = (emg.shape[1] - engine.sub_window_len) // chunk_size
        if total_chunks <= 0:
            raise ValueError(f"Recording too short ({emg.shape[1]} samples). Minimum required: {engine.sub_window_len}")

        predictions_list = []
        latencies = []

        # Feed initial sub-window buffer
        engine.raw_buffer = emg[:, : engine.sub_window_len].copy()

        # Stream chunk by chunk (10 Hz update loop)
        for c_idx in range(total_chunks):
            start = engine.sub_window_len + c_idx * chunk_size
            end = start + chunk_size
            if end > emg.shape[1]:
                break
            chunk = emg[:, start:end]
            pred, lat = engine.step(chunk)
            predictions_list.append(pred)
            latencies.append(lat)

        predictions = np.stack(predictions_list)
        timestamps = np.arange(len(predictions)) * 0.100 + (engine.sub_window_len / args.fs)

        lat_arr = np.array(latencies)
        print("\n=== Streaming Performance Summary ===")
        print(f"Total steps executed: {len(predictions)}")
        print(f"Mean Step Latency:    {lat_arr.mean():.2f} ms ± {lat_arr.std():.2f} ms")
        print(f"Min / Max Latency:    {lat_arr.min():.2f} ms / {lat_arr.max():.2f} ms")
        print(f"Real-Time Throughput: {1000.0 / lat_arr.mean():.1f} predictions/sec (Target: 10 Hz)")

    else:
        print(f"--- Starting Batch Offline Inference (Device: {args.device}, Units: {args.units}) ---")
        model, cfg, dev = load_inference_artifacts(args.checkpoint, args.config, device=args.device)
        predictions, timestamps, stats = infer_recording(
            emg=emg,
            model=model,
            cfg=cfg,
            fs=args.fs,
            units=args.units,
            device=dev,
            stride=args.stride,
        )
        print("\n=== Batch Performance Summary ===")
        print(f"Decoded Windows:      {stats['total_windows']}")
        print(f"Total Compute Time:   {stats['total_time_sec']:.2f} s")
        print(f"Throughput:           {stats['throughput_windows_per_sec']:.1f} windows/sec")
        print(f"Average Latency:      {stats['avg_latency_ms']:.2f} ms/window")

    # Evaluate against ground truth if targets are available
    if targets is not None:
        print("\n=== Regression Metrics vs Ground Truth ===")
        # Interpolate ground truth to prediction timestamps
        source_ts = np.arange(targets.shape[1]) / args.fs
        true_aligned = np.zeros_like(predictions)
        for j in range(targets.shape[0]):
            true_aligned[:, j] = np.interp(timestamps, source_ts, targets[j])

        # Convert targets if units are radians
        if args.units == "rad":
            true_aligned = convert_angles(true_aligned, from_unit="deg", to_unit="rad")

        metrics = regression_metrics(true_aligned, predictions)
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.2f} {args.units}")
        print(f"Mean Squared Error (MSE):  {metrics['mse']:.2f}")
        print(f"Coefficient of Det. (R2):  {metrics['r2']:.4f}")
        print(f"Pearson Correlation (r):   {metrics['pearson']:.4f}")
        print(f"Jitter Ratio:              {metrics['jitter_ratio']:.2f} (< 1.0 indicates smooth motion)")

        if args.plot:
            plot_path = (
                Path(args.output).with_suffix(".png")
                if args.output
                else Path("runs/cross_subject") / f"{input_path.stem}_trajectory.png"
            )
            plot_predictions(timestamps, true_aligned, predictions, target_names, args.units, plot_path)

    # Save output if specified
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            header = ["timestamp_sec"] + [f"joint_{i+1}" for i in range(predictions.shape[1])]
            rows = np.column_stack([timestamps, predictions])
            np.savetxt(out_path, rows, delimiter=",", header=",".join(header), comments="")
        else:
            np.savez_compressed(
                out_path,
                predictions=predictions,
                timestamps=timestamps,
                units=args.units,
            )
        print(f"\nPredictions successfully saved to: {out_path}")


if __name__ == "__main__":
    main()
