"""Run the extra experiments on this machine, without DVC, and resume after an interruption.

    python -m scripts.sweep_extra_local relu 10 40 [--start aug_gain] [--out output/extra_relu_k10_s40]

The same changes as scripts/sweep_extra.sh, each on its own on top of src/params.yaml's train params with the
given activation, k and example stride. Experiment <label> writes to <out>/<label>/ (params.json, metrics.json,
history.csv, seed_<n>/). Rerunning the same command skips finished experiments (metrics.json with the same params)
and finished seeds, so after the machine turns off, at most the seed in progress is lost. While it runs, Windows is
kept from sleeping on idle; closing the lid or shutting down still stops it.
"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml

from scripts.train import DATA_DIR, run, write_atomic

# (label, change); the same order and values as scripts/sweep_extra.sh
CHANGES = [
    ("baseline", {}),
    ("aug_gain", {"aug_gain": 0.2}),
    ("aug_rotate", {"aug_rotate": 1}),
    ("aug_shift", {"aug_shift": 0.5}),
    ("norm_set", {"normalization": "set"}),
    ("downsample_2", {"downsample": 2}),
    ("downsample_4", {"downsample": 4}),
    ("speed_1", {"ssm_speed": 1}),
    ("speed_0p25", {"ssm_speed": 0.25}),
    ("ssm_lr_0p1", {"ssm_lr_scale": 0.1}),
    ("no_decay_a", {"ssm_decay_A": False}),
]


def load_train_params() -> dict:
    return yaml.safe_load(open("src/params.yaml"))["train"]


def experiment_params(base: dict, activation: str, k: int, stride: int, change: dict) -> dict:
    """Train params of one experiment: base with the activation, k, example stride and the change applied."""
    return {**base, "activation": activation, "k": k, "example_stride": stride, **change}


@contextmanager
def keep_awake():
    """Keep Windows from sleeping on idle (SetThreadExecutionState); does nothing elsewhere."""
    try:
        import ctypes

        set_state = ctypes.windll.kernel32.SetThreadExecutionState
    except (AttributeError, OSError):
        yield
        return
    continuous, system_required = 0x80000000, 0x00000001
    set_state(continuous | system_required)
    try:
        yield
    finally:
        set_state(continuous)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("activation", choices=["tanh", "relu"])
    parser.add_argument("k", type=int)
    parser.add_argument("stride", type=int)
    labels = [label for label, _ in CHANGES]
    parser.add_argument("--start", default="baseline", help=f"first experiment to run, one of {', '.join(labels)}")
    parser.add_argument("--out", help="default: output/extra_<activation>_k<k>_s<stride>")
    args = parser.parse_args(argv)
    if args.start not in labels:
        parser.error(f"--start {args.start!r} is not one of {', '.join(labels)}")
    todo = CHANGES[labels.index(args.start) :]
    length = 10 * args.k  # kernel length in tangent steps
    for label, change in todo:
        if length % change.get("downsample", 1):
            parser.error(f"{label}: downsample {change['downsample']} does not divide L = {length} (k = {args.k})")

    out = Path(args.out or f"output/extra_{args.activation}_k{args.k}_s{args.stride}")
    out.mkdir(parents=True, exist_ok=True)
    base = experiment_params(load_train_params(), args.activation, args.k, args.stride, {})
    write_atomic(out / "base_params.json", json.dumps(base, indent=2))
    with keep_awake():
        for label, change in todo:
            exp_dir = out / label
            params = experiment_params(base, args.activation, args.k, args.stride, change)
            done = (exp_dir / "metrics.json").exists() and (exp_dir / "params.json").exists()
            if done and json.loads((exp_dir / "params.json").read_text()) == json.loads(json.dumps(params)):
                print(f"=== {label}: finished earlier, skipping")
                continue
            exp_dir.mkdir(parents=True, exist_ok=True)
            write_atomic(exp_dir / "params.json", json.dumps(params, indent=2))
            print(f"=== {label}: {change or 'no change'} -> {exp_dir}", flush=True)
            run(params, DATA_DIR, exp_dir, resume=True)


if __name__ == "__main__":
    main(sys.argv[1:])
