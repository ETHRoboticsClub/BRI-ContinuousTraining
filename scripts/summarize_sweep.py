"""Summarize a sweep: python scripts/summarize_sweep.py sweep_tanh.csv [--activation tanh]

Reads the CSV written by `dvc exp show --csv`, prints every configuration's validation and test MAE,
selects the configuration with the lowest validation MAE (test is never used for selection) and
gives the verdict against the paper's cross-subject test MAE of 16.71 degrees.

For an extra sweep (extra_<activation>_k<K>_s<STRIDE>.csv, from scripts/sweep_extra.sh) it prints each change
against the baseline rerun instead, and selects the change with the lowest validation MAE.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

PAPER_MAE = 16.71
TOLERANCE = 0.5  # degrees above the paper that still counts as a threshold result

VAL = "val.mae.mean"
TEST = "test.mae.mean"
EXPECTED_CONFIGS = 7  # k=1 with 3 strides, k=5 with 4 strides
VAL_STD = "val.mae.std"
COLUMNS = (VAL, TEST, "train.k", "train.example_stride", "train.activation")
EXTRA_COLUMNS = ("Experiment", VAL, VAL_STD, TEST)
NUMBERS = (VAL, VAL_STD, TEST)


def select(rows: list[dict]) -> dict:
    """Row with the lowest validation MAE."""
    return min(rows, key=lambda row: row[VAL])


def verdict(test_mae: float) -> str:
    if test_mae < PAPER_MAE:
        return "success"
    if test_mae <= PAPER_MAE + TOLERANCE:
        return "threshold"
    return "miss"


def _number(value: str) -> float | None:
    """Float, or None when missing or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _column(header: list[str], name: str) -> str | None:
    """Header equal to `name` or ending with ':' + name (dvc prefixes columns with the file path)."""
    return next((h for h in header if h == name or h.endswith(":" + name)), None)


def read_rows(path: str, columns=COLUMNS) -> list[dict]:
    """Rows with the canonical column names, all activations.

    Rows without a finite validation MAE (baseline workspace row, failed experiments) are skipped. Every row also
    gets "params": all train.* columns, by their canonical name.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        names = {c: _column(header, c) for c in columns}
        missing = [c for c, h in names.items() if h is None]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        train = {h: h.rsplit(":", 1)[-1] for h in header if h.rsplit(":", 1)[-1].startswith("train.")}
        rows = []
        for raw in reader:
            row = {c: raw[h] for c, h in names.items()}
            row["params"] = {name: raw[h] for h, name in train.items()}
            for c in NUMBERS:
                if c in row:
                    row[c] = _number(row[c])
            if row[VAL] is not None:
                rows.append(row)
    return rows


def for_activation(rows: list[dict], activation: str) -> list[dict]:
    """Rows of one activation, one per (k, example_stride) (the first one listed).

    `dvc exp show` lists every experiment of the commit, so a relu sweep csv also holds the tanh runs.
    """
    seen = {}
    for row in rows:
        if row["train.activation"] == activation:
            seen.setdefault((row["train.k"], row["train.example_stride"]), row)
    return list(seen.values())


def activation_from_filename(path: str) -> str | None:
    match = re.fullmatch(r"sweep_(.+)\.csv", os.path.basename(path))
    return match.group(1) if match else None


def base_from_filename(path: str) -> tuple[str, str, str] | None:
    """(activation, k, stride) of an extra_<activation>_k<K>_s<STRIDE>.csv file name."""
    match = re.fullmatch(r"extra_(.+)_k(\d+)_s(\d+)\.csv", os.path.basename(path))
    return match.groups() if match else None


def extra_rows(rows: list[dict], activation: str, k: str, stride: str) -> dict[str, dict]:
    """Label -> row for the experiments named x-<activation>-k<k>-s<stride>-<label> (the first one listed).

    Every experiment but the baseline must differ from it in exactly one train param; that one is stored as the
    row's "change" ("key=value"). Otherwise ValueError: the comparison with the baseline would be mislabelled.
    """
    prefix = f"x-{activation}-k{k}-s{stride}-"
    labeled = {}
    for row in rows:
        if row["Experiment"].startswith(prefix):
            labeled.setdefault(row["Experiment"][len(prefix) :], row)
    if "baseline" not in labeled:
        raise ValueError(f"no {prefix}baseline experiment with {VAL}")
    return label_changes(labeled)


def label_changes(labeled: dict[str, dict]) -> dict[str, dict]:
    """Store each row's single param difference from labeled["baseline"] as its "change"; ValueError otherwise."""
    base = labeled["baseline"]["params"]
    labeled["baseline"]["change"] = ""
    problems = []
    for label, row in labeled.items():
        if label == "baseline":
            continue
        diff = sorted(key for key in base.keys() | row["params"].keys() if row["params"].get(key) != base.get(key))
        if len(diff) == 1:
            row["change"] = f"{diff[0].removeprefix('train.')}={row['params'].get(diff[0])}"
        else:
            problems.append(f"{label}: {', '.join(diff) if diff else 'no change'}")
    if problems:
        raise ValueError(f"experiments must differ from the baseline in exactly one train param: {'; '.join(problems)}")
    return labeled


def beats_baseline(row: dict, base: dict) -> bool:
    """Validation MAE lower than the baseline's by more than the baseline's seed std."""
    return base[VAL] - row[VAL] > (base[VAL_STD] or 0.0)


def _experiment_row(exp_dir: Path, params: dict) -> dict:
    metrics = json.loads((exp_dir / "metrics.json").read_text())
    return {
        VAL: metrics["val"]["mae"]["mean"],
        VAL_STD: metrics["val"]["mae"]["std"],
        TEST: metrics["test"]["mae"]["mean"],
        "params": {f"train.{key}": value for key, value in params.items()},
    }


def extra_rows_from_dirs(out_dir, baseline_dir=None) -> dict[str, dict]:
    """Label -> row for the finished experiments of scripts/sweep_extra_local.py in out_dir (<label>/metrics.json).

    The baseline is out_dir/baseline, or baseline_dir (e.g. an earlier run of the same base configuration), whose
    params are taken to be out_dir/base_params.json. Each change is checked as in label_changes.
    """
    out_dir = Path(out_dir)
    base_params = json.loads((out_dir / "base_params.json").read_text())
    labeled = {"baseline": _experiment_row(Path(baseline_dir or out_dir / "baseline"), base_params)}
    for exp_dir in sorted(out_dir.iterdir()):
        if exp_dir.name != "baseline" and (exp_dir / "metrics.json").exists():
            labeled[exp_dir.name] = _experiment_row(exp_dir, json.loads((exp_dir / "params.json").read_text()))
    return label_changes(labeled)


def main_extra(path: str, activation: str, k: str, stride: str) -> None:
    rows = extra_rows(read_rows(path, EXTRA_COLUMNS), activation, k, stride)
    print_extra(rows, f"activation={activation} k={k} stride={stride}")


def main_extra_dirs(out_dir, baseline_dir=None) -> None:
    base = json.loads((Path(out_dir) / "base_params.json").read_text())
    rows = extra_rows_from_dirs(out_dir, baseline_dir)
    print_extra(rows, f"activation={base.get('activation')} k={base.get('k')} stride={base.get('example_stride')}")


def print_extra(rows: dict[str, dict], title: str) -> None:
    """Each change against the baseline, sorted by val MAE, and the change selected by validation."""
    base = rows["baseline"]

    def fmt(value: float | None, signed: bool = False) -> str:
        return "-" if value is None else f"{value:+.2f}" if signed else f"{value:.2f}"

    print(f"base: {title}  (* = val MAE below baseline by > its seed std)")
    print(f"{'experiment':<14}{'change':<22}{'val MAE':>15}{'d val':>8}{'test MAE':>10}{'d test':>8}")
    for label, r in sorted(rows.items(), key=lambda item: item[1][VAL]):
        val = f"{r[VAL]:.2f} +/- {fmt(r[VAL_STD])}"
        d_test = None if r[TEST] is None or base[TEST] is None else r[TEST] - base[TEST]
        star = " *" if label != "baseline" and beats_baseline(r, base) else ""
        d_val = fmt(r[VAL] - base[VAL], True)
        print(f"{label:<14}{r['change']:<22}{val:>15}{d_val:>8}{fmt(r[TEST]):>10}{fmt(d_test, True):>8}{star}")
    label, best = min(rows.items(), key=lambda item: item[1][VAL])
    print(f"\nselected by validation: {label} (val MAE {best[VAL]:.2f})")
    if best[TEST] is None:
        print("no test MAE for the selected change")
    else:
        print(f"test MAE {best[TEST]:.2f} vs paper {PAPER_MAE}: {verdict(best[TEST])}")


def main(path: str, activation: str) -> None:
    rows = for_activation(read_rows(path), activation)
    if not rows:
        sys.exit(f"no {activation} experiment with {VAL} in {path}")
    if len(rows) < EXPECTED_CONFIGS:
        print(f"warning: {len(rows)} of {EXPECTED_CONFIGS} configurations found")
        print()
    print(f"{'activation':<11}{'k':>3}{'stride':>8}{'val MAE':>10}{'test MAE':>10}")
    for r in sorted(rows, key=lambda r: r[VAL]):
        test = "-" if r[TEST] is None else f"{r[TEST]:.2f}"
        print(f"{r['train.activation']:<11}{r['train.k']:>3}{r['train.example_stride']:>8}{r[VAL]:>10.2f}{test:>10}")
    best = select(rows)
    print(
        f"\nselected by validation: k={best['train.k']} stride={best['train.example_stride']} "
        f"activation={best['train.activation']} (val MAE {best[VAL]:.2f})"
    )
    if best[TEST] is None:
        print("no test MAE for the selected configuration")
    else:
        print(f"test MAE {best[TEST]:.2f} vs paper {PAPER_MAE}: {verdict(best[TEST])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "csv", help="sweep_<activation>.csv, extra_<activation>_k<K>_s<STRIDE>.csv, or a sweep_extra_local directory"
    )
    parser.add_argument("--activation", help="default: taken from the csv file name")
    parser.add_argument("--baseline", help="directory mode: baseline run directory (default: <directory>/baseline)")
    args = parser.parse_args()
    if os.path.isdir(args.csv):
        try:
            main_extra_dirs(args.csv, args.baseline)
        except (ValueError, FileNotFoundError) as e:
            sys.exit(str(e))
        sys.exit(0)
    extra = base_from_filename(args.csv)
    if extra is not None:
        try:
            main_extra(args.csv, *extra)
        except ValueError as e:
            sys.exit(str(e))
        sys.exit(0)
    activation = args.activation or activation_from_filename(args.csv)
    if activation is None:
        parser.error("pass --activation (the file name is not sweep_<activation>.csv)")
    try:
        main(args.csv, activation)
    except ValueError as e:
        sys.exit(str(e))
