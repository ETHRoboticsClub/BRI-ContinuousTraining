"""Summarize a sweep: python scripts/summarize_sweep.py sweep_tanh.csv [--activation tanh]

Reads the CSV written by `dvc exp show --csv`, prints every configuration's validation and test MAE,
selects the configuration with the lowest validation MAE (test is never used for selection) and
gives the verdict against the paper's cross-subject test MAE of 16.71 degrees.
"""

import argparse
import csv
import math
import os
import re
import sys

PAPER_MAE = 16.71
TOLERANCE = 0.5  # degrees above the paper that still counts as a threshold result

VAL = "val.mae.mean"
TEST = "test.mae.mean"
EXPECTED_CONFIGS = 7  # k=1 with 3 strides, k=5 with 4 strides
COLUMNS = (VAL, TEST, "train.k", "train.example_stride", "train.activation")


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


def read_rows(path: str) -> list[dict]:
    """Rows with the canonical column names, all activations.

    Rows without a finite validation MAE (baseline workspace row, failed experiments) are skipped.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        names = {c: _column(header, c) for c in COLUMNS}
        missing = [c for c, h in names.items() if h is None]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        rows = []
        for raw in reader:
            row = {c: raw[h] for c, h in names.items()}
            for c in (VAL, TEST):
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
    parser.add_argument("csv", help="sweep_<activation>.csv")
    parser.add_argument("--activation", help="default: taken from the csv file name")
    args = parser.parse_args()
    activation = args.activation or activation_from_filename(args.csv)
    if activation is None:
        parser.error("pass --activation (the file name is not sweep_<activation>.csv)")
    try:
        main(args.csv, activation)
    except ValueError as e:
        sys.exit(str(e))
