"""Summarize a sweep: python scripts/summarize_sweep.py sweep_tanh.csv

Reads the CSV written by `dvc exp show --csv`, prints every configuration's validation and test MAE,
selects the configuration with the lowest validation MAE (test is never used for selection) and
gives the verdict against the paper's cross-subject test MAE of 16.71 degrees.
"""

import csv
import sys

PAPER_MAE = 16.71
TOLERANCE = 0.5  # degrees above the paper that still counts as a threshold result

VAL = "val.mae.mean"
TEST = "test.mae.mean"
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: str) -> list[dict]:
    """Rows with the canonical column names; dvc prefixes columns with the file path, so match by suffix.

    Rows without a validation MAE (baseline workspace row, failed experiments) are skipped.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        names = {c: next((h for h in header if h.endswith(c)), None) for c in COLUMNS}
        rows = []
        for raw in reader:
            row = {c: raw[h] if h else "" for c, h in names.items()}
            for c in (VAL, TEST):
                row[c] = _number(row[c])
            if row[VAL] is not None:
                rows.append(row)
    return rows


def main(path: str) -> None:
    rows = read_rows(path)
    if not rows:
        sys.exit(f"no experiment with {VAL} in {path}")
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
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/summarize_sweep.py sweep_<activation>.csv")
    main(sys.argv[1])
