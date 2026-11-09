import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts/sweep_extra.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def sweep(*args, base_sets=""):
    """Run the sweep script with DRY_RUN=1; returns (exit code, stdout lines)."""
    env = {**os.environ, "DRY_RUN": "1", "BASE_SETS": base_sets}
    result = subprocess.run([BASH, str(SCRIPT), *args], env=env, capture_output=True, text=True)
    return result.returncode, result.stdout.splitlines()


def test_sweep_extra_dry_run():
    code, out = sweep("tanh", "5", "40")
    assert code == 0
    assert len(out) == 11 and all(line.startswith("dvc exp run -n x-tanh-k5-s40-") for line in out)
    assert all(
        "src/params.yaml:train.activation=tanh" in line
        and "src/params.yaml:train.k=5" in line
        and "src/params.yaml:train.example_stride=40" in line
        for line in out
    )
    assert sum("src/params.yaml:train.aug_gain=0.2" in line for line in out) == 1
    names = [line.split()[4] for line in out]
    assert len(set(names)) == 11 and "x-tanh-k5-s40-baseline" in names


def test_sweep_extra_passes_base_sets():
    code, out = sweep("relu", "1", "40", base_sets="weight_decay=0.05 lr=0.005")
    assert code == 0 and len(out) == 11
    assert all("-S src/params.yaml:train.weight_decay=0.05 -S src/params.yaml:train.lr=0.005" in line for line in out)


@pytest.mark.parametrize("base_sets", ["downsample=2", "k=1", "lr=0.005 ssm_decay_A=false"])
def test_sweep_extra_rejects_conflicting_base(base_sets):
    code, out = sweep("tanh", "5", "40", base_sets=base_sets)
    assert code == 1 and out == []


STUB_DVC = """#!/usr/bin/env bash
# Stand-in for dvc: on `exp run`, log params.yaml as the run sees it, then leave an edit behind like dvc's -S does.
if [ "$1 $2" = "exp run" ]; then
  { echo "--- $4"; cat src/params.yaml; } >> "$DVC_LOG"
  echo "  leftover_from_$4: 1" >> src/params.yaml
fi
"""


def test_each_experiment_starts_from_the_original_params(tmp_path):
    """dvc exp run -S writes into the workspace params.yaml and leaves it there; no change may reach the next run."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "bin").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts/sweep_extra.sh")
    params = "train:\n  aug_gain: 0.0\n"
    (tmp_path / "src/params.yaml").write_text(params, newline="\n")
    (tmp_path / "bin/dvc").write_text(STUB_DVC, newline="\n")
    log = tmp_path / "dvc.log"
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}", "DVC_LOG": str(log)}
    result = subprocess.run(
        [BASH, str(tmp_path / "scripts/sweep_extra.sh"), "tanh", "5", "40"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    runs = log.read_text().split("--- ")[1:]
    assert len(runs) == 11
    for run in runs:
        assert run.split("\n", 1)[1] == params, run.split("\n", 1)[0]
    assert (tmp_path / "src/params.yaml").read_text() == params  # restored on exit


def test_sweep_extra_usage():
    assert sweep("tanh", "5")[0] == 1
    assert sweep("gelu", "5", "40")[0] == 1
