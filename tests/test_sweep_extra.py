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


def test_sweep_extra_usage():
    assert sweep("tanh", "5")[0] == 1
    assert sweep("gelu", "5", "40")[0] == 1
