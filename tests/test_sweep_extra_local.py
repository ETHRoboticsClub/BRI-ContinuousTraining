import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import scripts.sweep_extra_local as local
from scripts.sweep_extra_local import CHANGES, experiment_params, main

SCRIPT = Path(__file__).resolve().parent.parent / "scripts/sweep_extra.sh"
BASE = {"activation": "tanh", "k": 5, "example_stride": 40, "aug_gain": 0.0, "downsample": 1, "lr": 2e-4}


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_changes_match_the_dvc_sweep():
    env = {**os.environ, "DRY_RUN": "1", "BASE_SETS": ""}
    out = subprocess.run(
        [shutil.which("bash"), str(SCRIPT), "relu", "10", "40"], env=env, capture_output=True, text=True
    ).stdout.splitlines()
    from_bash = []
    for line in out:
        words = line.split()
        label = words[4].removeprefix("x-relu-k10-s40-")
        sets = [w.removeprefix("src/params.yaml:train.") for w in words[6::2]][3:]  # after activation, k, stride
        from_bash.append((label, dict(s.split("=", 1) for s in sets)))
    assert [(label, {k: str(v).lower() for k, v in change.items()}) for label, change in CHANGES] == from_bash


def test_experiment_params_change_one_key():
    p = experiment_params(BASE, "relu", 10, 40, {"aug_gain": 0.2})
    assert p == {**BASE, "activation": "relu", "k": 10, "example_stride": 40, "aug_gain": 0.2}
    assert BASE["aug_gain"] == 0.0  # the base is not modified


def fake_run(calls):
    """Stand-in for scripts.train.run: records the call and writes a metrics.json."""

    def run(params, data_dir, out_dir, resume=False):
        calls.append((Path(out_dir).name, params, resume))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "metrics.json").write_text("{}")
        return {}

    return run


def test_main_runs_from_start_label_and_skips_finished(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(local, "run", fake_run(calls))
    monkeypatch.setattr(local, "load_train_params", lambda: dict(BASE))
    main(["relu", "10", "40", "--start", "aug_gain", "--out", str(tmp_path)])
    assert [c[0] for c in calls] == [label for label, _ in CHANGES[1:]]
    assert all(resume for _, _, resume in calls)
    assert json.loads((tmp_path / "base_params.json").read_text())["k"] == 10
    assert json.loads((tmp_path / "aug_gain/params.json").read_text())["aug_gain"] == 0.2
    calls.clear()
    main(["relu", "10", "40", "--start", "aug_gain", "--out", str(tmp_path)])
    assert calls == []  # every experiment finished: nothing reruns


def test_main_reruns_an_unfinished_experiment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(local, "run", fake_run(calls))
    monkeypatch.setattr(local, "load_train_params", lambda: dict(BASE))
    main(["relu", "10", "40", "--out", str(tmp_path)])
    (tmp_path / "speed_1/metrics.json").unlink()  # interrupted before it finished
    calls.clear()
    main(["relu", "10", "40", "--out", str(tmp_path)])
    assert [c[0] for c in calls] == ["speed_1"]


@pytest.mark.parametrize("k", ["5", "15"])
def test_main_refuses_downsample_that_does_not_divide_L(tmp_path, monkeypatch, k):
    calls = []
    monkeypatch.setattr(local, "run", fake_run(calls))
    monkeypatch.setattr(local, "load_train_params", lambda: dict(BASE))
    with pytest.raises(SystemExit):
        main(["relu", k, "40", "--out", str(tmp_path)])
    assert calls == []


def test_unknown_start_label_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "load_train_params", lambda: dict(BASE))
    with pytest.raises(SystemExit):
        main(["relu", "10", "40", "--start", "aug_gian", "--out", str(tmp_path)])
