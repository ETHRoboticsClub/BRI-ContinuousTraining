import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.train import load_data, run
from src.dataset import recording_path, save_recording

PARAMS = {
    **yaml.safe_load((Path(__file__).resolve().parent.parent / "src/params.yaml").read_text())["train"],
    "val_subjects": [3],
    "test_subjects": [4],
    "epochs": 1,
    "batch_size": 8,
    "k": 1,
    "seeds": [0, 1],
    "device": "cpu",
}


@pytest.fixture
def data_dir(tmp_path):
    """Synthetic recordings for subjects 1-4, 3000 raw samples each, subject s with EMG std s."""
    d = tmp_path / "data"
    d.mkdir()
    for s in range(1, 5):
        rng = np.random.default_rng(s)
        emg = s * rng.standard_normal((8, 3000))  # subjects differ in scale, so pooled and per-subject stats differ
        save_recording(recording_path(d, s), emg, rng.standard_normal((15, 3000)))
    return d


def test_run_writes_aggregate_metrics(tmp_path, data_dir):
    m = run(PARAMS, data_dir, tmp_path / "out")
    assert m["kernel_length"] == 10 and m["example_length"] == 240
    assert set(m["seeds"]) == {"0", "1"}
    for split in ("val", "test"):
        assert set(m[split]) == {"mse", "r2", "nmse", "mae"}
        assert m[split]["mae"]["mean"] == pytest.approx(np.mean([m["seeds"][s][split]["mae"] for s in ("0", "1")]))
        assert m[split]["mae"]["std"] == pytest.approx(
            np.std([m["seeds"][s][split]["mae"] for s in ("0", "1")], ddof=1)
        )
    assert json.loads((tmp_path / "out/metrics.json").read_text()) == m
    assert (tmp_path / "out/seed_0/best_model.pt").exists()


def test_eval_sets_use_stride_40(data_dir):
    _, _, val_loader, test_loader = load_data({**PARAMS, "example_stride": 1040}, data_dir, 240)
    assert len(val_loader.dataset) == (3000 - 240) // 40 + 1
    assert len(test_loader.dataset) == (3000 - 240) // 40 + 1


def test_set_normalization_changes_training_only(data_dir):
    sets = {n: load_data({**PARAMS, "normalization": n}, data_dir, 240) for n in ("subject", "set")}
    for loader in (2, 3):  # val, test: one subject each, so pooled == per-subject
        assert torch.equal(sets["subject"][loader].dataset[0][0], sets["set"][loader].dataset[0][0])
    assert not torch.equal(sets["subject"][0][0][0], sets["set"][0][0][0])


def test_only_training_examples_are_augmented(data_dir):
    train_set, _, val_loader, test_loader = load_data({**PARAMS, "aug_gain": 0.2}, data_dir, 240)
    assert not torch.equal(train_set[0][0], train_set[0][0])
    for ds in (val_loader.dataset, test_loader.dataset):
        assert ds.augment is None and torch.equal(ds[0][0], ds[0][0])


EXTRA_CHANGES = [
    {"aug_gain": 0.2},
    {"aug_rotate": 1},
    {"aug_shift": 0.5},
    {"downsample": 2},
    {"downsample": 4, "k": 2},  # L = 20: 4 must divide the kernel length
    {"ssm_speed": 1},
    {"ssm_speed": 0.25},
    {"ssm_lr_scale": 0.1},
    {"ssm_decay_A": False},
]


@pytest.mark.parametrize("change", EXTRA_CHANGES)
def test_run_with_each_extra_change(tmp_path, data_dir, change):
    m = run({**PARAMS, "seeds": [0], **change}, data_dir, tmp_path / "out")
    assert np.isfinite(m["val"]["mae"]["mean"])


def count_fit(monkeypatch):
    """Patch scripts.train.fit to count its calls; returns the list it appends to."""
    import scripts.train as train_module

    calls, real_fit = [], train_module.fit
    monkeypatch.setattr(train_module, "fit", lambda *a, **k: calls.append(1) or real_fit(*a, **k))
    return calls


def test_run_resumes_finished_seeds(tmp_path, data_dir, monkeypatch):
    out = tmp_path / "out"
    first = run(PARAMS, data_dir, out, resume=True)
    (out / "metrics.json").unlink()
    (out / "seed_1/result.json").unlink()  # as if the laptop turned off while training seed 1
    calls = count_fit(monkeypatch)
    second = run(PARAMS, data_dir, out, resume=True)
    assert len(calls) == 1
    assert second["seeds"]["0"] == first["seeds"]["0"]
    assert {r["seed"] for r in csv.DictReader(open(out / "history.csv"))} == {"0", "1"}


def test_resume_ignores_results_from_other_params(tmp_path, data_dir, monkeypatch):
    run(PARAMS, data_dir, tmp_path / "out", resume=True)
    calls = count_fit(monkeypatch)
    run({**PARAMS, "aug_gain": 0.2}, data_dir, tmp_path / "out", resume=True)
    assert len(calls) == 2


def test_run_without_resume_retrains(tmp_path, data_dir, monkeypatch):
    run(PARAMS, data_dir, tmp_path / "out")
    calls = count_fit(monkeypatch)
    run(PARAMS, data_dir, tmp_path / "out")
    assert len(calls) == 2
