import json

import numpy as np
import yaml

from src.dataset import build_dataset, make_loaders
from src.model import Model
from src.preprocessing import preprocess
from src.trainer import set_seed, train


def write_recordings(raw_dir, fs=2000, seconds=3, channels=8, targets=3):
    rng = np.random.default_rng(0)
    raw_dir.mkdir(parents=True)
    mixing = rng.uniform(0, 1, (channels, targets))
    t = np.arange(fs * seconds) / fs
    for movement in (1, 2):
        for repetition in range(1, 7):
            freq, phase = rng.uniform(0.1, 0.5, (targets, 1)), rng.uniform(0, 2 * np.pi, (targets, 1))
            act = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t + phase)
            emg = mixing @ (act * rng.standard_normal(act.shape)) + 0.05 * rng.standard_normal((channels, len(t)))
            np.savez(
                raw_dir / f"m{movement}_r{repetition}.npz",
                emg=emg.astype(np.float32),
                fs=fs,
                channel_names=[f"ch{i}" for i in range(channels)],
                targets=act.astype(np.float32),
                target_names=[f"actuator_{i}" for i in range(targets)],
                meta_subject=1,
                meta_exercise=1,
                meta_movement=movement,
                meta_repetition=repetition,
            )


def test_build_and_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = yaml.safe_load(open(__file__.replace("tests/test_pipeline.py", "params.yaml")))
    params["train"].update(epochs=2, batch_size=32)
    write_recordings(tmp_path / params["dataset"]["raw_dir"])

    build_dataset(params, preprocess)
    set_seed(0)
    train_loader, val_loader, test_loader = make_loaders(params)
    x, y = train_loader.dataset[0]
    assert x.shape == (8, params["dataset"]["window"]) and y.shape == (3,)

    model = Model(8, 3, **params["model"])
    results = train(model, train_loader, val_loader, params["train"]["out_dir"], test_loader, epochs=2)

    out = tmp_path / params["train"]["out_dir"]
    assert {"val", "test"} <= json.loads((out / "metrics.json").read_text()).keys()
    assert np.isfinite(results["test"]["r2"])
    probes = (out / "probes" / "scalars.csv").read_text()
    for name in ("features", "embedding", "output", "linear_probe_r2", "dead_channels", "sat_high"):
        assert name in probes
    assert (out / "probes" / "embedding" / "pca" / "best.png").exists()
