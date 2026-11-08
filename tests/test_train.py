import json
from pathlib import Path

import numpy as np
import pytest
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
    """Synthetic recordings for subjects 1-4, 3000 raw samples each."""
    d = tmp_path / "data"
    d.mkdir()
    for s in range(1, 5):
        rng = np.random.default_rng(s)
        save_recording(recording_path(d, s), rng.standard_normal((8, 3000)), rng.standard_normal((15, 3000)))
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
