import pytest

from scripts.summarize_sweep import (
    activation_from_filename,
    for_activation,
    read_rows,
    select,
    verdict,
)


def test_select_uses_validation_not_test():
    rows = [
        {"val.mae.mean": 17.0, "test.mae.mean": 15.0},
        {"val.mae.mean": 16.0, "test.mae.mean": 18.0},
    ]
    assert select(rows) is rows[1]


@pytest.mark.parametrize(
    "mae,expected",
    [(16.70, "success"), (16.71, "threshold"), (17.21, "threshold"), (17.22, "miss")],
)
def test_verdict(mae, expected):
    assert verdict(mae) == expected


HEADER = (
    "Experiment,output/metrics.json:val.mae.mean,output/metrics.json:test.mae.mean,"
    "src/params.yaml:train.k,src/params.yaml:train.example_stride,src/params.yaml:train.activation"
)


def _write(tmp_path, lines, name="sweep_relu.csv"):
    path = tmp_path / name
    path.write_text("\n".join([HEADER, *lines]) + "\n")
    return str(path)


def test_relu_selection_ignores_tanh_rows(tmp_path):
    path = _write(
        tmp_path,
        [
            "workspace,,,5,40,relu",
            "a,10.0,11.0,1,40,tanh",
            "b,17.0,17.5,1,40,relu",
            "c,16.5,16.0,5,240,relu",
            "d,16.0,nan,5,240,relu",  # duplicate configuration: the first one is kept
            "e,inf,1.0,5,40,relu",  # non-finite is missing
        ],
    )
    rows = for_activation(read_rows(path), "relu")
    assert [r["train.k"] for r in rows] == ["1", "5"]
    assert select(rows)["test.mae.mean"] == 16.0
    assert activation_from_filename(path) == "relu"
    assert activation_from_filename("results.csv") is None


def test_missing_column_is_named(tmp_path):
    path = tmp_path / "sweep_relu.csv"
    path.write_text("Experiment,output/metrics.json:val.mae.mean\nx,1.0\n")
    with pytest.raises(ValueError, match="test.mae.mean"):
        read_rows(str(path))


def test_column_match_is_exact(tmp_path):
    path = tmp_path / "sweep_relu.csv"
    path.write_text(HEADER.replace("train.k", "xtrain.k") + "\n")
    with pytest.raises(ValueError, match="train.k"):
        read_rows(str(path))
