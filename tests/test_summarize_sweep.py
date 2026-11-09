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


def test_dvc_367_header_without_file_prefix(tmp_path):
    header = (
        "Experiment,rev,typ,Created,parent,State,Executor,kernel_length,example_length,"
        "val.mae.mean,val.mae.std,test.mae.mean,test.mae.std,"
        "train.k,train.example_stride,train.activation"
    )
    lines = [
        "workspace,abc,baseline,-,-,-,-,!,!,!,!,!,!,5,40,relu",
        "e1,r1,branch_commit,-,-,Success,-,100,200,17.0,0.1,17.5,0.1,1,40,relu",
        "e2,r2,branch_commit,-,-,Success,-,100,200,16.0,0.1,16.4,0.1,5,240,relu",
        "e3,r3,branch_commit,-,-,Success,-,100,200,10.0,0.1,11.0,0.1,1,40,tanh",
        "e4,r4,baseline,-,-,-,-,,,,,,,5,40,relu",
    ]
    path = tmp_path / "sweep_relu.csv"
    path.write_text("\n".join([header, *lines]) + "\n")
    rows = for_activation(read_rows(str(path)), "relu")
    assert [r["train.k"] for r in rows] == ["1", "5"]
    assert select(rows)["test.mae.mean"] == 16.4
