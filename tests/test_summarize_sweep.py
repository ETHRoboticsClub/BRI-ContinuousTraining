import pytest

from scripts.summarize_sweep import (
    EXTRA_COLUMNS,
    activation_from_filename,
    base_from_filename,
    beats_baseline,
    extra_rows,
    for_activation,
    main_extra,
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


EXTRA_HEADER = "Experiment,rev,val.mae.mean,val.mae.std,test.mae.mean,train.k"


def _write_extra(tmp_path, lines, name="extra_tanh_k5_s40.csv"):
    path = tmp_path / name
    path.write_text("\n".join([EXTRA_HEADER, *lines]) + "\n")
    return str(path)


EXTRA_LINES = [
    "workspace,abc,!,!,!,5",
    "x-tanh-k5-s40-baseline,r1,17.0,0.3,17.5,5",
    "x-tanh-k5-s40-aug_gain,r2,16.6,0.2,17.9,5",
    "x-tanh-k5-s40-speed_1,r3,16.8,0.2,16.0,5",
    "x-tanh-k1-s40-aug_gain,r4,10.0,0.1,10.0,1",  # another base configuration
    "x-relu-k5-s40-aug_gain,r5,10.0,0.1,10.0,5",  # another activation
]


def test_extra_rows_by_experiment_name(tmp_path):
    rows = extra_rows(read_rows(_write_extra(tmp_path, EXTRA_LINES), EXTRA_COLUMNS), "tanh", "5", "40")
    assert set(rows) == {"baseline", "aug_gain", "speed_1"}
    assert rows["aug_gain"]["val.mae.mean"] == 16.6 and rows["baseline"]["val.mae.std"] == 0.3


def test_extra_rows_without_baseline_raise(tmp_path):
    path = _write_extra(tmp_path, [line for line in EXTRA_LINES if "baseline" not in line])
    with pytest.raises(ValueError, match="baseline"):
        extra_rows(read_rows(path, EXTRA_COLUMNS), "tanh", "5", "40")


def test_base_from_filename():
    assert base_from_filename("out/extra_tanh_k5_s40.csv") == ("tanh", "5", "40")
    assert base_from_filename("sweep_tanh.csv") is None


def test_beats_baseline():
    base = {"val.mae.mean": 17.0, "val.mae.std": 0.3}
    assert beats_baseline({"val.mae.mean": 16.6}, base)
    assert not beats_baseline({"val.mae.mean": 16.8}, base)


def test_extra_selection_ignores_test(tmp_path, capsys):
    main_extra(_write_extra(tmp_path, EXTRA_LINES), "tanh", "5", "40")
    out = capsys.readouterr().out
    assert "selected by validation: aug_gain" in out  # speed_1 has the best test MAE but a worse val MAE
    assert "-0.40" in out  # aug_gain's val delta vs the baseline
