import json

import pytest

from scripts.summarize_sweep import (
    EXTRA_COLUMNS,
    activation_from_filename,
    base_from_filename,
    beats_baseline,
    extra_rows,
    extra_rows_from_dirs,
    for_activation,
    main_extra,
    main_extra_dirs,
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


EXTRA_HEADER = (
    "Experiment,rev,val.mae.mean,val.mae.std,test.mae.mean,"
    "src/params.yaml:train.k,src/params.yaml:train.aug_gain,src/params.yaml:train.ssm_speed"
)


def _write_extra(tmp_path, lines, name="extra_tanh_k5_s40.csv"):
    path = tmp_path / name
    path.write_text("\n".join([EXTRA_HEADER, *lines]) + "\n")
    return str(path)


EXTRA_LINES = [
    "workspace,abc,!,!,!,5,0.0,4",
    "x-tanh-k5-s40-baseline,r1,17.0,0.3,17.5,5,0.0,4",
    "x-tanh-k5-s40-aug_gain,r2,16.6,0.2,17.9,5,0.2,4",
    "x-tanh-k5-s40-speed_1,r3,16.8,0.2,16.0,5,0.0,1",
    "x-tanh-k1-s40-aug_gain,r4,10.0,0.1,10.0,1,0.2,4",  # another base configuration
    "x-relu-k5-s40-aug_gain,r5,10.0,0.1,10.0,5,0.2,4",  # another activation
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
    assert "ssm_speed=1" in out  # the change each row made, read from its params


@pytest.mark.parametrize(
    "line,problem",
    [
        ("x-tanh-k5-s40-speed_1,r3,16.8,0.2,16.0,5,0.2,1", "aug_gain"),  # a previous change carried over
        ("x-tanh-k5-s40-speed_1,r3,16.8,0.2,16.0,5,0.0,4", "no change"),  # nothing differs from the baseline
    ],
)
def test_extra_rows_require_exactly_one_change(tmp_path, line, problem):
    lines = [x for x in EXTRA_LINES if "speed_1" not in x] + [line]
    with pytest.raises(ValueError, match=problem):
        extra_rows(read_rows(_write_extra(tmp_path, lines), EXTRA_COLUMNS), "tanh", "5", "40")


def write_experiment(d, val, std, test, params=None):
    """A finished experiment directory as scripts/train.py writes it (metrics.json), plus params.json if given."""
    d.mkdir(parents=True)
    split = lambda mae, sd: {"mae": {"mean": mae, "std": sd}}  # noqa: E731
    (d / "metrics.json").write_text(json.dumps({"val": split(val, std), "test": split(test, 0.0)}))
    if params is not None:
        (d / "params.json").write_text(json.dumps(params))


BASE_PARAMS = {"k": 10, "aug_gain": 0.0, "ssm_speed": 4}


def extra_dirs(tmp_path):
    """An extra sweep directory with two finished experiments and one unfinished, and a separate old baseline."""
    out = tmp_path / "extra_relu_k10_s40"
    out.mkdir()
    (out / "base_params.json").write_text(json.dumps(BASE_PARAMS))
    write_experiment(tmp_path / "k10_s40", 21.08, 0.55, 18.40)  # old baseline run: no params.json
    write_experiment(out / "aug_gain", 20.40, 0.30, 18.90, {**BASE_PARAMS, "aug_gain": 0.2})
    write_experiment(out / "speed_1", 21.30, 0.20, 17.00, {**BASE_PARAMS, "ssm_speed": 1})
    (out / "aug_rotate").mkdir()  # unfinished: no metrics.json
    return out, tmp_path / "k10_s40"


def test_extra_rows_from_dirs(tmp_path):
    rows = extra_rows_from_dirs(*extra_dirs(tmp_path))
    assert set(rows) == {"baseline", "aug_gain", "speed_1"}
    assert rows["baseline"]["val.mae.mean"] == 21.08 and rows["baseline"]["val.mae.std"] == 0.55
    assert rows["aug_gain"]["change"] == "aug_gain=0.2" and rows["speed_1"]["change"] == "ssm_speed=1"


def test_extra_rows_from_dirs_require_exactly_one_change(tmp_path):
    out, baseline = extra_dirs(tmp_path)
    (out / "speed_1/params.json").write_text(json.dumps({**BASE_PARAMS, "ssm_speed": 1, "aug_gain": 0.2}))
    with pytest.raises(ValueError, match="aug_gain"):
        extra_rows_from_dirs(out, baseline)


def test_directory_summary_selects_by_validation(tmp_path, capsys):
    out, baseline = extra_dirs(tmp_path)
    main_extra_dirs(out, baseline)
    printed = capsys.readouterr().out
    assert "selected by validation: aug_gain" in printed  # speed_1 has the best test MAE but a worse val MAE
    assert "-0.68" in printed  # aug_gain's val delta vs the baseline
