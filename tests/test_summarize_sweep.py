import pytest

from scripts.summarize_sweep import select, verdict


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
