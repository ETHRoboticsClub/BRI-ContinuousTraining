import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from src.dataset import EMGFK, kernel_length, recording_path, save_recording, shortest_recording
from src.model.tangent_space import n_steps


def make(tmp_path, s, T):
    """Write a seeded synthetic recording for subject s with T raw samples and return (emg, angles)."""
    rng = np.random.default_rng(s)
    emg = rng.standard_normal((8, T)).astype(np.float32)
    angles = rng.standard_normal((15, T)).astype(np.float32)
    save_recording(recording_path(tmp_path, s), emg, angles)
    return emg, angles


def test_k1_stride40_matches_old_windows(tmp_path):  # the pre-change build logic
    emg, angles = make(tmp_path, 1, T=2000)
    ds = EMGFK(tmp_path, [1], 240, 40)
    old_X = sliding_window_view(emg, 240, axis=1)[:, ::40].transpose(1, 0, 2)
    old_y = angles[:, 239::40].T
    assert len(ds) == len(old_X)
    for i in range(len(ds)):
        assert np.array_equal(ds[i][0].numpy(), old_X[i]) and np.array_equal(ds[i][1].numpy(), old_y[i])


@pytest.mark.parametrize("length,stride", [(1040, 40), (1040, 520), (1040, 1040), (240, 120)])
def test_counts_and_alignment(tmp_path, length, stride):
    emg, angles = make(tmp_path, 1, T=5000)
    ds = EMGFK(tmp_path, [1], length, stride)
    assert len(ds) == (5000 - length) // stride + 1
    e = length - 1 + 3 * stride
    assert np.array_equal(ds[3][0].numpy(), emg[:, e - length + 1 : e + 1])
    assert np.array_equal(ds[3][1].numpy(), angles[:, e])


def test_subject_filter_and_concat(tmp_path):
    emg1, angles1 = make(tmp_path, 1, T=3000)
    emg2, angles2 = make(tmp_path, 2, T=4000)
    make(tmp_path, 3, T=5000)
    ds = EMGFK(tmp_path, [1, 2], 1040, 40)
    n1, n2 = (3000 - 1040) // 40 + 1, (4000 - 1040) // 40 + 1
    assert (ds.channels, ds.targets) == (8, 15)
    assert len(ds) == n1 + n2
    assert np.array_equal(ds[n1 - 1][1].numpy(), angles1[:, 1039 + (n1 - 1) * 40])
    assert np.array_equal(ds[n1][0].numpy(), emg2[:, :1040])
    assert np.array_equal(ds[n1][1].numpy(), angles2[:, 1039])


def test_short_recording_yields_no_examples(tmp_path):
    make(tmp_path, 1, T=500)
    make(tmp_path, 2, T=3000)
    ds = EMGFK(tmp_path, [1, 2], 1040, 40)
    assert len(ds) == (3000 - 1040) // 40 + 1


def test_shortest_recording(tmp_path):
    make(tmp_path, 1, T=3000)
    make(tmp_path, 2, T=4000)
    assert shortest_recording(tmp_path) == 3000


def test_kernel_length_caps_and_warns():
    assert kernel_length(5, 360_000) == 50
    with pytest.warns(UserWarning):
        assert kernel_length(5, 500) == n_steps(500)
