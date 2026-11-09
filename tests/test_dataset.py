import numpy as np
import pytest
import torch
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
    ds = EMGFK(tmp_path, [1], 240, 40, normalization="none")
    old_X = sliding_window_view(emg, 240, axis=1)[:, ::40].transpose(1, 0, 2)
    old_y = angles[:, 239::40].T
    assert len(ds) == len(old_X)
    for i in range(len(ds)):
        assert np.array_equal(ds[i][0].numpy(), old_X[i]) and np.array_equal(ds[i][1].numpy(), old_y[i])


@pytest.mark.parametrize("length,stride", [(1040, 40), (1040, 520), (1040, 1040), (240, 120)])
def test_counts_and_alignment(tmp_path, length, stride):
    emg, angles = make(tmp_path, 1, T=5000)
    ds = EMGFK(tmp_path, [1], length, stride, normalization="none")
    assert len(ds) == (5000 - length) // stride + 1
    e = length - 1 + 3 * stride
    assert np.array_equal(ds[3][0].numpy(), emg[:, e - length + 1 : e + 1])
    assert np.array_equal(ds[3][1].numpy(), angles[:, e])


def test_subject_filter_and_concat(tmp_path):
    emg1, angles1 = make(tmp_path, 1, T=3000)
    emg2, angles2 = make(tmp_path, 2, T=4000)
    make(tmp_path, 3, T=5000)
    ds = EMGFK(tmp_path, [1, 2], 1040, 40, normalization="none")
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


def make_raw(tmp_path, s, T, scale):
    """Write an un-normalized recording (per-channel offset, std about `scale`) and return its EMG."""
    rng = np.random.default_rng(s)
    emg = (scale * rng.standard_normal((8, T)) + 1e-6 * rng.standard_normal((8, 1))).astype(np.float32)
    save_recording(recording_path(tmp_path, s), emg, np.zeros((15, T)))
    return emg


def test_subject_normalization_matches_old_build(tmp_path):
    raw = make_raw(tmp_path, 1, 3000, 3e-5)
    old = (raw - raw.mean(1, keepdims=True)) / (raw.std(1, keepdims=True) + 1e-6)  # the old build_dataset line
    ds = EMGFK(tmp_path, [1], 240, 40)  # default normalization
    np.testing.assert_allclose(ds[5][0].numpy(), old[:, 200:440], rtol=1e-6, atol=1e-7)  # e = 239 + 5 * 40


def test_set_normalization_pools_the_datasets_subjects(tmp_path):
    e1 = make_raw(tmp_path, 1, 3000, 1e-5)
    e2 = make_raw(tmp_path, 2, 3000, 4e-5)
    make_raw(tmp_path, 3, 3000, 1.0)  # not loaded, so not in the statistics
    ds = EMGFK(tmp_path, [1, 2], 240, 40, normalization="set")
    cat = np.concatenate([e1, e2], 1)
    expected = (e2 - cat.mean(1, keepdims=True)) / (cat.std(1, keepdims=True) + 1e-6)
    first_of_2 = (3000 - 240) // 40 + 1
    np.testing.assert_allclose(ds[first_of_2][0].numpy(), expected[:, :240], rtol=1e-5, atol=1e-6)


def test_set_equals_subject_for_one_subject(tmp_path):
    make_raw(tmp_path, 1, 3000, 3e-5)
    subject = EMGFK(tmp_path, [1], 240, 40, normalization="subject")
    pooled = EMGFK(tmp_path, [1], 240, 40, normalization="set")
    assert torch.equal(subject[7][0], pooled[7][0])


def test_none_returns_stored_emg(tmp_path):
    raw = make_raw(tmp_path, 1, 3000, 3e-5)
    assert np.array_equal(EMGFK(tmp_path, [1], 240, 40, normalization="none")[0][0].numpy(), raw[:, :240])


def test_unknown_normalization_raises(tmp_path):
    make_raw(tmp_path, 1, 3000, 3e-5)
    with pytest.raises(ValueError):
        EMGFK(tmp_path, [1], 240, 40, normalization="global")
