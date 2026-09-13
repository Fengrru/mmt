import numpy as np

from phase1.experiments.common import (
    distance_separation,
    resample_trajectory,
    trajectory_distance,
)


def _traj(n=64, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(n, d)), axis=0)


def test_resample_trajectory_length():
    z = _traj(50, 3)
    assert resample_trajectory(z, 20).shape == (20, 3)


def test_identical_distance_zero():
    z = _traj()
    assert abs(trajectory_distance(z, z, mode="norm")) < 1e-9
    assert abs(trajectory_distance(z, z, mode="raw")) < 1e-9


def test_norm_distance_is_speed_invariant():
    z = _traj(40, 3)
    stretched = resample_trajectory(z, 120)
    d_norm = trajectory_distance(z, stretched, mode="norm")
    d_raw = trajectory_distance(z, stretched, mode="raw")
    assert d_norm < 0.2 * d_raw
    other = _traj(40, 3, seed=7)
    assert d_norm < trajectory_distance(z, other, mode="norm")


def test_dtw_finite_and_prefers_close():
    z = _traj(30, 2, seed=1)
    same = _traj(30, 2, seed=1)
    other = _traj(30, 2, seed=2)
    d_close = trajectory_distance(z, same, mode="dtw")
    d_far = trajectory_distance(z, other, mode="dtw")
    assert np.isfinite(d_close) and np.isfinite(d_far)
    assert d_close < d_far


def test_distance_separation_auc():
    preserving = [0.1, 0.2, 0.15, 0.3]
    other = [1.0, 1.2, 0.9, 1.1]
    out = distance_separation(preserving, other)
    assert out["auc"] == 1.0
    assert out["gap"] > 0
