import numpy as np

from phase1.visualization import plot_motion_curves, plot_trajectories


def test_plot_trajectories_shared_projection(tmp_path):
    rng = np.random.default_rng(0)
    trajs = {"a": rng.normal(size=(20, 6)), "b": rng.normal(size=(25, 6))}
    out = plot_trajectories(trajs, tmp_path / "t.png", resample_length=8, smooth=3)
    assert out.exists() and out.stat().st_size > 0


def test_plot_motion_curves(tmp_path):
    sr = 22050
    y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    out = plot_motion_curves({"a": y}, sr, tmp_path / "m.png")
    assert out.exists() and out.stat().st_size > 0
