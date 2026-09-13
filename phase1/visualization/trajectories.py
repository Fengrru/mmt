"""Plot latent trajectories to make 'same gesture, different realization' visible."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pca2(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    zc = z - z.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    return zc @ vt[:2].T


def _smooth(z: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or z.shape[0] <= window:
        return z
    kernel = np.ones(window) / window
    return np.stack(
        [np.convolve(z[:, d], kernel, mode="same") for d in range(z.shape[1])], axis=1
    )


def _coarse(z: np.ndarray, length: int | None) -> np.ndarray:
    """Average contiguous time bins to a coarse trajectory of `length` points."""
    if length is None or z.shape[0] <= length:
        return z
    edges = np.linspace(0, z.shape[0], length + 1).astype(int)
    out = []
    for i in range(length):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        out.append(z[a:b].mean(axis=0))
    return np.stack(out)


def plot_trajectories(
    trajectories: dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "Latent trajectories",
    project_2d: bool = True,
    annotate_start: bool = True,
    smooth: int = 1,
    stride: int = 1,
    resample_length: int | None = None,
):
    """Plot several trajectories in a *shared* 2D projection.

    A single PCA basis is fit on all frames so that different trajectories are
    directly comparable; per-trajectory PCA would give each curve its own axes
    and is therefore useless for comparing realizations of the same motion.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = {}
    for name, z in trajectories.items():
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = z[:, None]
        z = _smooth(z, smooth)
        z = _coarse(z, resample_length)
        if stride > 1:
            z = z[::stride]
        arrays[name] = z

    if project_2d:
        concat = np.concatenate(list(arrays.values()), axis=0)
        mu = concat.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(concat - mu, full_matrices=False)
        basis = vt[:2].T

        def project(z: np.ndarray) -> np.ndarray:
            return (z - mu) @ basis
    else:

        def project(z: np.ndarray) -> np.ndarray:
            return z

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, z in arrays.items():
        pts = project(z)
        if pts.shape[1] < 2:
            pts = np.pad(pts, ((0, 0), (0, 2 - pts.shape[1])))
        ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=3, label=name)
        ax.scatter(pts[0, 0], pts[0, 1], marker="s", s=40)
        if annotate_start:
            ax.annotate("start", (pts[0, 0], pts[0, 1]), fontsize=6, alpha=0.6)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlabel("shared dim 1")
    ax.set_ylabel("shared dim 2")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_dimension_curves(z: np.ndarray, out_path: str | Path, max_dims: int = 8, title: str = "z(t)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(z, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    for d in range(min(max_dims, z.shape[1])):
        ax.plot(z[:, d], label=f"dim {d}")
    ax.set_xlabel("time step")
    ax.set_title(title)
    if z.shape[1] <= max_dims:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
