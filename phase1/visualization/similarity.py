"""RDM heatmaps and MDS embeddings for model-vs-human geometry comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_rdm(rdm_condensed: np.ndarray, out_path: str | Path, labels: list[str] | None = None, title: str = "RDM"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import squareform

    m = squareform(np.asarray(rdm_condensed, dtype=float))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(m, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8)
    if labels is not None and len(labels) <= 40:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_mds(embeddings: np.ndarray, out_path: str | Path, labels: list[str] | None = None, title: str = "MDS"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS

    x = np.asarray(embeddings, dtype=float)
    xy = MDS(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=0,
        n_init=4,
    ).fit_transform(_to_dissimilarity(x))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xy[:, 0], xy[:, 1], s=18)
    if labels is not None:
        for (x0, y0), lab in zip(xy, labels):
            ax.annotate(str(lab), (x0, y0), fontsize=6)
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _to_dissimilarity(x: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform

    if x.ndim == 2 and x.shape[0] == x.shape[1] and np.allclose(np.diag(x), 0):
        return x
    return squareform(pdist(x, metric="euclidean"))
