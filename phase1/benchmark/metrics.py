"""Agreement, correlation, RSA and collapse diagnostics.

Every quantity here answers a specific pre-registered question:
- agreement: can the human transformation ontology be trusted?
- RSA / correlation: does model geometry match human geometry?
- collapse: is the learned representation non-degenerate?
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist

__all__ = [
    "fleiss_kappa",
    "krippendorff_alpha",
    "bootstrap_ci",
    "spearman",
    "kendall",
    "triplet_accuracy",
    "rdm_from_embeddings",
    "rsa",
    "per_dim_variance",
    "effective_rank",
    "participation_ratio",
    "uniformity",
    "collapse_report",
]


def _as_counts(ratings: np.ndarray, categories: list | None = None) -> tuple[np.ndarray, list]:
    ratings = np.asarray(ratings)
    if ratings.ndim == 2 and categories is None:
        categories = list(range(ratings.shape[1]))
        return ratings.astype(float), categories
    if categories is None:
        categories = sorted({v for row in ratings for v in row if v is not None})
    idx = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((ratings.shape[0], len(categories)))
    for i, row in enumerate(ratings):
        for v in row:
            if v is not None:
                counts[i, idx[v]] += 1
    return counts, categories


def fleiss_kappa(ratings: np.ndarray, categories: list | None = None) -> float:
    counts, _ = _as_counts(ratings, categories)
    n_items, n_cat = counts.shape
    n_raters = counts.sum(axis=1)
    if n_items == 0 or np.any(n_raters < 2):
        return float("nan")
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss' kappa requires a constant number of raters per item")
    n = n_raters[0]
    p_i = (np.sum(counts**2, axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_items * n)
    p_e = np.sum(p_j**2)
    if np.isclose(p_e, 1.0):
        return 1.0
    return float((p_bar - p_e) / (1.0 - p_e))


def _ordinal_delta(marginals: np.ndarray) -> np.ndarray:
    n_cat = len(marginals)
    cum = np.cumsum(marginals)
    delta = np.zeros((n_cat, n_cat))
    for c in range(n_cat):
        for k in range(c, n_cat):
            if c == k:
                continue
            between = cum[k] - cum[c - 1] if c > 0 else cum[k]
            gap = between - 0.5 * (marginals[c] + marginals[k])
            delta[c, k] = delta[k, c] = gap**2
    return delta


def krippendorff_alpha(ratings: np.ndarray, level: str = "nominal") -> float:
    ratings = np.asarray(ratings, dtype=object)
    if level not in {"nominal", "ordinal"}:
        raise ValueError("level must be 'nominal' or 'ordinal'")
    values = sorted({v for row in ratings for v in row if v is not None})
    idx = {v: i for i, v in enumerate(values)}
    n_cat = len(values)
    if n_cat == 0:
        return float("nan")

    coincidences = np.zeros((n_cat, n_cat))
    for row in ratings:
        present = [idx[v] for v in row if v is not None]
        m = len(present)
        if m < 2:
            continue
        counts = np.bincount(present, minlength=n_cat).astype(float)
        for c in range(n_cat):
            if counts[c] == 0:
                continue
            coincidences[c, c] += counts[c] * (counts[c] - 1) / (m - 1)
            for k in range(n_cat):
                if k != c and counts[k] > 0:
                    coincidences[c, k] += counts[c] * counts[k] / (m - 1)

    n = coincidences.sum()
    if n <= 0:
        return float("nan")
    marginals = coincidences.sum(axis=1)
    if level == "nominal":
        delta = 1.0 - np.eye(n_cat)
    else:
        delta = _ordinal_delta(marginals)
    d_o = np.sum(coincidences * delta) / n
    d_e = np.sum(np.outer(marginals, marginals) * delta) / (n * (n - 1))
    if np.isclose(d_e, 0.0):
        return 1.0
    return float(1.0 - d_o / d_e)


def bootstrap_ci(values, statistic=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = statistic(values[idx], axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(values)), float(lo), float(hi)


def spearman(a, b) -> tuple[float, float]:
    rho, p = stats.spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel())
    return float(rho), float(p)


def kendall(a, b) -> tuple[float, float]:
    tau, p = stats.kendalltau(np.asarray(a).ravel(), np.asarray(b).ravel())
    return float(tau), float(p)


def triplet_accuracy(
    pos_scores, neg_scores, higher_is_more_similar: bool = False
) -> float:
    """Fraction of triplets where the positive is preferred.

    Default assumes distance-like scores (lower = better). Set
    `higher_is_more_similar=True` for similarity scores.
    """
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if pos.shape != neg.shape:
        raise ValueError("pos_scores and neg_scores must have the same shape")
    if pos.size == 0:
        return float("nan")
    correct = pos > neg if higher_is_more_similar else pos < neg
    return float(np.mean(correct) + 0.5 * np.mean(pos == neg))


def rdm_from_embeddings(x: np.ndarray, metric: str = "cosine") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return pdist(x, metric=metric)


def rsa(rdm_a: np.ndarray, rdm_b: np.ndarray, method: str = "spearman") -> tuple[float, float]:
    a = np.asarray(rdm_a, dtype=float).ravel()
    b = np.asarray(rdm_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("RDM lengths differ")
    if method == "spearman":
        return spearman(a, b)
    if method == "pearson":
        r, p = stats.pearsonr(a, b)
        return float(r), float(p)
    if method == "kendall":
        return kendall(a, b)
    raise ValueError("method must be spearman, pearson or kendall")


def per_dim_variance(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return z.var(axis=0)


def effective_rank(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    s = np.linalg.svd(z - z.mean(axis=0, keepdims=True), compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov)
    tr = np.trace(cov)
    tr2 = np.sum(cov**2)
    if tr2 <= 0:
        return 0.0
    return float(tr**2 / tr2)


def uniformity(z: np.ndarray, t: float = 2.0) -> float:
    z = np.asarray(z, dtype=float)
    if z.shape[0] < 2:
        return float("nan")
    sq = np.sum(z**2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * z @ z.T
    np.fill_diagonal(d2, 0.0)
    iu = np.triu_indices(z.shape[0], k=1)
    return float(np.log(np.mean(np.exp(-t * d2[iu]) + 1e-12)))


def collapse_report(z: np.ndarray, variance_floor: float = 1e-3, rank_frac_floor: float = 0.1) -> dict:
    z = np.asarray(z, dtype=float)
    var = per_dim_variance(z)
    rank = effective_rank(z)
    pr = participation_ratio(z)
    healthy = bool(np.mean(var) > variance_floor and rank > rank_frac_floor * z.shape[1])
    return {
        "n_samples": int(z.shape[0]),
        "dim": int(z.shape[1]),
        "mean_variance": float(var.mean()),
        "min_variance": float(var.min()),
        "frac_dims_above_floor": float(np.mean(var > variance_floor)),
        "effective_rank": float(rank),
        "participation_ratio": float(pr),
        "uniformity": uniformity(z),
        "healthy": healthy,
    }
