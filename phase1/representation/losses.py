"""Phase 1 objectives.

Anti-collapse is not optional: `variance_loss` and `covariance_loss` are always
kept in the total objective. Continuity is expressed as prediction error, never
as an L2 penalty on adjacent latents (which rewards a constant trajectory).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-4


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = _EPS) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    n, d = z.shape
    if n < 2 or d < 2:
        return z.new_zeros(())
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return off.pow(2).sum() / d


def vicreg_loss(z: torch.Tensor, var_coef: float = 25.0, cov_coef: float = 1.0, gamma: float = 1.0):
    v = variance_loss(z, gamma=gamma)
    c = covariance_loss(z)
    return var_coef * v + cov_coef * c, {
        "variance": float(v.detach()),
        "covariance": float(c.detach()),
    }


def invariance_loss(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(za, zb)


def alignment_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_zeros(())
    d = torch.cdist(z, z, p=2)
    n = z.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return d[iu[0], iu[1]].mean()


def multi_step_prediction_loss(
    pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    k = pred.shape[1]
    loss = (pred - target).pow(2).mean(dim=(0, 2))
    if weights is None:
        weights = 1.0 / torch.arange(1, k + 1, device=pred.device, dtype=pred.dtype)
    weights = weights.to(pred.device, pred.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(_EPS)


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
    metric: str = "euclidean",
) -> torch.Tensor:
    if metric == "euclidean":
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
    elif metric == "cosine":
        d_pos = 1.0 - F.cosine_similarity(anchor, positive)
        d_neg = 1.0 - F.cosine_similarity(anchor, negative)
    else:
        raise ValueError("metric must be euclidean or cosine")
    return F.relu(d_pos - d_neg + margin).mean()


def phase1_loss(
    z: torch.Tensor,
    z_aug: torch.Tensor,
    pred: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    anchor: torch.Tensor | None = None,
    positive: torch.Tensor | None = None,
    negative: torch.Tensor | None = None,
    weights: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    w = {"var": 25.0, "cov": 1.0, "inv": 1.0, "pred": 1.0, "triplet": 1.0}
    if weights:
        w.update(weights)
    parts: dict[str, torch.Tensor] = {}
    total = z.new_zeros(())

    vc, vc_parts = vicreg_loss(torch.cat([z, z_aug], dim=0), var_coef=w["var"], cov_coef=w["cov"])
    total = total + vc
    parts.update(vc_parts)

    inv = invariance_loss(z, z_aug)
    total = total + w["inv"] * inv
    parts["invariance"] = inv

    if pred is not None and target is not None:
        p = multi_step_prediction_loss(pred, target)
        total = total + w["pred"] * p
        parts["prediction"] = p

    if anchor is not None and positive is not None and negative is not None:
        t = triplet_loss(anchor, positive, negative)
        total = total + w["triplet"] * t
        parts["triplet"] = t

    parts = {k: float(v.detach()) for k, v in parts.items()}
    parts["total"] = float(total.detach())
    return total, parts
