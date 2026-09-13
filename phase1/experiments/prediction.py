"""Multi-step trajectory prediction.

Compares a learned temporal trajectory against persistence, linear
extrapolation, a ridge predictor in frozen-SSL space, and a constant predictor.
Splits are by source segment to avoid window leakage.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ..data.manifest import Segment, read_jsonl
from ..representation.losses import vicreg_loss
from ..representation.trajectory_encoder import TrajectoryEncoder
from .common import dump_json, encode_cache, resolve_encoder


def _standardize(feats, mean, std):
    return {k: (v - mean) / std for k, v in feats.items()}


def _make_windows(feats: dict[str, np.ndarray], k: int, min_context: int, stride: int, rng, max_windows: int):
    windows = []
    for sid, x in feats.items():
        if x.shape[0] < min_context + k:
            continue
        for t in range(min_context, x.shape[0] - k + 1, stride):
            windows.append((sid, x[:t], x[t : t + k]))
    rng.shuffle(windows)
    return windows[:max_windows]


def _fit_linear_predictor(train, k):
    from sklearn.linear_model import Ridge

    x = np.stack([w[1][-1] for w in train])
    models = []
    for i in range(k):
        y = np.stack([w[2][i] for w in train])
        m = Ridge(alpha=1.0).fit(x, y)
        models.append(m)
    return models


def _predict_baselines(train, val, k, global_mean, ridge_models, last_k=5):
    out = {}
    preds = {"persistence": [], "linear_extrapolation": [], "ridge_frozen_feature": [], "constant_mean": []}
    for _, ctx, tgt in val:
        preds["persistence"].append(np.repeat(ctx[-1][None], k, axis=0))
        L = min(last_k, ctx.shape[0])
        xs = np.arange(L, dtype=float)
        slopes = np.polyfit(xs, ctx[-L:], 1)
        future = np.arange(L, L + k, dtype=float)
        lin = (slopes[0][:, None] * future[None, :] + slopes[1][:, None]).T
        preds["linear_extrapolation"].append(lin)
        x_last = ctx[-1][None]
        preds["ridge_frozen_feature"].append(np.stack([m.predict(x_last)[0] for m in ridge_models]))
        preds["constant_mean"].append(np.repeat(global_mean[None], k, axis=0))
    for name, ps in preds.items():
        p = np.stack(ps)
        t = np.stack([w[2] for w in val])
        out[name] = float(np.mean((p - t) ** 2))
    return out


def _train_learned(train, val, k, input_dim, epochs, lr, out_dim, kind, seed):
    torch.manual_seed(seed)
    encoder = TrajectoryEncoder(input_dim=input_dim, out_dim=out_dim)
    head = torch.nn.Linear(out_dim, k * input_dim)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    def batchify(windows, max_len=64):
        ctxs, tgts = [], []
        for _, ctx, tgt in windows:
            c = ctx[-max_len:]
            ctxs.append(torch.tensor(c, dtype=torch.float32))
            tgts.append(torch.tensor(tgt, dtype=torch.float32))
        x = torch.nn.utils.rnn.pad_sequence(ctxs, batch_first=True)
        mask = torch.zeros(x.shape[:2], dtype=torch.bool)
        for i, c in enumerate(ctxs):
            mask[i, c.shape[0] :] = True
        y = torch.stack(tgts)
        return x, mask, y

    xtr, mtr, ytr = batchify(train)
    xva, mva, yva = batchify(val)
    for _ in range(epochs):
        encoder.train()
        opt.zero_grad()
        z = encoder(xtr, mtr)
        flat = z.reshape(-1, z.shape[-1])
        vc, _ = vicreg_loss(flat)
        h = z[:, -1]
        pred = head(h).reshape(-1, k, input_dim)
        mse = torch.nn.functional.mse_loss(pred, ytr)
        loss = mse + 0.1 * vc
        loss.backward()
        opt.step()
    encoder.eval()
    with torch.no_grad():
        z = encoder(xva, mva)
        pred = head(z[:, -1]).reshape(-1, k, input_dim)
        learned_mse = float(torch.nn.functional.mse_loss(pred, yva))
    return learned_mse


def run(
    encoder_name: str,
    stimuli_dir: str | Path,
    out_path: str | Path | None = None,
    k: int = 4,
    max_seconds: float = 12.0,
    limit: int | None = None,
    epochs: int = 30,
    seed: int = 0,
) -> dict:
    stimuli_dir = Path(stimuli_dir)
    segments = read_jsonl(stimuli_dir / "segments.jsonl", cls=Segment)
    if limit:
        segments = segments[:limit]
    encoder = resolve_encoder(encoder_name)
    feats = encode_cache(encoder, segments, stimuli_dir / "cache" / f"{encoder_name}_seg.npz", max_seconds)
    feats = {k_: v for k_, v in feats.items() if v.size}

    all_x = np.concatenate(list(feats.values()), axis=0)
    mean, std = all_x.mean(axis=0), all_x.std(axis=0) + 1e-6
    feats = _standardize(feats, mean, std)
    input_dim = next(iter(feats.values())).shape[1]

    rng = random.Random(seed)
    ids = sorted(feats)
    rng.shuffle(ids)
    n_val = max(1, int(0.2 * len(ids)))
    val_ids, train_ids = set(ids[:n_val]), set(ids[n_val:])
    train_feats = {k_: v for k_, v in feats.items() if k_ in train_ids}
    val_feats = {k_: v for k_, v in feats.items() if k_ in val_ids}

    w_rng = random.Random(seed)
    train_w = _make_windows(train_feats, k, 3, 2, w_rng, 4000)
    val_w = _make_windows(val_feats, k, 3, 2, w_rng, 1000)
    if not train_w or not val_w:
        raise ValueError("not enough long-enough segments for prediction; increase segment duration")

    global_mean = np.stack([w[2] for w in train_w]).reshape(-1, input_dim).mean(axis=0)
    ridge = _fit_linear_predictor(train_w, k)
    baselines = _predict_baselines(train_w, val_w, k, global_mean, ridge)
    learned = _train_learned(train_w, val_w, k, input_dim, epochs, 1e-3, 64, "transformer", seed)

    best_baseline = min(baselines.items(), key=lambda kv: kv[1])
    result = {
        "encoder": encoder_name,
        "k": k,
        "input_dim": input_dim,
        "n_train_windows": len(train_w),
        "n_val_windows": len(val_w),
        "baseline_mse": baselines,
        "learned_mse": learned,
        "best_baseline": best_baseline[0],
        "best_baseline_mse": best_baseline[1],
        "prediction_supported": bool(learned < best_baseline[1]),
    }
    if out_path is not None:
        dump_json(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-step trajectory prediction")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    res = run(args.encoder, args.stimuli, args.out, k=args.k, epochs=args.epochs, limit=args.limit)
    dump_json(args.out or (Path(args.stimuli) / f"prediction_{args.encoder}.json"), res)
    print(f"learned={res['learned_mse']:.4f} best_baseline={res['best_baseline']}={res['best_baseline_mse']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
