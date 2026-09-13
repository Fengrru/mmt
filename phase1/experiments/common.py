"""Shared experiment utilities: encoder resolution, trajectory distances, cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..baselines import AcousticTrajectory, CLAPBaseline, MERTBaseline
from ..data.audio_io import load_audio
from ..data.manifest import Segment, read_jsonl
from ..representation.ssl_encoder import RandomMelEncoder

NAN = float("nan")


def resolve_encoder(name: str):
    key = name.lower()
    if key in {"handcrafted", "acoustic", "hand"}:
        return AcousticTrajectory()
    if key.startswith("mert"):
        return MERTBaseline()
    if key.startswith("clap"):
        return CLAPBaseline()
    if key in {"mel", "random", "randommel"}:
        return RandomMelEncoder()
    raise ValueError(f"unknown encoder '{name}'")


def load_segments(path: str | Path, limit: int | None = None) -> list[Segment]:
    segs = read_jsonl(path, cls=Segment)
    return segs[:limit] if limit else segs


def encode_segment(encoder, segment: Segment, max_seconds: float | None = None) -> np.ndarray:
    y, sr = load_audio(segment.path, sr=encoder.sample_rate, mono=True)
    if max_seconds is not None:
        y = y[: int(round(max_seconds * sr))]
    return encoder.encode(y)


def resample_trajectory(z: np.ndarray, length: int = 64) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if z.shape[0] == 0:
        return np.zeros((length, z.shape[1]), dtype=float)
    if z.shape[0] == length:
        return z
    t_old = np.linspace(0.0, 1.0, z.shape[0])
    t_new = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(t_new, t_old, z[:, d]) for d in range(z.shape[1])], axis=-1)


def trajectory_distance(
    a: np.ndarray,
    b: np.ndarray,
    mode: str = "norm",
    norm_len: int = 64,
    scale: np.ndarray | None = None,
) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if scale is not None:
        a = a / scale
        b = b / scale
    if mode == "raw":
        n = min(a.shape[0], b.shape[0])
        if n == 0:
            return NAN
        return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())
    if mode == "norm":
        aa = resample_trajectory(a, norm_len)
        bb = resample_trajectory(b, norm_len)
        return float(np.linalg.norm(aa - bb, axis=1).mean())
    if mode == "dtw":
        import librosa

        D, wp = librosa.sequence.dtw(X=a.T, Y=b.T, metric="euclidean")
        return float(D[-1, -1] / max(1, len(wp)))
    raise ValueError("mode must be raw, norm or dtw")


def distance_separation(preserving: list[float], other: list[float]) -> dict:
    from sklearn.metrics import roc_auc_score

    p = np.asarray([x for x in preserving if np.isfinite(x)], dtype=float)
    o = np.asarray([x for x in other if np.isfinite(x)], dtype=float)
    if p.size == 0 or o.size == 0:
        return {"auc": NAN, "gap": NAN, "mean_preserving": NAN, "mean_other": NAN}
    y = np.r_[np.zeros(p.size), np.ones(o.size)]
    scores = np.r_[p, o]
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = NAN
    return {
        "auc": auc,
        "gap": float(o.mean() - p.mean()),
        "mean_preserving": float(p.mean()),
        "mean_other": float(o.mean()),
    }


def encode_cache(
    encoder,
    segments: list[Segment],
    cache_path: str | Path,
    max_seconds: float | None = None,
) -> dict[str, np.ndarray]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    feats = {s.segment_id: encode_segment(encoder, s, max_seconds=max_seconds) for s in segments}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **feats)
    return feats


def dump_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
