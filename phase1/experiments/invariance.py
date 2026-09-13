"""Invariance / specificity experiment at (transform, parameter) granularity.

Answers: for (transform, parameter) settings humans validated as *preserving*,
is d(z_A, z_T(A)) smaller than the distance between genuinely different
gestures? And do *altering* settings grow? Ambiguous/uncertain settings are
reported but excluded from the success decision.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..benchmark.transformation_validation import (
    ALTERING,
    PRESERVING,
    ontology_from_hypotheses,
    ontology_key,
    ontology_lookup,
)
from ..data.manifest import Segment, Variant, read_json, read_jsonl
from .common import distance_separation, dump_json, encode_cache, resolve_encoder, trajectory_distance


def _scale_from(feats: dict[str, np.ndarray]) -> np.ndarray:
    frames = np.concatenate([f for f in feats.values() if f.size], axis=0)
    return frames.std(axis=0) + 1e-6


def _cross_distances(feats, segments, mode, norm_len, scale, n_pairs=300, seed=0):
    rng = random.Random(seed)
    ids = [s.segment_id for s in segments if s.segment_id in feats]
    if len(ids) < 2:
        return []
    out = []
    for _ in range(n_pairs):
        a, b = rng.sample(ids, 2)
        out.append(trajectory_distance(feats[a], feats[b], mode=mode, norm_len=norm_len, scale=scale))
    return out


def run(
    encoder_name: str,
    stimuli_dir: str | Path,
    ontology_path: str | Path | None = None,
    out_path: str | Path | None = None,
    mode: str = "norm",
    norm_len: int = 64,
    max_seconds: float | None = 10.0,
    limit: int | None = None,
    normalized: bool = True,
    n_cross: int = 300,
) -> dict:
    stimuli_dir = Path(stimuli_dir)
    segments = read_jsonl(stimuli_dir / "segments.jsonl", cls=Segment)
    variants = read_jsonl(stimuli_dir / "variants.jsonl", cls=Variant)
    if limit:
        keep = {s.segment_id for s in segments[:limit]}
        segments = [s for s in segments if s.segment_id in keep]
        variants = [v for v in variants if v.segment_id in keep]

    if ontology_path is None:
        ontology_path = stimuli_dir / "ontology.json"
    ontology_path = Path(ontology_path)
    if ontology_path.exists():
        ontology = read_json(ontology_path).get("ontology", {})
        validated = bool(ontology.get("validated", True))
    else:
        ontology = ontology_from_hypotheses()
        validated = False

    encoder = resolve_encoder(encoder_name)
    seg_feats = encode_cache(encoder, segments, stimuli_dir / "cache" / f"{encoder_name}_seg.npz", max_seconds)
    var_segments = [
        Segment(v.variant_id, v.segment_id, "", 0.0, 0.0, v.sample_rate, v.path) for v in variants
    ]
    var_feats = encode_cache(encoder, var_segments, stimuli_dir / "cache" / f"{encoder_name}_var.npz", max_seconds)

    scale = _scale_from(seg_feats) if normalized else None
    by_key: dict[str, list[float]] = defaultdict(list)
    key_meta: dict[str, tuple[str, dict, str]] = {}
    for v in variants:
        a = seg_feats.get(v.segment_id)
        b = var_feats.get(v.variant_id)
        if a is None or b is None:
            continue
        k = ontology_key(v.transform, v.params)
        by_key[k].append(trajectory_distance(a, b, mode=mode, norm_len=norm_len, scale=scale))
        key_meta[k] = (v.transform, v.params, ontology_lookup(ontology, v.transform, v.params))

    cross = _cross_distances(seg_feats, segments, mode, norm_len, scale, n_pairs=n_cross)
    cross_mean = float(np.mean(cross)) if cross else float("nan")

    per_key = {
        k: {
            "transform": key_meta[k][0],
            "params": key_meta[k][1],
            "group": key_meta[k][2],
            "n": len(d),
            "mean": float(np.mean(d)) if d else float("nan"),
            "median": float(np.median(d)) if d else float("nan"),
            "below_cross": bool(np.mean(d) < cross_mean) if cross and d else None,
        }
        for k, d in by_key.items()
    }

    preserving_d = [x for k, d in by_key.items() if key_meta[k][2] == PRESERVING for x in d]
    altering_d = [x for k, d in by_key.items() if key_meta[k][2] == ALTERING for x in d]
    separation = distance_separation(preserving_d, altering_d + cross)
    per_preserving_ok = {
        k: per_key[k]["below_cross"] for k in per_key if per_key[k]["group"] == PRESERVING
    }

    result = {
        "encoder": encoder_name,
        "distance_mode": mode,
        "validated_ontology": validated,
        "ontology_by_transform": ontology.get("by_transform", ontology),
        "per_key": per_key,
        "cross_segment_mean": cross_mean,
        "separation": separation,
        "per_preserving_below_cross": per_preserving_ok,
        "invariance_supported": bool(
            validated
            and separation["auc"] == separation["auc"]
            and separation["auc"] > 0.5
            and per_preserving_ok
            and all(per_preserving_ok.values())
        ),
    }
    if out_path is not None:
        dump_json(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Invariance / specificity experiment")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--ontology", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="norm", choices=["raw", "norm", "dtw"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    res = run(args.encoder, args.stimuli, args.ontology, args.out, mode=args.mode, limit=args.limit)
    dump_json(args.out or (Path(args.stimuli) / f"invariance_{args.encoder}.json"), res)
    print(f"invariance_supported={res['invariance_supported']} auc={res['separation']['auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
