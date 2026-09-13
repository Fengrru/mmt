"""Collapse diagnostics on a frozen or learned encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..benchmark.metrics import collapse_report
from ..data.manifest import Segment, read_jsonl
from .common import dump_json, encode_cache, resolve_encoder


def run(
    encoder_name: str,
    stimuli_dir: str | Path,
    out_path: str | Path | None = None,
    max_seconds: float | None = 10.0,
    limit: int | None = None,
) -> dict:
    stimuli_dir = Path(stimuli_dir)
    segments = read_jsonl(stimuli_dir / "segments.jsonl", cls=Segment)
    if limit:
        segments = segments[:limit]
    encoder = resolve_encoder(encoder_name)
    feats = encode_cache(encoder, segments, stimuli_dir / "cache" / f"{encoder_name}_seg.npz", max_seconds)

    pooled = np.stack([f.mean(axis=0) for f in feats.values() if f.size])
    frames = np.concatenate([f for f in feats.values() if f.size], axis=0)

    result = {
        "encoder": encoder_name,
        "pooled_segments": collapse_report(pooled),
        "all_frames": collapse_report(frames),
    }
    if out_path is not None:
        dump_json(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collapse diagnostics")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    res = run(args.encoder, args.stimuli, args.out, limit=args.limit)
    dump_json(args.out or (Path(args.stimuli) / f"collapse_{args.encoder}.json"), res)
    print(f"healthy={res['all_frames']['healthy']} eff_rank={res['all_frames']['effective_rank']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
