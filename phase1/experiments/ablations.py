"""Baseline comparison and ablation driver.

Runs invariance + collapse for a list of frozen encoders/representations and
returns a single table. Learned architecture/objective ablations are driven via
`prediction.run` with different settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import collapse, invariance
from .common import dump_json

DEFAULT_ENCODERS = ("handcrafted", "mel", "mert", "clap")


def compare_encoders(
    encoders: list[str],
    stimuli_dir: str | Path,
    out_path: str | Path | None = None,
    ontology_path: str | Path | None = None,
    limit: int | None = None,
    mode: str = "norm",
) -> dict:
    table = {}
    for name in encoders:
        inv = invariance.run(name, stimuli_dir, ontology_path, out_path=None, mode=mode, limit=limit)
        col = collapse.run(name, stimuli_dir, out_path=None, limit=limit)
        table[name] = {
            "invariance_supported": inv["invariance_supported"],
            "invariance_auc": inv["separation"]["auc"],
            "mean_preserving_distance": inv["separation"]["mean_preserving"],
            "mean_other_distance": inv["separation"]["mean_other"],
            "cross_segment_mean": inv["cross_segment_mean"],
            "collapse_healthy": col["all_frames"]["healthy"],
            "effective_rank": col["all_frames"]["effective_rank"],
            "participation_ratio": col["all_frames"]["participation_ratio"],
        }
    result = {"distance_mode": mode, "encoders": table}
    if out_path is not None:
        dump_json(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare frozen encoders")
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--encoders", nargs="+", default=list(DEFAULT_ENCODERS))
    ap.add_argument("--ontology", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="norm", choices=["raw", "norm", "dtw"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    res = compare_encoders(args.encoders, args.stimuli, args.out, args.ontology, args.limit, args.mode)
    dump_json(args.out or (Path(args.stimuli) / "encoder_comparison.json"), res)
    for name, row in res["encoders"].items():
        print(f"{name:12s} auc={row['invariance_auc']:.3f} healthy={row['collapse_healthy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
