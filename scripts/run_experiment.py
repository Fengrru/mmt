"""Run a Phase 1 experiment: invariance, collapse, prediction or comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1.experiments import compare_encoders, run_collapse, run_invariance, run_prediction
from phase1.experiments.common import dump_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1 experiment runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_inv = sub.add_parser("invariance")
    p_inv.add_argument("--encoder", required=True)
    p_inv.add_argument("--stimuli", required=True)
    p_inv.add_argument("--ontology", default=None)
    p_inv.add_argument("--out", default=None)
    p_inv.add_argument("--mode", default="norm", choices=["raw", "norm", "dtw"])
    p_inv.add_argument("--limit", type=int, default=None)

    p_col = sub.add_parser("collapse")
    p_col.add_argument("--encoder", required=True)
    p_col.add_argument("--stimuli", required=True)
    p_col.add_argument("--out", default=None)
    p_col.add_argument("--limit", type=int, default=None)

    p_pred = sub.add_parser("prediction")
    p_pred.add_argument("--encoder", required=True)
    p_pred.add_argument("--stimuli", required=True)
    p_pred.add_argument("--out", default=None)
    p_pred.add_argument("--k", type=int, default=4)
    p_pred.add_argument("--epochs", type=int, default=30)
    p_pred.add_argument("--limit", type=int, default=None)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--stimuli", required=True)
    p_cmp.add_argument("--encoders", nargs="+", default=["handcrafted", "mel", "mert", "clap"])
    p_cmp.add_argument("--ontology", default=None)
    p_cmp.add_argument("--out", default=None)
    p_cmp.add_argument("--mode", default="norm")
    p_cmp.add_argument("--limit", type=int, default=None)

    args = ap.parse_args(argv)
    stimuli = Path(args.stimuli)

    if args.cmd == "invariance":
        res = run_invariance(args.encoder, args.stimuli, args.ontology, args.out, mode=args.mode, limit=args.limit)
        dump_json(args.out or stimuli / f"invariance_{args.encoder}.json", res)
        print(f"supported={res['invariance_supported']} auc={res['separation']['auc']:.3f}")
    elif args.cmd == "collapse":
        res = run_collapse(args.encoder, args.stimuli, args.out, limit=args.limit)
        dump_json(args.out or stimuli / f"collapse_{args.encoder}.json", res)
        print(f"healthy={res['all_frames']['healthy']}")
    elif args.cmd == "prediction":
        res = run_prediction(args.encoder, args.stimuli, args.out, k=args.k, epochs=args.epochs, limit=args.limit)
        dump_json(args.out or stimuli / f"prediction_{args.encoder}.json", res)
        print(f"learned={res['learned_mse']:.4f} baseline={res['best_baseline_mse']:.4f}")
    else:
        res = compare_encoders(args.encoders, args.stimuli, args.out, args.ontology, args.limit, args.mode)
        dump_json(args.out or stimuli / "encoder_comparison.json", res)
        for name, row in res["encoders"].items():
            print(f"{name:12s} auc={row['invariance_auc']:.3f} healthy={row['collapse_healthy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
