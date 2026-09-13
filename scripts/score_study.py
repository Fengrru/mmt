"""Score collected human responses (rating -> ontology, choice -> triplet accuracy)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1.benchmark.human_triplets import score_responses
from phase1.benchmark.transformation_validation import run_validation
from phase1.experiments.common import dump_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score human studies")
    ap.add_argument("--kind", choices=["rating", "triplet"], required=True)
    ap.add_argument("--responses", required=True, nargs="+", help="one or more rater JSON files")
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-raters", type=int, default=5)
    ap.add_argument("--min-trials", type=int, default=10)
    args = ap.parse_args(argv)

    if args.kind == "rating":
        result = run_validation(
            args.responses,
            args.trials,
            out_path=args.out,
            min_raters=args.min_raters,
            min_trials=args.min_trials,
        )
        print("ontology:", result["ontology"])
    else:
        result = score_responses(args.responses, args.trials)
        dump_json(args.out, result)
        print(f"accuracy={result['accuracy']:.3f} n={result['n_responses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
