"""Build the static human studies: transformation rating and/or triplets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1.benchmark.human_triplets import save_triplet_study
from phase1.benchmark.transformation_validation import save_rating_study
from phase1.data.manifest import Variant, read_json, read_jsonl
from phase1.data.manifest import Segment


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build human studies")
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["rating", "triplet", "both"], default="rating")
    ap.add_argument("--max-segments", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ontology", default=None, help="ontology.json produced by score_study")
    ap.add_argument("--n-triplets", type=int, default=300)
    args = ap.parse_args(argv)

    stimuli = Path(args.stimuli)
    segments = read_jsonl(stimuli / "segments.jsonl", cls=Segment)
    variants = read_jsonl(stimuli / "variants.jsonl", cls=Variant)

    if args.mode in {"rating", "both"}:
        path = save_rating_study(segments, variants, args.out, max_segments=args.max_segments, seed=args.seed)
        print(f"rating study: {path}")

    if args.mode in {"triplet", "both"}:
        ontology_path = Path(args.ontology) if args.ontology else (stimuli / "ontology.json")
        if not ontology_path.exists():
            raise SystemExit(f"ontology not found at {ontology_path}; run score_study first")
        ontology = read_json(ontology_path).get("ontology", {})
        path = save_triplet_study(segments, variants, ontology, args.out, n_triplets=args.n_triplets)
        print(f"triplet study: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
