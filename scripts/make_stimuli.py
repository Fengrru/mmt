"""Build segment and transform stimuli from a solo-piano audio directory.

Either point `--audio-dir` at a folder, or pass `--metadata` (a CSV such as a
MAESTRO subset) plus `--audio-root` to restrict to listed files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1.data.segments import SegmenterConfig, build_segments
from phase1.data.transforms import DEFAULT_TRANSFORMS, materialize_variants


def _paths_from_metadata(metadata: str | Path, audio_root: str | Path, column: str) -> list[Path]:
    import pandas as pd

    df = pd.read_csv(metadata)
    if column not in df.columns:
        raise SystemExit(f"column '{column}' not in {metadata}; columns={list(df.columns)}")
    root = Path(audio_root)
    paths = [root / str(name) for name in df[column].tolist()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} audio files not found, e.g. {missing[0]}")
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build Phase 1 stimuli")
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--metadata", default=None, help="CSV of audio filenames (e.g. MAESTRO subset)")
    ap.add_argument("--audio-root", default=None, help="root for --metadata relative paths")
    ap.add_argument("--filename-column", default="audio_filename")
    ap.add_argument("--out", required=True)
    ap.add_argument("--durations", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-segments-per-source", type=int, default=200)
    ap.add_argument("--transforms", nargs="+", default=[t.name for t in DEFAULT_TRANSFORMS])
    ap.add_argument("--include-original", action="store_true")
    args = ap.parse_args(argv)

    if not args.audio_dir and not args.metadata:
        raise SystemExit("provide --audio-dir or --metadata/--audio-root")

    paths = None
    if args.metadata:
        root = args.audio_root or Path(args.metadata).parent
        paths = _paths_from_metadata(args.metadata, root, args.filename_column)

    cfg = SegmenterConfig(
        durations=tuple(args.durations),
        max_files=args.max_files,
        max_segments_per_source=args.max_segments_per_source,
    )
    segments = build_segments(args.audio_dir, args.out, config=cfg, paths=paths)
    variants = materialize_variants(
        segments, args.out, names=tuple(args.transforms), include_original=args.include_original
    )
    print(f"segments={len(segments)} variants={len(variants)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
