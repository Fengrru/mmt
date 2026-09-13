"""Select a small, composer-stratified solo-piano pilot subset from MAESTRO metadata.

Produces `datasets/maestro_subset/metadata.csv` (and optionally copies audio).
Keep the pilot small: the goal is to shake out the protocol, not to power a
final statistical claim.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def stratified_sample(df, n: int, seed: int, stratify: str | None):
    rng = random.Random(seed)
    if stratify is None or stratify not in df.columns:
        idx = list(df.index)
        rng.shuffle(idx)
        return df.loc[idx[:n]]
    pools = [list(g.index) for _, g in df.groupby(stratify)]
    for p in pools:
        rng.shuffle(p)
    chosen: list = []
    while len(chosen) < n and any(pools):
        for p in pools:
            if p and len(chosen) < n:
                chosen.append(p.pop())
    return df.loc[chosen]


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Prepare a MAESTRO pilot subset")
    ap.add_argument("--metadata", required=True, help="full MAESTRO metadata.csv")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", default="datasets/maestro_subset")
    ap.add_argument("--n", type=int, default=40, help="20-50 recommended for the pilot")
    ap.add_argument("--split", default=None, choices=[None, "train", "validation", "test"])
    ap.add_argument("--stratify", default="composer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy audio into the subset folder")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.metadata)
    if args.split is not None and "split" in df.columns:
        df = df[df["split"] == args.split]
    if "audio_filename" not in df.columns:
        raise SystemExit(f"'audio_filename' not in metadata columns: {list(df.columns)}")

    stratify = args.stratify
    if stratify not in df.columns:
        stratify = next((c for c in ("canonical_composer", "composer") if c in df.columns), None)
    subset = stratified_sample(df, args.n, args.seed, stratify).reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root)

    if args.copy:
        dest = out / "audio"
        dest.mkdir(parents=True, exist_ok=True)
        names = []
        for name in subset["audio_filename"]:
            src = audio_root / str(name)
            if not src.exists():
                raise SystemExit(f"missing audio: {src}")
            target = dest / Path(str(name)).name
            shutil.copyfile(src, target)
            names.append(target.name)
        subset["audio_filename"] = names
        (out / "audio_root.txt").write_text(str(dest), encoding="utf-8")
    else:
        (out / "audio_root.txt").write_text(str(audio_root), encoding="utf-8")

    subset.to_csv(out / "metadata.csv", index=False)
    print(f"wrote {len(subset)} rows to {out / 'metadata.csv'}")
    print(f"stratified by: {stratify}")
    if stratify in subset.columns:
        print(subset[stratify].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
