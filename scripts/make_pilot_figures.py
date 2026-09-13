"""Generate RDM / MDS / shared-projection trajectory figures from the real pilot.

These figures are *illustrative*: they use the provisional transformation
hypotheses (not a validated ontology) and a frozen baseline encoder, so they must
be labeled as engineering visualization, not evidence.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1.data.audio_io import load_audio
from phase1.data.manifest import Segment, Variant, read_jsonl
from phase1.experiments.common import resolve_encoder
from phase1.visualization.motion import plot_motion_curves
from phase1.visualization.similarity import plot_mds, plot_rdm
from phase1.visualization.trajectories import plot_trajectories
from scipy.spatial.distance import pdist

SETTINGS = [
    ("__anchor__", {}, "anchor"),
    ("pitch_shift", {"semitones": 2.0}, "+2st"),
    ("gain", {"db": 6.0}, "+6dB"),
    ("brightness", {"high_gain_db": 9.0}, "bright+"),
    ("time_stretch", {"rate": 0.9}, "tempo0.9"),
    ("time_stretch", {"rate": 0.8}, "tempo0.8"),
]

# A reduced set for the trajectory figure: one weak preserving transform and one
# strong altering transform, so the shared projection stays readable.
TRAJ_SETTINGS = [
    ("__anchor__", {}, "anchor"),
    ("pitch_shift", {"semitones": 2.0}, "+2st (preserving prior)"),
    ("time_stretch", {"rate": 0.8}, "tempo x0.8 (altering prior)"),
]


def _match(variants_by_seg, segment_id, name, params):
    for v in variants_by_seg.get(segment_id, []):
        if v.transform == name and all(
            abs(float(v.params.get(k, 1e9)) - float(val)) < 1e-6 for k, val in params.items()
        ):
            return v
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pilot figures (provisional ontology)")
    ap.add_argument("--stimuli", default="artifacts/maestro_pilot/stimuli")
    ap.add_argument("--out", default="artifacts/maestro_pilot/figures")
    ap.add_argument("--encoder", default="mel")
    ap.add_argument("--num-anchors", type=int, default=4)
    ap.add_argument("--min-duration", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    stimuli = Path(args.stimuli)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    segments = [s for s in read_jsonl(stimuli / "segments.jsonl", cls=Segment) if s.duration >= args.min_duration - 1e-6]
    variants = read_jsonl(stimuli / "variants.jsonl", cls=Variant)
    by_seg: dict[str, list[Variant]] = defaultdict(list)
    for v in variants:
        by_seg[v.segment_id].append(v)

    rng = random.Random(args.seed)
    anchors = rng.sample(segments, min(args.num_anchors, len(segments)))
    encoder = resolve_encoder(args.encoder)

    def encode(seg: Segment) -> np.ndarray:
        y, sr = load_audio(seg.path, sr=encoder.sample_rate, mono=True)
        return encoder.encode(y)

    items: list[tuple[str, np.ndarray]] = []
    for i, seg in enumerate(anchors):
        aid = f"A{i + 1}"
        a_traj = encode(seg)
        items.append((aid, a_traj))
        for name, params, short in SETTINGS[1:]:
            v = _match(by_seg, seg.segment_id, name, params)
            if v is None:
                continue
            items.append((f"{aid}·{short}", encode(v)))

    labels = [lab for lab, _ in items]
    pooled = np.stack([t.mean(axis=0) for _, t in items])
    print(f"items={len(items)} emb_dim={pooled.shape[1]} encoder={args.encoder}")

    rdm = pdist(pooled, metric="euclidean")
    plot_rdm(rdm, out / "pilot_rdm.png", labels=labels, title="Pilot RDM (provisional, illustrative)")
    plot_mds(pooled, out / "pilot_mds.png", labels=labels, title="Pilot MDS (provisional, illustrative)")

    first = anchors[0]
    first_traj = encode(first)
    trajs = {"A1 (anchor)": first_traj}
    for name, params, short in TRAJ_SETTINGS[1:]:
        v = _match(by_seg, first.segment_id, name, params)
        if v is not None:
            trajs[f"A1 · {short}"] = encode(v)
    plot_trajectories(
        trajs,
        out / "pilot_trajectories.png",
        title="Pilot trajectories, shared 2D projection (provisional)",
        smooth=3,
        resample_length=8,
    )

    waveforms = {"A1 (anchor)": load_audio(first.path, sr=encoder.sample_rate, mono=True)[0]}
    for name, params, short in TRAJ_SETTINGS[1:]:
        v = _match(by_seg, first.segment_id, name, params)
        if v is not None:
            waveforms[f"A1 · {short}"] = load_audio(v.path, sr=encoder.sample_rate, mono=True)[0]
    plot_motion_curves(
        waveforms,
        encoder.sample_rate,
        out / "pilot_motion.png",
        title="Acoustic motion descriptors (illustrative, provisional)",
    )

    for f in ["pilot_rdm.png", "pilot_mds.png", "pilot_trajectories.png", "pilot_motion.png"]:
        p = out / f
        print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
