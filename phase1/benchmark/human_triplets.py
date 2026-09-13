"""Step 5: human triplet benchmark for perceived musical motion similarity.

Triplets are sampled using the *validated* ontology from Step 1, so the human
benchmark and the model benchmark share a fixed notion of positive/negative.
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

from ..data.manifest import Segment, Variant, read_json, write_jsonl
from .runner import write_static_study
from .transformation_validation import ALTERING, PRESERVING, merge_responses, ontology_lookup

TRIPLET_PROMPT = (
    "A is a reference musical gesture. Which of the two options (B or C) "
    "has the same musical motion as A?"
)


def _copy(out_dir: Path, src: str | Path, name: str) -> str:
    dst = out_dir / "audio" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return str(Path("audio") / name).replace("\\", "/")


def _role_options(pos: Variant, neg: Variant, rng: random.Random) -> list[dict]:
    pair = [
        {"id": pos.variant_id, "role": "positive", "audio_path": pos.path},
        {"id": neg.variant_id, "role": "negative", "audio_path": neg.path},
    ]
    rng.shuffle(pair)
    labels = ["B", "C"]
    for label, opt in zip(labels, pair):
        opt["label"] = label
    return pair


def build_triplets(
    segments: list[Segment],
    variants: list[Variant],
    ontology: dict[str, str],
    out_dir: str | Path,
    n_triplets: int = 300,
    seed: int = 0,
    max_per_anchor: int = 5,
) -> list[dict]:
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    seg_by_id = {s.segment_id: s for s in segments}
    pos_by_seg: dict[str, list[Variant]] = defaultdict(list)
    neg_by_seg: dict[str, list[Variant]] = defaultdict(list)
    for v in variants:
        group = ontology_lookup(ontology, v.transform, v.params)
        if group == PRESERVING:
            pos_by_seg[v.segment_id].append(v)
        elif group == ALTERING:
            neg_by_seg[v.segment_id].append(v)

    anchors_with_pos = [sid for sid, vs in pos_by_seg.items() if vs]
    if not anchors_with_pos:
        raise ValueError(
            "no (transform, parameter) validated as preserving; complete Step 1 first"
        )

    all_ids = sorted(seg_by_id)
    trials: list[dict] = []
    attempts = 0
    while len(trials) < n_triplets and attempts < n_triplets * 50:
        attempts += 1
        anchor_id = rng.choice(anchors_with_pos)
        pos = rng.choice(pos_by_seg[anchor_id])
        neg_pool = neg_by_seg.get(anchor_id, [])
        if neg_pool:
            neg = rng.choice(neg_pool)
        else:
            other = [sid for sid in all_ids if sid != anchor_id]
            if not other:
                break
            neg_id = rng.choice(other)
            candidates = [v for v in variants if v.segment_id == neg_id and v.transform != "original"]
            if not candidates:
                continue
            neg = rng.choice(candidates)
        if pos.variant_id == neg.variant_id:
            continue
        per_anchor = sum(1 for t in trials if t["anchor_segment_id"] == anchor_id)
        if per_anchor >= max_per_anchor:
            continue
        anchor_rel = _copy(out_dir, seg_by_id[anchor_id].path, f"{anchor_id}__anchor.wav")
        options = _role_options(pos, neg, rng)
        for opt in options:
            opt["audio"] = _copy(out_dir, opt.pop("audio_path"), f"{opt['id']}.wav")
        trials.append(
            {
                "trial_id": f"{anchor_id}__{pos.variant_id}__vs__{neg.variant_id}",
                "mode": "choice",
                "prompt": TRIPLET_PROMPT,
                "anchor": {"label": "A (reference)", "audio": anchor_rel},
                "options": options,
                "anchor_segment_id": anchor_id,
                "positive_id": pos.variant_id,
                "negative_id": neg.variant_id,
            }
        )

    trial_ids = [t["trial_id"] for t in trials]
    return [t for _, t in sorted(zip(trial_ids, trials))]


def save_triplet_study(
    segments: list[Segment],
    variants: list[Variant],
    ontology: dict[str, str],
    out_dir: str | Path,
    n_triplets: int = 300,
    seed: int = 0,
) -> Path:
    trials = build_triplets(segments, variants, ontology, out_dir, n_triplets=n_triplets, seed=seed)
    write_jsonl(Path(out_dir) / "triplets.jsonl", trials)
    return write_static_study(
        out_dir,
        study_id="human_triplets",
        trials=trials,
        title="Musical motion similarity",
        instructions=TRIPLET_PROMPT,
        scale=7,
    )


def score_responses(responses_path: str | Path, trials_path: str | Path) -> dict:
    payload = read_json(trials_path)
    trials = payload["trials"] if isinstance(payload, dict) else payload
    role = {}
    for t in trials:
        for opt in t["options"]:
            role[(t["trial_id"], opt["id"])] = opt["role"]

    responses = merge_responses(responses_path)
    per_rater: dict[str, list[int]] = defaultdict(list)
    for r in responses:
        key = (r["trial_id"], r["answer"])
        if key in role:
            per_rater[r.get("rater_id", "anonymous")].append(int(role[key] == "positive"))

    per_rater_acc = {k: sum(v) / len(v) for k, v in per_rater.items() if v}
    all_scores = [s for v in per_rater.values() for s in v]
    accuracy = sum(all_scores) / len(all_scores) if all_scores else float("nan")
    return {
        "accuracy": accuracy,
        "n_responses": len(all_scores),
        "per_rater_accuracy": per_rater_acc,
    }


def write_trials(path: str | Path, trials: list[dict]) -> Path:
    return write_jsonl(path, trials)
