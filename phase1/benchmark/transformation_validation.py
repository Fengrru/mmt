"""Step 1: establish the *parameterized* transformation ontology from humans.

The ontology is keyed by (transform, parameter value), not by transform name.
This lets a transform be preserving at one magnitude and altering at another
(e.g. a perceptual boundary alpha* for tempo) instead of forcing a binary label.

Output schema (`ontology.json`):

    {
      "scale": 7, "midpoint": 4.0, "margin": 0.5,
      "entries": [
        {"key": "time_stretch:rate=0.8", "transform": "time_stretch",
         "params": {"rate": 0.8}, "n_ratings": 80, "n_raters": 8,
         "mean": 5.2, "ci_low": 4.8, "ci_high": 5.6,
         "judgment": {"preserving": 0.31, "ambiguous": 0.44, "altering": 0.25},
         "agreement": {"krippendorff_alpha_ordinal": 0.67, "fleiss_kappa": 0.6},
         "status": "ambiguous"}
      ],
      "by_key": {"time_stretch:rate=0.8": "ambiguous"},
      "by_transform": {"time_stretch": "mixed"}
    }
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data.manifest import Segment, Variant, read_json, write_json
from .metrics import bootstrap_ci, fleiss_kappa, krippendorff_alpha
from .runner import write_static_study

RATING_PROMPT = (
    "A and B are two versions of the same musical excerpt. "
    "Rate how similar B's musical motion is to A's, "
    "from 1 (very different) to 7 (same motion)."
)

PRESERVING = "preserving"
ALTERING = "altering"
AMBIGUOUS = "ambiguous"
UNCERTAIN = "uncertain"
MIXED = "mixed"
UNVALIDATED = UNCERTAIN


def ontology_key(transform: str, params: dict | None) -> str:
    if not params:
        return f"{transform}:default"
    body = ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in sorted(params.items()))
    return f"{transform}:{body}"


def ontology_lookup(ontology: dict | None, transform: str, params: dict | None = None) -> str:
    if not ontology:
        return UNCERTAIN
    if "by_key" in ontology:
        key = ontology_key(transform, params)
        if key in ontology["by_key"]:
            return ontology["by_key"][key]
        return ontology.get("by_transform", {}).get(transform, UNCERTAIN)
    return ontology.get(transform, UNCERTAIN)


def ontology_from_hypotheses() -> dict:
    from ..data.transforms import DEFAULT_TRANSFORMS

    by_key: dict[str, str] = {}
    by_transform: dict[str, str] = {}
    for spec in DEFAULT_TRANSFORMS:
        status = spec.hypothesis.replace("_candidate", "")
        by_transform[spec.name] = MIXED
        for params in spec.grid:
            by_key[ontology_key(spec.name, params)] = status
    return {"entries": [], "by_key": by_key, "by_transform": by_transform, "validated": False}


def _copy(out_dir: Path, src: str | Path, name: str) -> str:
    dst = out_dir / "audio" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return str(Path("audio") / name).replace("\\", "/")


def build_rating_trials(
    segments: list[Segment],
    variants: list[Variant],
    out_dir: str | Path,
    max_segments: int | None = None,
    seed: int = 0,
) -> list[dict]:
    out_dir = Path(out_dir)
    seg_by_id = {s.segment_id: s for s in segments}
    chosen = sorted(seg_by_id)
    if max_segments is not None and len(chosen) > max_segments:
        chosen = sorted(random.Random(seed).sample(chosen, max_segments))
    by_segment: dict[str, list[Variant]] = defaultdict(list)
    for v in variants:
        if v.transform != "original":
            by_segment[v.segment_id].append(v)

    trials: list[dict] = []
    for sid in chosen:
        seg = seg_by_id[sid]
        anchor_rel = _copy(out_dir, seg.path, f"{sid}__anchor.wav")
        for v in by_segment.get(sid, []):
            v_rel = _copy(out_dir, v.path, f"{v.variant_id}.wav")
            trials.append(
                {
                    "trial_id": f"{v.variant_id}__rating",
                    "mode": "rating",
                    "prompt": RATING_PROMPT,
                    "anchor": {"label": "A (reference)", "audio": anchor_rel},
                    "options": [{"id": v.variant_id, "label": "B", "audio": v_rel}],
                    "segment_id": sid,
                    "transform": v.transform,
                    "params": v.params,
                    "key": ontology_key(v.transform, v.params),
                }
            )
    return trials


def save_rating_study(
    segments: list[Segment],
    variants: list[Variant],
    out_dir: str | Path,
    max_segments: int | None = None,
    seed: int = 0,
) -> Path:
    trials = build_rating_trials(segments, variants, out_dir, max_segments=max_segments, seed=seed)
    return write_static_study(
        out_dir,
        study_id="transformation_validation",
        trials=trials,
        title="Transformation validation",
        instructions=RATING_PROMPT,
        scale=7,
    )


def load_responses(path: str | Path) -> list[dict]:
    payload = read_json(path)
    responses = payload["responses"] if isinstance(payload, dict) else payload
    return [r for r in responses if r.get("trial_id") is not None and r.get("answer") is not None]


def merge_responses(paths, out_path: str | Path | None = None) -> list[dict]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged: list[dict] = []
    for p in paths:
        merged.extend(load_responses(p))
    if out_path is not None:
        write_json(out_path, {"responses": merged})
    return merged


def _response_category(answer: float, mid: float, margin: float) -> str:
    if answer >= mid + margin:
        return PRESERVING
    if answer <= mid - margin:
        return ALTERING
    return AMBIGUOUS


def aggregate_ratings(
    responses: list[dict],
    trials: list[dict],
    mid: float = 4.0,
    margin: float = 0.5,
) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if t.get("transform"):
            groups[t["key"]].append(t)

    answer_by_trial: dict[str, list] = defaultdict(list)
    for r in responses:
        answer_by_trial[r["trial_id"]].append((r.get("rater_id", "anonymous"), r["answer"]))

    out: dict[str, dict] = {}
    for key, group in groups.items():
        trial_ids = [t["trial_id"] for t in group]
        raters = sorted({r for tid in trial_ids for r, _ in answer_by_trial.get(tid, [])})
        r_index = {r: i for i, r in enumerate(raters)}
        t_index = {t: i for i, t in enumerate(trial_ids)}
        mat = np.full((len(trial_ids), len(raters)), None, dtype=object)
        responses_here = []
        for tid in trial_ids:
            for rater, answer in answer_by_trial.get(tid, []):
                mat[t_index[tid], r_index[rater]] = answer
                responses_here.append(float(answer))
        values = np.asarray(responses_here, dtype=float)
        mean, lo, hi = bootstrap_ci(values) if values.size else (float("nan"),) * 3

        categories = [PRESERVING, AMBIGUOUS, ALTERING]
        judgment = {c: 0.0 for c in categories}
        for v in values:
            judgment[_response_category(float(v), mid, margin)] += 1.0
        if values.size:
            judgment = {c: judgment[c] / values.size for c in categories}

        alpha = float("nan")
        kappa = float("nan")
        if mat.shape[0] and mat.shape[1] >= 2:
            try:
                alpha = krippendorff_alpha(mat, level="ordinal")
            except Exception:
                alpha = float("nan")
            if all(v is not None for row in mat for v in row):
                try:
                    kappa = fleiss_kappa(mat.astype(float))
                except Exception:
                    kappa = float("nan")

        t = group[0]
        out[key] = {
            "key": key,
            "transform": t["transform"],
            "params": t.get("params", {}),
            "n_ratings": int(values.size),
            "n_trials": int(mat.shape[0]),
            "n_raters": int(mat.shape[1]),
            "mean": float(mean),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "judgment": judgment,
            "agreement": {
                "krippendorff_alpha_ordinal": alpha,
                "fleiss_kappa": kappa,
            },
        }
    return out


def _status_for(
    stats: dict,
    mid: float,
    margin: float,
    min_raters: int,
    min_trials: int,
    min_alpha: float,
) -> str:
    if stats["n_raters"] < min_raters or stats["n_trials"] < min_trials:
        return UNCERTAIN
    alpha = stats["agreement"]["krippendorff_alpha_ordinal"]
    if not np.isnan(alpha) and alpha < min_alpha:
        return UNCERTAIN
    if stats["ci_low"] >= mid + margin:
        return PRESERVING
    if stats["ci_high"] <= mid - margin:
        return ALTERING
    return AMBIGUOUS


def classify_ontology(
    aggregated: dict[str, dict],
    mid: float = 4.0,
    margin: float = 0.5,
    min_raters: int = 5,
    min_trials: int = 10,
    min_alpha: float = 0.2,
    scale: int = 7,
) -> dict:
    entries = []
    for stats in aggregated.values():
        entry = dict(stats)
        entry["status"] = _status_for(stats, mid, margin, min_raters, min_trials, min_alpha)
        entries.append(entry)
    entries.sort(key=lambda e: (e["transform"], str(e["params"])))

    by_key = {e["key"]: e["status"] for e in entries}
    grouped: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        grouped[e["transform"]].append(e["status"])
    by_transform = {}
    for name, statuses in grouped.items():
        by_transform[name] = statuses[0] if len(set(statuses)) == 1 else MIXED

    return {
        "scale": scale,
        "midpoint": mid,
        "margin": margin,
        "validated": True,
        "entries": entries,
        "by_key": by_key,
        "by_transform": by_transform,
    }


def run_validation(
    responses_path: str | Path,
    trials_path: str | Path,
    out_path: str | Path | None = None,
    mid: float = 4.0,
    margin: float = 0.5,
    scale: int = 7,
    **classify_kwargs,
) -> dict:
    responses = merge_responses(responses_path)
    payload = read_json(trials_path)
    trials = payload["trials"] if isinstance(payload, dict) else payload
    aggregated = aggregate_ratings(responses, trials, mid=mid, margin=margin)
    ontology = classify_ontology(aggregated, mid=mid, margin=margin, scale=scale, **classify_kwargs)
    result = {"aggregated": aggregated, "ontology": ontology}
    if out_path is not None:
        write_json(out_path, result)
    return result
