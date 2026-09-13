"""Controlled audio transformations and their *provisional* hypothesis labels.

No transform is treated as a positive or a negative until humans classify it in
`phase1.benchmark.transformation_validation`. The `hypothesis` field here is a
prior only and must be overwritten by the empirical ontology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import db_to_amp, load_audio, save_audio
from .manifest import Segment, Variant, write_jsonl


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(y=y.astype(np.float32), sr=sr, n_steps=float(semitones))


def time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    import librosa

    return librosa.effects.time_stretch(y=y.astype(np.float32), rate=float(rate))


def gain(y: np.ndarray, sr: int, db: float) -> np.ndarray:
    return (y.astype(np.float32) * db_to_amp(db)).astype(np.float32)


def brightness(y: np.ndarray, sr: int, high_gain_db: float, cutoff_alpha: float = 0.2) -> np.ndarray:
    from scipy.signal import lfilter

    y = y.astype(np.float32)
    low = lfilter([cutoff_alpha], [1.0, -(1.0 - cutoff_alpha)], y).astype(np.float32)
    high = y - low
    return (low + db_to_amp(high_gain_db) * high).astype(np.float32)


@dataclass(frozen=True)
class TransformSpec:
    name: str
    family: str
    hypothesis: str
    grid: tuple[dict, ...]
    fn: Callable[..., np.ndarray]


DEFAULT_TRANSFORMS: tuple[TransformSpec, ...] = (
    TransformSpec(
        name="pitch_shift",
        family="pitch",
        hypothesis="preserving_candidate",
        grid=({"semitones": 2.0}, {"semitones": -3.0}),
        fn=pitch_shift,
    ),
    TransformSpec(
        name="gain",
        family="dynamics",
        hypothesis="preserving_candidate",
        grid=({"db": 6.0}, {"db": -6.0}),
        fn=gain,
    ),
    TransformSpec(
        name="brightness",
        family="timbre",
        hypothesis="preserving_candidate",
        grid=({"high_gain_db": 9.0}, {"high_gain_db": -9.0}),
        fn=brightness,
    ),
    TransformSpec(
        name="time_stretch",
        family="tempo",
        hypothesis="ambiguous_candidate",
        grid=({"rate": 0.8}, {"rate": 0.9}, {"rate": 1.1}, {"rate": 1.25}),
        fn=time_stretch,
    ),
)

_REGISTRY: dict[str, TransformSpec] = {spec.name: spec for spec in DEFAULT_TRANSFORMS}


def get_transform(name: str) -> TransformSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown transform '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def transform_tag(params: dict) -> str:
    if not params:
        return "default"
    return "-".join(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}" for k, v in params.items())


def apply_transform(y: np.ndarray, sr: int, name: str, **params) -> np.ndarray:
    spec = get_transform(name)
    out = spec.fn(y, sr, **params)
    if out.ndim > 1:
        out = np.asarray(out).mean(axis=-1)
    return np.asarray(out, dtype=np.float32)


def materialize_variants(
    segments: list[Segment],
    out_dir: str | Path,
    names: tuple[str, ...] | None = None,
    include_original: bool = False,
) -> list[Variant]:
    out_dir = Path(out_dir)
    chosen = tuple(_REGISTRY) if names is None else names
    variants: list[Variant] = []

    for segment in segments:
        y, sr = load_audio(segment.path, sr=segment.sample_rate, mono=True)
        if include_original:
            rel = Path("variants") / segment.segment_id / "original.wav"
            save_audio(out_dir / rel, y, sr)
            variants.append(
                Variant(
                    variant_id=f"{segment.segment_id}__original",
                    segment_id=segment.segment_id,
                    transform="original",
                    params={},
                    path=str(out_dir / rel),
                    sample_rate=sr,
                )
            )
        for name in chosen:
            spec = get_transform(name)
            for params in spec.grid:
                tag = transform_tag(params)
                variant_id = f"{segment.segment_id}__{name}__{tag}"
                rel = Path("variants") / segment.segment_id / f"{name}__{tag}.wav"
                out = apply_transform(y, sr, name, **params)
                save_audio(out_dir / rel, out, sr)
                variants.append(
                    Variant(
                        variant_id=variant_id,
                        segment_id=segment.segment_id,
                        transform=name,
                        params=dict(params),
                        path=str(out_dir / rel),
                        sample_rate=sr,
                    )
                )

    write_jsonl(out_dir / "variants.jsonl", variants)
    return variants
