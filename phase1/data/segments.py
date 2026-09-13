"""Fixed-window segmentation: source of gesture-trajectory candidates.

Phase 1 deliberately does not ask annotators to cut gestures. Candidates are
produced by sliding windows at several durations; later stages filter them with
the human-studied transformation ontology.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import list_audio_files, load_audio, save_audio
from .manifest import Segment, write_jsonl


@dataclass
class SegmenterConfig:
    durations: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    hop_ratio: float = 0.5
    target_sr: int = 22050
    min_rms_db: float = -50.0
    max_files: int | None = None
    max_segments_per_source: int | None = 200


def _rms_db(y: np.ndarray) -> float:
    if y.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20.0 * np.log10(rms + 1e-12)


def iter_windows(n_samples: int, win: int, hop: int):
    if win <= 0 or hop <= 0:
        raise ValueError("win and hop must be positive")
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += hop


def build_segments(
    audio_dir: str | Path,
    out_dir: str | Path,
    config: SegmenterConfig | None = None,
    seed: int = 0,
    paths: list[str | Path] | None = None,
) -> list[Segment]:
    cfg = config or SegmenterConfig()
    out_dir = Path(out_dir)
    files = [Path(p) for p in paths] if paths is not None else list_audio_files(audio_dir)
    if cfg.max_files is not None:
        files = files[: cfg.max_files]
    segments: list[Segment] = []

    for path in files:
        y, sr = load_audio(path, sr=cfg.target_sr, mono=True)
        source_id = path.stem
        stable = int(hashlib.sha256(f"{source_id}:{seed}".encode()).hexdigest(), 16)
        rng = np.random.default_rng(stable % (2**32))

        candidates: list[tuple[float, int, int, int]] = []
        for duration in cfg.durations:
            win = int(round(duration * sr))
            hop = max(1, int(round(win * cfg.hop_ratio)))
            for idx, (a, b) in enumerate(iter_windows(y.shape[0], win, hop)):
                if _rms_db(y[a:b]) < cfg.min_rms_db:
                    continue
                candidates.append((duration, idx, a, b))

        if cfg.max_segments_per_source is not None and len(candidates) > cfg.max_segments_per_source:
            pick = rng.choice(len(candidates), size=cfg.max_segments_per_source, replace=False)
            candidates = [candidates[i] for i in sorted(pick)]

        for duration, idx, a, b in candidates:
            seg_id = f"{source_id}__{duration:g}s__{idx:04d}"
            rel = Path("segments") / source_id / f"{duration:g}s" / f"{idx:04d}.wav"
            seg_path = save_audio(out_dir / rel, y[a:b], sr)
            segments.append(
                Segment(
                    segment_id=seg_id,
                    source_id=source_id,
                    source_path=str(path),
                    start=a / sr,
                    end=b / sr,
                    sample_rate=sr,
                    path=str(seg_path),
                )
            )

    write_jsonl(out_dir / "segments.jsonl", segments)
    return segments


def load_segment_audio(segment: Segment) -> tuple[np.ndarray, int]:
    return load_audio(segment.path, sr=segment.sample_rate, mono=True)
