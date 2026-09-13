"""Audio loading/saving with a hard soundfile requirement and a librosa resampler."""

from __future__ import annotations

from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif")


def list_audio_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in AUDIO_EXTENSIONS]
    return files


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        return y.mean(axis=1).astype(np.float32)
    return y.astype(np.float32, copy=False)


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if mono:
        y = to_mono(y)
    if sr is not None and sr != file_sr:
        y = resample(y, file_sr, sr)
        file_sr = sr
    return y.astype(np.float32, copy=False), int(file_sr)


def save_audio(path: str | Path, y: np.ndarray, sr: int) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(y, dtype=np.float32), int(sr))
    return path


def peak_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < eps:
        return y.astype(np.float32, copy=False)
    return (y / peak).astype(np.float32)


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))
