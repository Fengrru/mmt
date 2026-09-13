"""Handcrafted/acoustic baselines.

These are the interpretable features the learned trajectory must beat. The
four-dimension version mirrors the original `energy + rhythm + pitch + timbre`
proposal, kept as a baseline rather than baked into the representation.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SR = 22050
DEFAULT_HOP = 0.1


class AcousticTrajectory:
    name = "handcrafted"

    def __init__(self, sample_rate: int = DEFAULT_SR, hop_seconds: float = DEFAULT_HOP):
        self.sample_rate = sample_rate
        self.hop_seconds = hop_seconds
        self.feature_names = (
            "rms",
            "spectral_centroid",
            "spectral_bandwidth",
            "zero_crossing_rate",
            "onset_strength",
            "f0",
        )

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        import librosa

        y = np.asarray(waveform, dtype=np.float32)
        hop = max(1, int(round(self.hop_seconds * self.sample_rate)))
        feats = []
        feats.append(librosa.feature.rms(y=y, hop_length=hop)[0])
        feats.append(librosa.feature.spectral_centroid(y=y, sr=self.sample_rate, hop_length=hop)[0])
        feats.append(librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate, hop_length=hop)[0])
        feats.append(librosa.feature.zero_crossing_rate(y, hop_length=hop)[0])
        feats.append(librosa.onset.onset_strength(y=y, sr=self.sample_rate, hop_length=hop))
        try:
            f0, _, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=self.sample_rate,
                hop_length=hop,
            )
            f0 = np.nan_to_num(f0)
        except Exception:
            f0 = np.zeros_like(feats[0])
        feats.append(f0)

        length = min(len(f) for f in feats)
        stacked = np.stack([f[:length] for f in feats], axis=-1)
        stacked = (stacked - stacked.mean(axis=0, keepdims=True)) / (
            stacked.std(axis=0, keepdims=True) + 1e-6
        )
        return stacked.astype(np.float32)

    def encode_gesture_features(self, waveform: np.ndarray) -> dict[str, np.ndarray]:
        x = self.encode(waveform)
        return {
            "energy": x[:, 0],
            "timbre": x[:, 1],
            "rhythm": x[:, 4],
            "pitch": x[:, 5],
        }
