"""Frozen CLAP baseline as a windowed frame encoder."""

from __future__ import annotations

import numpy as np

from ..representation.ssl_encoder import CLAPEncoder


class CLAPBaseline(CLAPEncoder):
    name = "baseline:clap"

    def pooled(self, waveform: np.ndarray) -> np.ndarray:
        frames = self.encode(waveform)
        if frames.shape[0] == 0:
            return np.zeros(frames.shape[1], dtype=np.float32)
        return frames.mean(axis=0)
