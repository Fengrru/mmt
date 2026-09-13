"""Frozen MERT baseline as a frame encoder."""

from __future__ import annotations

import numpy as np

from ..representation.ssl_encoder import MERTEncoder


class MERTBaseline(MERTEncoder):
    name = "baseline:mert"

    def pooled(self, waveform: np.ndarray) -> np.ndarray:
        return self.encode(waveform).mean(axis=0)
