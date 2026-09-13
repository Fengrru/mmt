"""Interpretable acoustic motion-descriptor trajectories over time.

Two descriptors are plotted against time: short-time RMS (energy) and spectral
centroid (brightness). This gives a clean, auditable picture of how a realization
changes the *motion* of an excerpt: transposition shifts the brightness curve
while preserving its shape; time stretching preserves the shape while changing
its duration. It is an illustration, not a representation result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def motion_descriptors(y: np.ndarray, sr: int, hop_seconds: float = 0.05):
    import librosa

    hop = max(1, int(round(hop_seconds * sr)))
    rms = librosa.feature.rms(y=np.asarray(y, dtype=np.float32), hop_length=hop)[0]
    rms_db = 20.0 * np.log10(rms + 1e-6)
    centroid = librosa.feature.spectral_centroid(
        y=np.asarray(y, dtype=np.float32), sr=sr, hop_length=hop
    )[0]
    t = np.arange(len(rms)) * hop / sr
    return t, rms_db, centroid


def plot_motion_curves(
    waveforms: dict[str, np.ndarray],
    sr: int,
    out_path: str | Path,
    title: str = "Acoustic motion descriptors",
    hop_seconds: float = 0.05,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, y in waveforms.items():
        t, rms_db, centroid = motion_descriptors(y, sr, hop_seconds=hop_seconds)
        axes[0].plot(t, rms_db, label=name)
        axes[1].plot(t, np.maximum(centroid, 1e-3), label=name)

    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("RMS energy (dB)")
    axes[0].set_title("energy trajectory")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("spectral centroid (Hz)")
    axes[1].set_title("brightness trajectory")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
