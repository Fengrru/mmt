"""Frame-level audio encoders.

`RandomMelEncoder` is the offline default (deterministic log-mel frames) used by
tests and by the handcrafted/acoustic baselines. `MERTEncoder` and `CLAPEncoder`
wrap frozen SSL models and are loaded lazily so the repo stays usable offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ..data.audio_io import load_audio


class FrameEncoder(Protocol):
    sample_rate: int
    name: str

    def encode(self, waveform: np.ndarray) -> np.ndarray: ...


class RandomMelEncoder:
    def __init__(self, sample_rate: int = 22050, n_mels: int = 64, hop_seconds: float = 0.1):
        self.name = "mel"
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_seconds = hop_seconds

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        import librosa

        hop = max(1, int(round(self.hop_seconds * self.sample_rate)))
        mel = librosa.feature.melspectrogram(
            y=np.asarray(waveform, dtype=np.float32),
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=hop,
            n_fft=1024,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        return logmel.T.astype(np.float32)


class MERTEncoder:
    def __init__(
        self,
        checkpoint: str = "m-a-p/MERT-v1-95M",
        layer: int = 7,
        hop_seconds: float = 0.1,
        device: str | None = None,
    ):
        self.name = f"mert:{checkpoint}"
        self.checkpoint = checkpoint
        self.layer = layer
        self.hop_seconds = hop_seconds
        self.device = device
        self._model = None
        self._processor = None
        self.sample_rate = 24000

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(self.checkpoint, trust_remote_code=True).eval()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        import torch

        self._load()
        inputs = self._processor(
            np.asarray(waveform, dtype=np.float32),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[self.layer][0]
        fps = 50.0
        step = max(1, int(round(self.hop_seconds * fps)))
        t = hidden.shape[0]
        usable = (t // step) * step
        if usable == 0:
            return hidden.cpu().numpy()
        pooled = hidden[:usable].reshape(usable // step, step, -1).mean(dim=1)
        return pooled.cpu().numpy().astype(np.float32)


class CLAPEncoder:
    def __init__(
        self,
        checkpoint: str = "laion/clap-htsat-unfused",
        window_seconds: float = 1.0,
        hop_seconds: float = 0.5,
        device: str | None = None,
    ):
        self.name = f"clap:{checkpoint}"
        self.checkpoint = checkpoint
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.device = device
        self._model = None
        self._processor = None
        self.sample_rate = 48000

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, ClapModel

        self._processor = AutoProcessor.from_pretrained(self.checkpoint)
        self._model = ClapModel.from_pretrained(self.checkpoint).eval()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        import torch

        from ..data.audio_io import resample

        self._load()
        y = np.asarray(waveform, dtype=np.float32)
        win = int(round(self.window_seconds * 48000))
        hop = max(1, int(round(self.hop_seconds * 48000)))
        if y.shape[0] < win:
            y = np.pad(y, (0, win - y.shape[0]))
        feats = []
        for start in range(0, y.shape[0] - win + 1, hop):
            chunk = y[start : start + win]
            inputs = self._processor(
                audios=chunk, sampling_rate=48000, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self._model.get_audio_features(**inputs)
            feats.append(emb[0].cpu().numpy())
        return np.stack(feats).astype(np.float32) if feats else np.zeros((0, 512), np.float32)


def load_frame_encoder(name: str, **kwargs) -> FrameEncoder:
    key = name.lower()
    if key in {"mel", "random", "randommel"}:
        return RandomMelEncoder(**kwargs)
    if key.startswith("mert"):
        return MERTEncoder(**kwargs)
    if key.startswith("clap"):
        return CLAPEncoder(**kwargs)
    raise ValueError(f"unknown frame encoder '{name}'")


def encode_paths(encoder: FrameEncoder, paths: list[str | Path]) -> list[np.ndarray]:
    out = []
    for p in paths:
        y, sr = load_audio(p, sr=encoder.sample_rate, mono=True)
        out.append(encoder.encode(y))
    return out
