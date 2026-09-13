"""Temporal encoders mapping frame features to a latent trajectory z_{1:T}.

A frozen SSL or acoustic encoder produces (T, F) frame features; these modules
turn them into z_t. `build_encoder` keeps architecture ablations cheap.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        out_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.pos(self.input_proj(x))
        h = self.encoder(h, src_key_padding_mask=mask)
        return self.out_proj(self.norm(h))

    @staticmethod
    def pool(z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return z.mean(dim=1)
        keep = (~mask).unsqueeze(-1).to(z.dtype)
        return (z * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1e-6)


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, out_dim: int = 64, n_layers: int = 2, **_):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.gru(self.input_proj(x))
        return self.out_proj(h)

    pool = TrajectoryEncoder.pool


class MeanPoolEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, out_dim: int = 64, **_):
        super().__init__()
        self.out_proj = nn.Linear(input_dim, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.out_proj(x)

    pool = TrajectoryEncoder.pool


def build_encoder(kind: str, input_dim: int, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind in {"transformer", "temporal_transformer"}:
        return TrajectoryEncoder(input_dim=input_dim, **kwargs)
    if kind in {"gru", "rnn"}:
        return GRUEncoder(input_dim=input_dim, **kwargs)
    if kind in {"mlp", "mean", "pool"}:
        return MeanPoolEncoder(input_dim=input_dim, **kwargs)
    raise ValueError(f"unknown encoder kind '{kind}'")
