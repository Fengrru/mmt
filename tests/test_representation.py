import torch

from phase1.representation.ssl_encoder import RandomMelEncoder
from phase1.representation.trajectory_encoder import TrajectoryEncoder, build_encoder


def test_mel_encoder_shapes():
    encoder = RandomMelEncoder(n_mels=32, hop_seconds=0.1)
    y = torch.randn(22050).numpy()
    feats = encoder.encode(y)
    assert feats.ndim == 2
    assert feats.shape[1] == 32


def test_trajectory_encoder_forward_and_pool():
    enc = TrajectoryEncoder(input_dim=16, d_model=32, out_dim=8, n_layers=1, n_heads=2)
    x = torch.randn(3, 10, 16)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[0, 6:] = True
    z = enc(x, mask)
    assert z.shape == (3, 10, 8)
    pooled = TrajectoryEncoder.pool(z, mask)
    assert pooled.shape == (3, 8)
    assert torch.isfinite(pooled).all()


def test_build_encoder_kinds():
    for kind in ["transformer", "gru", "mlp"]:
        enc = build_encoder(kind, input_dim=8, out_dim=4, d_model=16, n_layers=1, n_heads=2)
        z = enc(torch.randn(2, 5, 8))
        assert z.shape == (2, 5, 4)


def test_gradients_flow():
    enc = TrajectoryEncoder(input_dim=8, d_model=16, out_dim=4, n_layers=1, n_heads=2)
    z = enc(torch.randn(2, 5, 8))
    loss = z.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
