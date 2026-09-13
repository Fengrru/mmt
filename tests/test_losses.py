import torch

from phase1.representation.losses import (
    multi_step_prediction_loss,
    triplet_loss,
    variance_loss,
    vicreg_loss,
)


def test_variance_loss_penalizes_collapse():
    collapsed = torch.ones(256, 64)
    diverse = torch.randn(256, 64)
    assert float(variance_loss(collapsed)) > 0.5
    assert float(variance_loss(diverse)) < float(variance_loss(collapsed))


def test_vicreg_prefers_diverse_batch():
    torch.manual_seed(0)
    collapsed = torch.ones(128, 16)
    diverse = torch.randn(128, 16)
    lc, _ = vicreg_loss(collapsed)
    ld, _ = vicreg_loss(diverse)
    assert float(lc) > float(ld)


def test_multi_step_prediction_loss_zero_when_equal():
    a = torch.randn(4, 3, 8)
    assert float(multi_step_prediction_loss(a, a)) < 1e-6
    b = a + 1.0
    assert float(multi_step_prediction_loss(a, b)) > 0.0


def test_triplet_loss_zero_when_ordering_correct():
    anchor = torch.zeros(8, 4)
    positive = torch.full((8, 4), 0.05)
    negative = torch.ones(8, 4)
    assert float(triplet_loss(anchor, positive, negative)) == 0.0
    assert float(triplet_loss(anchor, negative, positive)) > 0.0
