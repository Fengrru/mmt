import numpy as np

from phase1.benchmark.metrics import (
    collapse_report,
    fleiss_kappa,
    krippendorff_alpha,
    triplet_accuracy,
)


def test_fleiss_perfect_agreement():
    counts = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4], [0, 4, 0], [4, 0, 0]], dtype=float)
    assert abs(fleiss_kappa(counts) - 1.0) < 1e-9


def test_fleiss_disagreement_below_one():
    counts = np.array([[2, 2, 0], [2, 0, 2], [0, 2, 2], [1, 1, 2], [2, 1, 1]], dtype=float)
    k = fleiss_kappa(counts)
    assert k < 1.0


def test_krippendorff_perfect_and_range():
    perfect = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert abs(krippendorff_alpha(perfect, level="ordinal") - 1.0) < 1e-9
    disagree = np.array([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
    a = krippendorff_alpha(disagree, level="ordinal")
    assert -1.0 <= a < 1.0


def test_triplet_accuracy():
    assert triplet_accuracy([0.1, 0.2], [0.9, 0.8]) == 1.0
    assert triplet_accuracy([0.9, 0.2], [0.1, 0.8]) == 0.5


def test_collapse_report_constant_vs_random():
    constant = np.ones((64, 32))
    report = collapse_report(constant)
    assert report["healthy"] is False

    rng = np.random.default_rng(0)
    random = rng.normal(size=(512, 32))
    report = collapse_report(random)
    assert report["healthy"] is True
    assert report["effective_rank"] > 10
