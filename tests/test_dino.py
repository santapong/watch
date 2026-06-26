"""Tests for DINOv3 embedding helpers (pure numpy)."""

import numpy as np
import pytest

from src.models.dinov3_backbone import cosine_similarity, l2_normalize


def test_l2_normalize_unit_length():
    assert np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))) == pytest.approx(1.0)


def test_l2_normalize_zero_is_finite():
    assert np.all(np.isfinite(l2_normalize(np.zeros(4))))


def test_cosine_similarity_extremes():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
