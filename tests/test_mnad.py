"""Tests for the MNAD-style reconstruction anomaly backend (learning-phase + maths)."""

import numpy as np
import pytest

from src.analytics.mnad_detector import MNADAnomalyDetector, reconstruction_threshold
from src.models.base import Detection


def _det():
    return Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="person")


def test_reconstruction_threshold_quantile():
    errs = np.arange(100, dtype=float)
    assert reconstruction_threshold(errs, 0.05) == pytest.approx(np.percentile(errs, 95))


def test_reconstruction_threshold_empty():
    assert reconstruction_threshold(np.array([]), 0.05) == float("inf")


class TestInterface:
    def test_initial_state(self):
        d = MNADAnomalyDetector(learning_frames=100)
        assert d.is_learning is True
        assert d.is_fitted is False
        assert d.learning_progress == 0.0

    def test_update_buffers_and_progress(self):
        d = MNADAnomalyDetector(learning_frames=100)
        for _ in range(10):
            d.update([_det()])
        assert d.learning_progress == pytest.approx(0.1)
        assert d.is_learning  # not yet at learning_frames

    def test_check_during_learning_is_quiet(self):
        d = MNADAnomalyDetector(learning_frames=100)
        score, anomalous = d.check([_det()])
        assert score == 0.0
        assert anomalous is False
