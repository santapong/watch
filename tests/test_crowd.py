"""Tests for P2PNet crowd-counting helpers (model injected/mocked)."""

import numpy as np
import pytest

from src.models.p2pnet_wrapper import P2PNetCounter, filter_points, points_to_detections


def test_filter_points_by_threshold():
    out = filter_points([(1, 1), (2, 2), (3, 3)], [0.9, 0.2, 0.6], 0.5)
    assert out == [(1.0, 1.0), (3.0, 3.0)]


def test_filter_points_empty():
    assert filter_points([], [], 0.5) == []


def test_points_to_detections():
    dets = points_to_detections([(10, 10)], box=8)
    assert len(dets) == 1
    assert dets[0].class_name == "head"
    assert dets[0].bbox == (6.0, 6.0, 14.0, 14.0)


def test_counter_requires_model():
    with pytest.raises(ValueError):
        P2PNetCounter()


def test_counter_with_injected_model(monkeypatch):
    counter = P2PNetCounter(model=object())  # injected -> no torch load
    monkeypatch.setattr(
        counter, "predict_points",
        lambda frame: (np.array([[1, 1], [2, 2]]), np.array([0.9, 0.1])),
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert counter.count(frame) == [(1.0, 1.0)]
    assert counter.count_only(frame) == 1
