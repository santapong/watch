"""Tests for heatmap generation."""

import numpy as np
import pytest

from src.models.base import Detection
from src.analytics.heatmap import HeatmapGenerator


def _det(x=100, y=100, w=50, h=80, class_name="person", class_id=0):
    return Detection(
        bbox=(float(x), float(y), float(x + w), float(y + h)),
        confidence=0.9,
        class_id=class_id,
        class_name=class_name,
    )


class TestHeatmapGenerator:
    def test_initial_state(self):
        hm = HeatmapGenerator(frame_shape=(480, 640))
        assert hm.frame_count == 0
        raw = hm.get_raw()
        assert raw.shape == (480, 640)
        assert raw.sum() == 0

    def test_update_increases_accumulator(self):
        hm = HeatmapGenerator(frame_shape=(480, 640), radius=20)
        hm.update([_det(x=100, y=100)])
        assert hm.frame_count == 1
        assert hm.get_raw().sum() > 0

    def test_decay(self):
        hm = HeatmapGenerator(frame_shape=(480, 640), decay=0.5, radius=20)
        hm.update([_det()])
        val1 = hm.get_raw().sum()
        hm.update([])  # Empty frame applies decay
        val2 = hm.get_raw().sum()
        assert val2 < val1

    def test_class_filter(self):
        hm = HeatmapGenerator(frame_shape=(480, 640), class_filter=["car"], radius=20)
        hm.update([_det(class_name="person")])
        assert hm.get_raw().sum() == 0  # Person filtered out
        hm.update([_det(class_name="car", class_id=2)])
        assert hm.get_raw().sum() > 0  # Car allowed

    def test_render(self):
        hm = HeatmapGenerator(frame_shape=(480, 640), radius=20)
        hm.update([_det()])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = hm.render(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_render_empty(self):
        hm = HeatmapGenerator(frame_shape=(480, 640))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = hm.render(frame)
        assert result.shape == frame.shape

    def test_reset(self):
        hm = HeatmapGenerator(frame_shape=(480, 640), radius=20)
        hm.update([_det()])
        assert hm.frame_count == 1
        hm.reset()
        assert hm.frame_count == 0
        assert hm.get_raw().sum() == 0

    def test_out_of_bounds_detection(self):
        hm = HeatmapGenerator(frame_shape=(100, 100), radius=10)
        # Detection center outside frame bounds
        hm.update([_det(x=200, y=200)])
        assert hm.frame_count == 1
        # Should not crash, accumulator should still be zero
        assert hm.get_raw().sum() == 0

    def test_save_snapshot(self, tmp_path):
        hm = HeatmapGenerator(frame_shape=(100, 100), radius=20)
        hm.update([_det(x=30, y=30, w=20, h=20)])
        path = str(tmp_path / "test_heatmap.png")
        hm.save_snapshot(path)
        import os
        assert os.path.exists(path)
