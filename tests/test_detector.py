"""Tests for detection base classes and utilities."""

import numpy as np
import pytest

from src.models.base import Detection
from src.utils.fps import FPSCounter


class TestDetection:
    def test_create_detection(self):
        det = Detection(
            bbox=(10.0, 20.0, 100.0, 200.0),
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        assert det.bbox == (10.0, 20.0, 100.0, 200.0)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.mask is None
        assert det.track_id is None

    def test_detection_properties(self):
        det = Detection(
            bbox=(10.0, 20.0, 110.0, 220.0),
            confidence=0.8,
            class_id=1,
            class_name="car",
        )
        assert det.width == 100.0
        assert det.height == 200.0
        assert det.center == (60.0, 120.0)

    def test_detection_with_tracking(self):
        det = Detection(
            bbox=(0.0, 0.0, 50.0, 50.0),
            confidence=0.5,
            class_id=2,
            class_name="bicycle",
            track_id=42,
        )
        assert det.track_id == 42

    def test_detection_with_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        det = Detection(
            bbox=(0.0, 0.0, 100.0, 100.0),
            confidence=0.9,
            class_id=0,
            class_name="person",
            mask=mask,
        )
        assert det.mask is not None
        assert det.mask.shape == (100, 100)

    def test_detection_zero_size(self):
        det = Detection(
            bbox=(50.0, 50.0, 50.0, 50.0),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        assert det.width == 0.0
        assert det.height == 0.0
        assert det.center == (50.0, 50.0)


class TestFPSCounter:
    def test_initial_fps_is_zero(self):
        counter = FPSCounter()
        assert counter.fps == 0.0

    def test_fps_after_single_tick(self):
        counter = FPSCounter()
        counter.tick()
        assert counter.fps == 0.0  # Need at least 2 ticks

    def test_fps_after_multiple_ticks(self):
        counter = FPSCounter(window_size=10)
        for _ in range(5):
            counter.tick()
        # FPS should be > 0 since ticks happen very fast
        assert counter.fps > 0

    def test_fps_window_size(self):
        counter = FPSCounter(window_size=3)
        for _ in range(10):
            counter.tick()
        # Should still work without error after exceeding window
        assert counter.fps > 0
