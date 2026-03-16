"""Tests for privacy filter module."""

import numpy as np
import pytest

from src.models.base import Detection
from src.privacy import PrivacyFilter


def _det(x=10, y=20, w=50, h=80, class_name="person", class_id=0):
    return Detection(
        bbox=(float(x), float(y), float(x + w), float(y + h)),
        confidence=0.9,
        class_id=class_id,
        class_name=class_name,
    )


class TestPrivacyFilter:
    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            PrivacyFilter(mode="invalid")

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="Invalid target"):
            PrivacyFilter(target="invalid")

    def test_blur_mode(self):
        pf = PrivacyFilter(mode="blur", target="person")
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        dets = [_det(x=10, y=10, w=80, h=100)]
        result = pf.apply(frame, dets)
        assert result.shape == frame.shape
        # Blurred region should differ from original
        assert not np.array_equal(result[20:90, 15:85], frame[20:90, 15:85])

    def test_pixelate_mode(self):
        pf = PrivacyFilter(mode="pixelate", target="person")
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        dets = [_det(x=10, y=10, w=80, h=100)]
        result = pf.apply(frame, dets)
        assert result.shape == frame.shape

    def test_blackout_mode(self):
        pf = PrivacyFilter(mode="blackout", target="person")
        frame = np.ones((200, 300, 3), dtype=np.uint8) * 128
        dets = [_det(x=10, y=10, w=80, h=100)]
        result = pf.apply(frame, dets)
        # Blacked-out region should be zeros
        assert result[15:85, 15:85].sum() == 0

    def test_no_matching_detections(self):
        pf = PrivacyFilter(mode="blur", target="person")
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        dets = [_det(class_name="car", class_id=2)]
        result = pf.apply(frame, dets)
        # No person detected, frame should be unchanged
        assert np.array_equal(result, frame)

    def test_face_target(self):
        pf = PrivacyFilter(mode="blur", target="face")
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        # "person" should not be filtered when target is "face"
        dets = [_det(class_name="person")]
        result = pf.apply(frame, dets)
        assert np.array_equal(result, frame)

    def test_all_target(self):
        pf = PrivacyFilter(mode="blur", target="all")
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        dets = [_det(class_name="person")]
        result = pf.apply(frame, dets)
        assert not np.array_equal(result, frame)

    def test_empty_detections(self):
        pf = PrivacyFilter(mode="blur")
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = pf.apply(frame, [])
        assert np.array_equal(result, frame)

    def test_mode_setter(self):
        pf = PrivacyFilter()
        pf.mode = "pixelate"
        assert pf.mode == "pixelate"
        with pytest.raises(ValueError):
            pf.mode = "bad"

    def test_target_setter(self):
        pf = PrivacyFilter()
        pf.target = "face"
        assert pf.target == "face"
        with pytest.raises(ValueError):
            pf.target = "bad"

    def test_does_not_modify_original(self):
        pf = PrivacyFilter(mode="blur")
        frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        original = frame.copy()
        pf.apply(frame, [_det()])
        assert np.array_equal(frame, original)
