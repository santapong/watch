"""Tests for drawing utilities."""

import numpy as np
import pytest

from src.models.base import Detection
from src.utils.drawing import (
    get_color,
    draw_detections,
    draw_fps,
    draw_info,
    draw_tracks,
    draw_zones,
    draw_skeleton,
    draw_action_label,
    draw_anomaly_alert,
    draw_scene_info,
    draw_event_log,
    COLORS,
)


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(x=10, y=20, w=50, h=80, class_id=0, class_name="person", track_id=None, conf=0.9):
    return Detection(
        bbox=(float(x), float(y), float(x + w), float(y + h)),
        confidence=conf,
        class_id=class_id,
        class_name=class_name,
        track_id=track_id,
    )


class TestGetColor:
    def test_returns_tuple(self):
        color = get_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_wraps_around(self):
        assert get_color(0) == get_color(len(COLORS))


class TestDrawDetections:
    def test_returns_frame(self):
        frame = _frame()
        result = draw_detections(frame, [_det()])
        assert result is frame

    def test_empty_detections(self):
        frame = _frame()
        result = draw_detections(frame, [])
        assert result is frame

    def test_with_track_id(self):
        frame = _frame()
        result = draw_detections(frame, [_det(track_id=5)])
        assert result is frame

    def test_modifies_frame(self):
        frame = _frame()
        original = frame.copy()
        draw_detections(frame, [_det()])
        assert not np.array_equal(frame, original)


class TestDrawFps:
    def test_returns_frame(self):
        frame = _frame()
        result = draw_fps(frame, 30.0)
        assert result is frame

    def test_modifies_frame(self):
        frame = _frame()
        original = frame.copy()
        draw_fps(frame, 60.0)
        assert not np.array_equal(frame, original)


class TestDrawInfo:
    def test_returns_frame(self):
        frame = _frame()
        result = draw_info(frame, "yolov8n", 5)
        assert result is frame


class TestDrawTracks:
    def test_empty_trajectories(self):
        frame = _frame()
        result = draw_tracks(frame, {})
        assert result is frame

    def test_single_point_trajectory(self):
        frame = _frame()
        result = draw_tracks(frame, {1: [(100.0, 100.0)]})
        assert result is frame

    def test_multi_point_trajectory(self):
        frame = _frame()
        traj = {1: [(100.0, 100.0), (150.0, 150.0), (200.0, 200.0)]}
        result = draw_tracks(frame, traj)
        assert result is frame


class TestDrawZones:
    def test_single_zone(self):
        frame = _frame()
        zones = {"Zone A": np.array([[100, 100], [300, 100], [300, 300], [100, 300]])}
        result = draw_zones(frame, zones)
        assert result is frame

    def test_zone_with_counts(self):
        frame = _frame()
        zones = {"Zone A": np.array([[100, 100], [300, 100], [300, 300], [100, 300]])}
        counts = {"Zone A": 3}
        result = draw_zones(frame, zones, counts=counts)
        assert result is frame


class TestDrawSkeleton:
    def test_basic_skeleton(self):
        frame = _frame()
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Set some visible keypoints
        for i in range(17):
            keypoints[i] = [100 + i * 10, 200 + i * 5, 0.9]
        result = draw_skeleton(frame, keypoints)
        assert result is frame

    def test_low_confidence_keypoints(self):
        frame = _frame()
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # All below threshold
        for i in range(17):
            keypoints[i] = [100, 200, 0.1]
        original = frame.copy()
        draw_skeleton(frame, keypoints, confidence_threshold=0.5)
        # Frame should not be modified since all keypoints are low confidence
        assert np.array_equal(frame, original)


class TestDrawActionLabel:
    def test_returns_frame(self):
        frame = _frame()
        result = draw_action_label(frame, (10, 20, 100, 200), "walking", 0.85)
        assert result is frame


class TestDrawAnomalyAlert:
    def test_normal(self):
        frame = _frame()
        result = draw_anomaly_alert(frame, 0.1, False)
        assert result is frame

    def test_anomalous(self):
        frame = _frame()
        result = draw_anomaly_alert(frame, -0.3, True)
        assert result is frame


class TestDrawSceneInfo:
    def test_returns_frame(self):
        frame = _frame()
        result = draw_scene_info(frame, "traffic", "A busy street scene")
        assert result is frame


class TestDrawEventLog:
    def test_empty_events(self):
        frame = _frame()
        result = draw_event_log(frame, [])
        assert result is frame

    def test_with_events(self):
        frame = _frame(w=800)
        events = [
            {"description": "Person appeared", "severity": "info"},
            {"description": "Crowd forming", "severity": "warning"},
            {"description": "Anomaly detected", "severity": "alert"},
        ]
        result = draw_event_log(frame, events)
        assert result is frame
