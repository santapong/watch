"""Tests for tracking module."""

import time

import numpy as np
import pytest

from src.models.base import Detection
from src.tracking.tracker import TrackInfo, TrackHistory, EnhancedTracker


def _make_detection(x=10, y=20, w=50, h=80, class_id=0, class_name="person", track_id=None, conf=0.9):
    return Detection(
        bbox=(float(x), float(y), float(x + w), float(y + h)),
        confidence=conf,
        class_id=class_id,
        class_name=class_name,
        track_id=track_id,
    )


class TestTrackInfo:
    def test_creation(self):
        info = TrackInfo(track_id=1, class_name="person", class_id=0)
        assert info.track_id == 1
        assert info.class_name == "person"
        assert info.is_active is True

    def test_duration(self):
        info = TrackInfo(track_id=1, class_name="car", class_id=2)
        info.first_seen = 100.0
        info.last_seen = 105.5
        assert info.duration == pytest.approx(5.5)

    def test_distance_traveled_empty(self):
        info = TrackInfo(track_id=1, class_name="person", class_id=0)
        assert info.distance_traveled == 0.0

    def test_distance_traveled_single_point(self):
        info = TrackInfo(track_id=1, class_name="person", class_id=0, positions=[(0, 0)])
        assert info.distance_traveled == 0.0

    def test_distance_traveled(self):
        info = TrackInfo(
            track_id=1,
            class_name="person",
            class_id=0,
            positions=[(0.0, 0.0), (3.0, 4.0)],
        )
        assert info.distance_traveled == pytest.approx(5.0)

    def test_distance_traveled_multiple(self):
        info = TrackInfo(
            track_id=1,
            class_name="person",
            class_id=0,
            positions=[(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)],
        )
        assert info.distance_traveled == pytest.approx(10.0)


class TestTrackHistory:
    def test_update_creates_track(self):
        history = TrackHistory()
        dets = [_make_detection(track_id=1)]
        history.update(dets)
        assert 1 in history.get_all_tracks()

    def test_update_ignores_untracked(self):
        history = TrackHistory()
        dets = [_make_detection(track_id=None)]
        history.update(dets)
        assert len(history.get_all_tracks()) == 0

    def test_get_trajectory(self):
        history = TrackHistory()
        history.update([_make_detection(x=10, track_id=1)])
        history.update([_make_detection(x=20, track_id=1)])
        traj = history.get_trajectory(1)
        assert len(traj) == 2

    def test_get_trajectory_nonexistent(self):
        history = TrackHistory()
        assert history.get_trajectory(999) == []

    def test_active_tracks(self):
        history = TrackHistory()
        history.update([_make_detection(track_id=1), _make_detection(track_id=2)])
        assert len(history.get_active_tracks()) == 2
        # Remove track 2
        history.update([_make_detection(track_id=1)])
        assert 1 in history.get_active_tracks()
        assert 2 not in history.get_active_tracks()

    def test_max_history(self):
        history = TrackHistory(max_history=5)
        for i in range(10):
            history.update([_make_detection(x=i * 10, track_id=1)])
        traj = history.get_trajectory(1)
        assert len(traj) == 5

    def test_track_stats(self):
        history = TrackHistory()
        history.update([_make_detection(track_id=1)])
        stats = history.get_track_stats(1)
        assert stats is not None
        assert stats["track_id"] == 1
        assert stats["class_name"] == "person"
        assert stats["is_active"] is True

    def test_track_stats_nonexistent(self):
        history = TrackHistory()
        assert history.get_track_stats(999) is None

    def test_clear(self):
        history = TrackHistory()
        history.update([_make_detection(track_id=1)])
        history.clear()
        assert len(history.get_all_tracks()) == 0


class TestEnhancedTracker:
    def test_update(self):
        tracker = EnhancedTracker()
        dets = [_make_detection(track_id=1)]
        result = tracker.update(dets)
        assert len(result) == 1

    def test_update_with_frame(self):
        tracker = EnhancedTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[20:100, 10:60] = 128  # Put some content in bbox region
        dets = [_make_detection(track_id=1)]
        result = tracker.update(dets, frame)
        assert len(result) == 1

    def test_get_trajectory(self):
        tracker = EnhancedTracker()
        tracker.update([_make_detection(x=10, track_id=1)])
        tracker.update([_make_detection(x=20, track_id=1)])
        traj = tracker.get_trajectory(1)
        assert len(traj) == 2

    def test_get_all_trajectories(self):
        tracker = EnhancedTracker()
        tracker.update([_make_detection(track_id=1), _make_detection(x=200, track_id=2)])
        trajs = tracker.get_all_trajectories()
        assert len(trajs) == 2

    def test_match_reid_no_embeddings(self):
        tracker = EnhancedTracker()
        assert tracker.match_reid(1, {}) is None

    def test_match_reid_with_similar_embedding(self):
        tracker = EnhancedTracker(reid_threshold=0.5)
        emb = np.random.rand(96).astype(np.float32)
        tracker._embeddings[1] = emb
        # Similar embedding should match
        similar_emb = emb + np.random.rand(96) * 0.01
        result = tracker.match_reid(1, {99: similar_emb})
        assert result == 99

    def test_match_reid_with_different_embedding(self):
        tracker = EnhancedTracker(reid_threshold=0.9)
        tracker._embeddings[1] = np.array([1.0, 0.0, 0.0] * 32, dtype=np.float32)
        different_emb = np.array([0.0, 1.0, 0.0] * 32, dtype=np.float32)
        result = tracker.match_reid(1, {99: different_emb})
        assert result is None
