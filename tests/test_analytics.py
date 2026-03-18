"""Tests for analytics modules: anomaly detection, scene understanding, temporal events."""

import numpy as np
import pytest

from src.models.base import Detection
from src.analytics.anomaly_detector import SceneDescriptor, AnomalyDetector
from src.analytics.scene_understanding import (
    SceneAnalyzer,
    SceneDescription,
    ObjectRelation,
    SCENE_SIGNATURES,
)
from src.analytics.temporal import (
    TemporalBuffer,
    EventDetector,
    EventType,
    TemporalEvent,
)


def _det(x=10, y=20, w=50, h=80, class_id=0, class_name="person", track_id=None, conf=0.9):
    return Detection(
        bbox=(float(x), float(y), float(x + w), float(y + h)),
        confidence=conf,
        class_id=class_id,
        class_name=class_name,
        track_id=track_id,
    )


# ── SceneDescriptor ──────────────────────────────────────────────

class TestSceneDescriptor:
    def test_empty_detections(self):
        desc = SceneDescriptor(num_classes=80)
        features = desc.describe([], (720, 1280))
        assert features.shape[0] == 80 + 1 + 8 + 16  # classes + count + stats + grid
        assert features[0] == 0  # total count

    def test_single_detection(self):
        desc = SceneDescriptor(num_classes=80)
        dets = [_det(class_id=0)]
        features = desc.describe(dets, (720, 1280))
        assert features[0] == 1  # total count
        assert features[1] == 1  # class 0 count

    def test_multiple_classes(self):
        desc = SceneDescriptor(num_classes=80)
        dets = [
            _det(class_id=0, class_name="person"),
            _det(x=100, class_id=2, class_name="car"),
            _det(x=200, class_id=2, class_name="car"),
        ]
        features = desc.describe(dets, (720, 1280))
        assert features[0] == 3
        assert features[1] == 1  # person
        assert features[3] == 2  # car

    def test_feature_vector_dtype(self):
        desc = SceneDescriptor()
        features = desc.describe([_det()], (720, 1280))
        assert features.dtype == np.float32


# ── AnomalyDetector ──────────────────────────────────────────────

class TestAnomalyDetector:
    def test_initial_state(self):
        ad = AnomalyDetector(learning_frames=100)
        assert ad.is_learning is True
        assert ad.is_fitted is False
        assert ad.learning_progress == 0.0

    def test_learning_progress(self):
        ad = AnomalyDetector(learning_frames=100)
        for _ in range(50):
            ad.update([_det()])
        assert ad.learning_progress == pytest.approx(0.5)

    def test_fit_requires_minimum_data(self):
        ad = AnomalyDetector(learning_frames=100)
        with pytest.raises(ValueError, match="at least 10"):
            ad.fit()

    def test_auto_fit(self):
        ad = AnomalyDetector(learning_frames=20)
        for i in range(25):
            ad.update([_det(x=i * 10)])
        assert ad.is_fitted is True
        assert ad.learning_progress == 1.0

    def test_check_during_learning(self):
        ad = AnomalyDetector(learning_frames=100)
        score, is_anomalous = ad.check([_det()], (720, 1280))
        assert score == 0.0
        assert is_anomalous is False

    def test_check_after_fit(self):
        ad = AnomalyDetector(learning_frames=20)
        # Train on normal data
        for i in range(25):
            ad.update([_det(x=50, y=50)])
        # Check normal scene
        score, is_anomalous = ad.check([_det(x=50, y=50)], (720, 1280))
        assert isinstance(score, float)
        assert isinstance(is_anomalous, bool)


# ── SceneAnalyzer ────────────────────────────────────────────────

class TestSceneAnalyzer:
    def test_empty_scene(self):
        analyzer = SceneAnalyzer()
        result = analyzer.analyze([], (720, 1280))
        assert result.scene_type == "empty"
        assert result.scene_confidence == 1.0
        assert len(result.object_summary) == 0
        assert "Empty scene" in result.description

    def test_traffic_scene(self):
        analyzer = SceneAnalyzer()
        dets = [
            _det(class_name="car", class_id=2),
            _det(x=100, class_name="car", class_id=2),
            _det(x=200, class_name="truck", class_id=7),
            _det(x=300, class_name="traffic light", class_id=9),
        ]
        result = analyzer.analyze(dets, (720, 1280))
        assert result.scene_type == "traffic"

    def test_object_summary(self):
        analyzer = SceneAnalyzer()
        dets = [
            _det(class_name="person", class_id=0),
            _det(x=100, class_name="person", class_id=0),
            _det(x=200, class_name="dog", class_id=16),
        ]
        result = analyzer.analyze(dets, (720, 1280))
        assert result.object_summary["person"] == 2
        assert result.object_summary["dog"] == 1

    def test_spatial_relations(self):
        analyzer = SceneAnalyzer()
        dets = [
            _det(x=100, y=100, class_name="person", class_id=0),
            _det(x=110, y=110, class_name="laptop", class_id=63),
        ]
        result = analyzer.analyze(dets, (720, 1280))
        assert len(result.relations) > 0

    def test_no_relations_for_single_object(self):
        analyzer = SceneAnalyzer()
        dets = [_det(class_name="person", class_id=0)]
        result = analyzer.analyze(dets, (720, 1280))
        assert len(result.relations) == 0

    def test_description_generated(self):
        analyzer = SceneAnalyzer()
        dets = [_det(class_name="person", class_id=0)]
        result = analyzer.analyze(dets, (720, 1280))
        assert len(result.description) > 0
        assert "person" in result.description.lower()


# ── TemporalBuffer ───────────────────────────────────────────────

class TestTemporalBuffer:
    def test_add_and_states(self):
        buf = TemporalBuffer()
        buf.add([_det(track_id=1)], timestamp=100.0)
        states = buf.object_states
        assert 1 in states
        assert states[1].class_name == "person"

    def test_ignores_untracked(self):
        buf = TemporalBuffer()
        buf.add([_det(track_id=None)], timestamp=100.0)
        assert len(buf.object_states) == 0

    def test_object_duration(self):
        buf = TemporalBuffer()
        buf.add([_det(track_id=1)], timestamp=100.0)
        buf.add([_det(track_id=1)], timestamp=105.0)
        assert buf.get_object_duration(1) == pytest.approx(5.0)

    def test_object_duration_nonexistent(self):
        buf = TemporalBuffer()
        assert buf.get_object_duration(999) == 0.0

    def test_inactive_marking(self):
        buf = TemporalBuffer()
        buf.add([_det(track_id=1)], timestamp=100.0)
        buf.add([], timestamp=101.0)
        assert buf.object_states[1].is_active is False

    def test_clear(self):
        buf = TemporalBuffer()
        buf.add([_det(track_id=1)], timestamp=100.0)
        buf.clear()
        assert len(buf.object_states) == 0


# ── EventDetector ────────────────────────────────────────────────

class TestEventDetector:
    def test_detect_appearance(self):
        buf = TemporalBuffer()
        detector = EventDetector()
        buf.add([_det(track_id=1)], timestamp=100.0)
        events = detector.check(buf)
        appeared = [e for e in events if e.event_type == EventType.OBJECT_APPEARED]
        assert len(appeared) == 1
        assert appeared[0].track_id == 1

    def test_detect_disappearance(self):
        buf = TemporalBuffer()
        detector = EventDetector()
        buf.add([_det(track_id=1)], timestamp=100.0)
        detector.check(buf)  # Register track 1
        buf.add([], timestamp=101.0)  # Track 1 gone
        events = detector.check(buf)
        disappeared = [e for e in events if e.event_type == EventType.OBJECT_DISAPPEARED]
        assert len(disappeared) == 1

    def test_no_duplicate_events(self):
        buf = TemporalBuffer()
        detector = EventDetector()
        buf.add([_det(track_id=1)], timestamp=100.0)
        events1 = detector.check(buf)
        events2 = detector.check(buf)
        appeared1 = [e for e in events1 if e.event_type == EventType.OBJECT_APPEARED]
        appeared2 = [e for e in events2 if e.event_type == EventType.OBJECT_APPEARED]
        assert len(appeared1) == 1
        assert len(appeared2) == 0

    def test_crowd_detection(self):
        buf = TemporalBuffer()
        detector = EventDetector(crowd_threshold=3)
        dets = [_det(x=i * 60, track_id=i + 1) for i in range(5)]
        buf.add(dets, timestamp=100.0)
        events = detector.check(buf)
        crowd = [e for e in events if e.event_type == EventType.CROWD_FORMING]
        assert len(crowd) >= 1

    def test_reset(self):
        buf = TemporalBuffer()
        detector = EventDetector()
        buf.add([_det(track_id=1)], timestamp=100.0)
        detector.check(buf)
        detector.reset()
        # After reset, should detect appearance again
        events = detector.check(buf)
        appeared = [e for e in events if e.event_type == EventType.OBJECT_APPEARED]
        assert len(appeared) == 1
