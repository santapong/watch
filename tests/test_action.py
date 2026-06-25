"""Tests for multi-cue, debounced fall detection and fall alerting."""

import numpy as np

from src.alerts import AlertManager, AlertRule
from src.models.pose_detector import ActionClassifier, PoseResult


def _pose(track_id, shoulders, hips, bbox, conf=0.9, extra=None):
    """Build a PoseResult with shoulders (5,6) and hips (11,12) populated.

    Args:
        shoulders/hips: (centre_x, y) — left/right offset by +-20/15 in x.
        bbox: (x1, y1, x2, y2).
        extra: optional {keypoint_index: (x, y, conf)} to set more joints.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    scx, sy = shoulders
    hcx, hy = hips
    kp[5] = [scx - 20, sy, conf]   # left shoulder
    kp[6] = [scx + 20, sy, conf]   # right shoulder
    kp[11] = [hcx - 15, hy, conf]  # left hip
    kp[12] = [hcx + 15, hy, conf]  # right hip
    for idx, vals in (extra or {}).items():
        kp[idx] = list(vals)
    return PoseResult(bbox=bbox, confidence=0.9, keypoints=kp, track_id=track_id)


def _standing(track_id=1):
    # Torso vertical (shoulders above hips), tall bbox.
    return _pose(track_id, (100, 100), (100, 200), (80, 20, 140, 200))


def _fallen(track_id=1):
    # Torso horizontal, wide (lying-shaped) bbox.
    return _pose(track_id, (100, 205), (200, 210), (80, 180, 280, 260))


class TestFallDetection:
    def test_genuine_fall_detected_after_debounce(self):
        clf = ActionClassifier(fall_debounce=3)
        for _ in range(3):
            assert clf.classify(_standing())[0] != "falling"
        results = [clf.classify(_fallen())[0] for _ in range(4)]
        assert results[0] != "falling"   # first fall-like frame: streak 1 < debounce
        assert results[-1] == "falling"  # confirmed after debounce

    def test_single_frame_blip_not_reported(self):
        clf = ActionClassifier(fall_debounce=3)
        clf.classify(_standing())
        clf.classify(_standing())
        assert clf.classify(_fallen())[0] != "falling"   # one blip, streak 1
        assert clf.classify(_standing())[0] != "falling"  # streak resets

    def test_sitting_not_classified_as_fall(self):
        clf = ActionClassifier(fall_debounce=3)

        def sit():
            # Upright torso, tall bbox, knees just below hips.
            return _pose(2, (100, 100), (100, 180), (80, 20, 140, 220),
                         extra={13: (90, 190, 0.9), 14: (110, 190, 0.9)})

        results = [clf.classify(sit())[0] for _ in range(5)]
        assert "falling" not in results

    def test_confidence_is_graded(self):
        clf = ActionClassifier(fall_debounce=3)
        for _ in range(2):
            clf.classify(_standing())
        confs = [clf.classify(_fallen())[1] for _ in range(4)]
        falling_confs = [c for c in confs if c > 0]
        # Later confirmed frames should be at least as confident as the first.
        assert falling_confs[-1] >= falling_confs[0]
        assert max(confs) <= 0.97


class TestFallAlert:
    def _manager(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="fall_detection",
            condition=lambda ctx: ctx.get("fall_detected", False),
            alert_type="fall",
            message="Fall detected",
            severity="critical",
            cooldown=0,
        ))
        return mgr

    def test_alert_fires_on_fall(self):
        alerts = self._manager().evaluate({"fall_detected": True, "track_ids": [1]})
        assert len(alerts) == 1
        assert alerts[0].alert_type == "fall"
        assert alerts[0].severity == "critical"

    def test_no_alert_without_fall(self):
        assert self._manager().evaluate({"fall_detected": False}) == []
