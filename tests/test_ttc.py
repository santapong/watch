"""Tests for time-to-collision estimation (pure numpy; RangeTracker injected)."""

import math

import pytest

from src.analytics.ttc import (
    TTCEstimator,
    bbox_scale,
    build_ttc_estimator,
    ttc_from_range,
    ttc_from_scale,
)
from src.models.base import Detection
from src.tracking.range_filter import RangeTracker


def _det(track_id, x2, y2, depth=None):
    d = Detection(bbox=(0.0, 0.0, float(x2), float(y2)), confidence=0.9,
                  class_id=0, class_name="person", track_id=track_id)
    d.depth = depth
    return d


class TestPureCues:
    def test_bbox_scale_modes(self):
        assert bbox_scale((0, 0, 4, 9), "width") == 4
        assert bbox_scale((0, 0, 4, 9), "height") == 9
        assert bbox_scale((0, 0, 4, 9), "area") == pytest.approx(6.0)  # sqrt(36)

    def test_scale_ttc_growing(self):
        assert ttc_from_scale(10, 11, 1.0, deadband=0.0) == pytest.approx(10.0)

    def test_scale_ttc_static_or_receding_is_inf(self):
        assert ttc_from_scale(10, 10.1, 1.0, deadband=0.02) == math.inf  # within deadband
        assert ttc_from_scale(10, 9, 1.0) == math.inf                    # receding
        assert ttc_from_scale(0, 11, 1.0) == math.inf                    # bad input

    def test_range_ttc(self):
        assert ttc_from_range(10.0, -2.0) == pytest.approx(5.0)  # closing 2 m/s
        assert ttc_from_range(10.0, 2.0) == math.inf             # receding
        assert ttc_from_range(10.0, 0.0) == math.inf            # not moving
        assert ttc_from_range(None, -2.0) == math.inf
        assert ttc_from_range(0.0, -2.0) == math.inf


class TestEstimator:
    def test_looming_approaching(self):
        est = TTCEstimator(dt=1.0, ema_alpha=1.0, deadband=0.0)
        assert est.update(1, (0, 0, 10, 10)).ttc_scale == math.inf  # first frame
        r = est.update(1, (0, 0, 11, 11))
        assert r.ttc_scale == pytest.approx(10.0)
        assert r.ttc == pytest.approx(10.0)

    def test_looming_static_is_inf(self):
        est = TTCEstimator(dt=1.0, ema_alpha=1.0)
        est.update(1, (0, 0, 10, 10))
        assert est.update(1, (0, 0, 10, 10)).ttc == math.inf

    def test_range_cue_disabled_for_relative_units(self):
        # the blocker fix: relative inverse-depth must NOT drive range TTC
        rt = RangeTracker(process_var=0.5, meas_var=0.5)
        est = TTCEstimator(ema_alpha=1.0, depth_units="relative", range_tracker=rt)
        est.update(1, (0, 0, 10, 10), depth=10.0)
        r = est.update(1, (0, 0, 10, 10), depth=8.0)  # depth changing, but relative
        assert r.ttc_range == math.inf

    def test_range_cue_active_for_metric(self):
        rt = RangeTracker(process_var=0.5, meas_var=0.5)
        est = TTCEstimator(ema_alpha=1.0, depth_units="metric", range_tracker=rt)
        r = None
        for z in (10.0, 9.0, 8.0, 7.0, 6.0, 5.0):  # approaching ~1 m/frame
            r = est.update(1, (0, 0, 10, 10), depth=z)  # static bbox -> looming inf
        assert math.isfinite(r.ttc_range) and r.ttc_range > 0
        assert r.ttc == r.ttc_range  # range cue wins (looming is inf)

    def test_update_batch_and_min_ttc(self):
        est = TTCEstimator(dt=1.0, ema_alpha=1.0, deadband=0.0)
        est.update_batch([_det(1, 10, 10), _det(2, 10, 10)])           # seed sizes
        est.update_batch([_det(1, 12, 12), _det(2, 10, 10), _det(None, 5, 5)])  # 1 grows
        assert math.isfinite(est.min_ttc())  # track 1 is approaching

    def test_drop(self):
        est = TTCEstimator(ema_alpha=1.0)
        est.update(1, (0, 0, 10, 10))
        est.drop(1)
        assert est.min_ttc() == math.inf


def test_build_ttc_estimator_passes_units():
    est = build_ttc_estimator({"deadband": 0.05}, depth_units="metric")
    assert isinstance(est, TTCEstimator)
    assert est._units == "metric"
    assert est._deadband == 0.05
