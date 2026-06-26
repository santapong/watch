"""Tests for single-view ground-plane ranging (pure numpy + cv2)."""

import math

import numpy as np
import pytest

from src.depth.ground_plane import (
    GroundPlaneHomographyRanger,
    PinholeGroundRanger,
    annotate_ground_range,
    build_ground_ranger,
)
from src.models.base import Detection


def _det(x1, y1, x2, y2):
    return Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                     confidence=0.9, class_id=0, class_name="person")


class TestPinhole:
    def _ranger(self, report="euclidean", pitch_deg=30.0):
        return PinholeGroundRanger(fx=1000, fy=1000, cx=320, cy=240,
                                   height_m=1.5, pitch_rad=math.radians(pitch_deg),
                                   report=report)

    def test_center_euclidean(self):
        # foot at the principal point: straight-line range = h / sin(pitch)
        d = self._ranger("euclidean").foot_to_meters(320, 240)
        assert d == pytest.approx(1.5 / math.sin(math.radians(30)), rel=1e-4)  # 3.0 m

    def test_center_forward(self):
        # horizontal ground range = h / tan(pitch)
        d = self._ranger("forward").foot_to_meters(320, 240)
        assert d == pytest.approx(1.5 / math.tan(math.radians(30)), rel=1e-4)  # ~2.598 m

    def test_lower_in_image_is_nearer(self):
        r = self._ranger()
        d_far, d_mid, d_near = r.foot_to_meters(320, 240), r.foot_to_meters(320, 300), r.foot_to_meters(320, 400)
        assert d_far > d_mid > d_near  # larger v (lower foot) = closer object

    def test_above_horizon_returns_none(self):
        # shallow pitch puts the horizon on-screen; a foot above it has no ground hit
        assert self._ranger(pitch_deg=5.0).foot_to_meters(320, 100) is None

    def test_detection_uses_foot_point(self):
        r = self._ranger()
        det = _det(300, 100, 340, 400)  # foot = (320, 400)
        assert r.detection_to_meters(det) == pytest.approx(r.foot_to_meters(320, 400))

    def test_units_metric(self):
        assert self._ranger().units == "metric"

    def test_bad_report_raises(self):
        with pytest.raises(ValueError):
            PinholeGroundRanger(1000, 1000, 320, 240, 1.5, 0.5, report="sideways")


class TestHomography:
    def _ranger(self):
        # image square (100px) -> ground unit square (meters)
        return GroundPlaneHomographyRanger.from_points(
            image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            ground_points=[(0, 0), (1, 0), (1, 1), (0, 1)],
        )

    def test_maps_to_ground_distance(self):
        r = self._ranger()
        assert r.foot_to_meters(50, 50) == pytest.approx(math.sqrt(0.5), rel=1e-4)
        assert r.foot_to_meters(100, 100) == pytest.approx(math.sqrt(2.0), rel=1e-4)

    def test_from_points_requires_four(self):
        with pytest.raises(ValueError):
            GroundPlaneHomographyRanger.from_points([(0, 0), (1, 1)], [(0, 0), (1, 1)])

    def test_units_metric(self):
        assert self._ranger().units == "metric"


class TestAnnotateAndFactory:
    def test_annotate_sets_depth_and_units(self):
        r = PinholeGroundRanger(1000, 1000, 320, 240, 1.5, math.radians(30))
        det = _det(300, 380, 340, 420)
        annotate_ground_range([det], r)
        assert det.depth is not None and det.depth > 0
        assert det.depth_units == "metric"

    def test_annotate_leaves_above_horizon_none(self):
        r = PinholeGroundRanger(1000, 1000, 320, 240, 1.5, math.radians(5))
        det = _det(300, 80, 340, 100)  # foot v=100 is above the horizon
        annotate_ground_range([det], r)
        assert det.depth is None and det.depth_units is None

    def test_build_disabled_returns_none(self):
        assert build_ground_ranger({"enabled": False}) is None
        assert build_ground_ranger({}) is None

    def test_build_pinhole(self):
        r = build_ground_ranger({
            "enabled": True, "mode": "pinhole",
            "fx": 1000, "fy": 1000, "cx": 320, "cy": 240, "height_m": 1.5, "pitch_deg": 30,
        })
        assert isinstance(r, PinholeGroundRanger)
        assert r.detection_to_meters(_det(300, 380, 340, 420)) is not None

    def test_build_pinhole_missing_param_raises(self):
        with pytest.raises(ValueError):
            build_ground_ranger({"enabled": True, "mode": "pinhole", "fx": 1000})

    def test_build_homography(self):
        r = build_ground_ranger({
            "enabled": True, "mode": "homography",
            "image_points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "ground_points": [(0, 0), (1, 0), (1, 1), (0, 1)],
        })
        assert isinstance(r, GroundPlaneHomographyRanger)

    def test_build_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            build_ground_ranger({"enabled": True, "mode": "lidar"})
