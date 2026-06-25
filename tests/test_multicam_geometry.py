"""Tests for geometry-grounded multi-camera identity."""

import pytest

from src.models.base import Detection
from src.multicam.geometry import (
    build_homography,
    ground_point,
    match_bev,
    project_point,
)
from src.multicam.manager import MultiCameraManager

# A unit ground square (image == ground for the reference camera).
GROUND = [(0, 0), (100, 0), (100, 100), (0, 100)]


def _det(cx, cy, w=20, h=20, cls=0, name="person", conf=0.9):
    """Detection whose bbox bottom-centre (foot point) is (cx, cy)."""
    return Detection(
        bbox=(cx - w / 2, cy - h, cx + w / 2, cy),
        confidence=conf, class_id=cls, class_name=name,
    )


class TestGeometry:
    def test_build_homography_identity(self):
        H = build_homography(GROUND, GROUND)
        assert H is not None
        assert project_point(H, (50, 50)) == pytest.approx((50, 50), abs=1e-3)

    def test_build_homography_too_few_points(self):
        assert build_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)]) is None

    def test_build_homography_translation(self):
        # Image is shifted +100 in x relative to the ground plane.
        img = [(100, 0), (200, 0), (200, 100), (100, 100)]
        H = build_homography(img, GROUND)
        assert project_point(H, (150, 50)) == pytest.approx((50, 50), abs=1e-3)

    def test_ground_point_is_bottom_center(self):
        assert ground_point(_det(50, 80)) == pytest.approx((50, 80))

    def test_match_bev_pairs_nearest_same_class(self):
        out = match_bev([(0, 0), (10, 10)], [(0.5, 0.5)], [0, 0], [0], max_distance=2.0)
        assert len(out) == 1
        assert out[0][0] == 0 and out[0][1] == 0
        assert out[0][2] == pytest.approx(0.707, abs=0.01)

    def test_match_bev_blocks_different_class(self):
        assert match_bev([(0, 0)], [(0, 0)], [0], [1], max_distance=5.0) == []

    def test_match_bev_respects_max_distance(self):
        assert match_bev([(0, 0)], [(10, 0)], [0], [0], max_distance=5.0) == []


class TestCrossCameraIdentity:
    def _manager(self):
        m = MultiCameraManager()
        assert m.set_homography("A", GROUND, GROUND)  # identity
        assert m.set_homography("B", [(100, 0), (200, 0), (200, 100), (100, 100)], GROUND)
        assert m.has_homography
        return m

    def test_same_person_gets_shared_id(self):
        m = self._manager()
        dets = {
            "A": [_det(50, 50), _det(10, 10)],  # person P, person Q
            "B": [_det(150, 50)],               # person P, shifted +100 in x
        }
        ids = m.assign_global_ids(dets, max_distance=5.0)
        assert ids["A"][0] == ids["B"][0]  # P matched across cameras
        assert ids["A"][1] != ids["A"][0]  # Q is a distinct identity

    def test_matches_report_bev_distance(self):
        m = self._manager()
        matches = m.find_cross_camera_matches(
            {"A": [_det(50, 50)], "B": [_det(150, 50)]}, max_distance=5.0
        )
        assert len(matches) == 1
        assert matches[0]["bev_distance"] == pytest.approx(0.0, abs=1e-3)

    def test_fallback_without_homography(self):
        m = MultiCameraManager()  # no homographies registered
        matches = m.find_cross_camera_matches(
            {"A": [_det(50, 50, conf=0.9)], "B": [_det(150, 50, conf=0.9)]}
        )
        assert len(matches) == 1
        assert "bev_distance" not in matches[0]  # legacy class+confidence path
