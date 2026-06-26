"""Tests for XFeat matching helpers (pure numpy + cv2)."""

import cv2
import numpy as np

from src.utils.features import estimate_homography, mutual_nearest_matches


def test_mutual_nearest_permuted_identicals():
    a = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    b = np.array([[2, 2], [0, 0], [1, 1]], dtype=np.float32)
    assert dict(mutual_nearest_matches(a, b)) == {0: 1, 1: 2, 2: 0}


def test_mutual_nearest_empty():
    assert mutual_nearest_matches(np.zeros((0, 4)), np.ones((3, 4))) == []


def test_estimate_homography_identity():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    H = estimate_homography(pts, pts)
    assert H is not None
    p = cv2.perspectiveTransform(np.array([[[5, 5]]], dtype=np.float32), H)
    assert np.allclose(p[0, 0], [5, 5], atol=1e-3)


def test_estimate_homography_too_few_points():
    assert estimate_homography(np.zeros((3, 2)), np.zeros((3, 2))) is None
