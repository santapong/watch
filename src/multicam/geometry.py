"""Camera ground-plane geometry for multi-camera fusion.

Build a per-camera homography from image<->ground-plane point correspondences,
project a detection's foot point into a shared bird's-eye view (BEV), and match
detections across cameras by BEV proximity via Hungarian assignment.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models.base import Detection

_BIG = 1e9


def build_homography(image_points, ground_points) -> np.ndarray | None:
    """Homography mapping image pixels to ground-plane coords (>=4 point pairs).

    Returns a 3x3 matrix, or None if the inputs are malformed / too few points.
    """
    img = np.asarray(image_points, dtype=np.float32)
    grd = np.asarray(ground_points, dtype=np.float32)
    if img.ndim != 2 or img.shape != grd.shape or img.shape[0] < 4 or img.shape[1] != 2:
        return None
    if img.shape[0] == 4:
        return cv2.getPerspectiveTransform(img, grd)
    H, _ = cv2.findHomography(img, grd, cv2.RANSAC)
    return H


def ground_point(det: Detection) -> tuple[float, float]:
    """The foot point used for projection: bottom-centre of the bbox."""
    x1, y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2.0, y2)


def project_point(H, point) -> tuple[float, float]:
    """Project an image point through homography ``H`` to ground coords."""
    pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, np.asarray(H, dtype=np.float32))
    return (float(out[0, 0, 0]), float(out[0, 0, 1]))


def project_detections(H, detections: list[Detection]) -> list[tuple[float, float]]:
    """Project each detection's foot point to the ground plane."""
    return [project_point(H, ground_point(d)) for d in detections]


def match_bev(
    points_a: list[tuple[float, float]],
    points_b: list[tuple[float, float]],
    classes_a: list[int],
    classes_b: list[int],
    max_distance: float = 2.0,
) -> list[tuple[int, int, float]]:
    """Hungarian match two sets of BEV points (same class, within max_distance).

    Returns a list of ``(index_a, index_b, distance)`` for accepted assignments.
    """
    if not points_a or not points_b:
        return []
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    blocked = cost > max_distance
    for i in range(len(points_a)):
        for j in range(len(points_b)):
            if classes_a[i] != classes_b[j]:
                blocked[i, j] = True
    cost = np.where(blocked, _BIG, cost)
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c), float(cost[r, c]))
        for r, c in zip(rows, cols)
        if cost[r, c] < _BIG
    ]
