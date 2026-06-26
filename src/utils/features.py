"""XFeat lightweight local-feature matching (lazy scaffold).

XFeat gives CPU-real-time deep local features for cross-camera correspondence and
frame-to-frame stabilization. The model is loaded lazily (or injected for tests);
``mutual_nearest_matches`` (pure numpy) and ``estimate_homography`` (cv2) are the
unit-tested core. Weights are not bundled (see requirements-phase2.txt).
"""

from __future__ import annotations

import cv2
import numpy as np


def mutual_nearest_matches(desc_a: np.ndarray, desc_b: np.ndarray) -> list[tuple[int, int]]:
    """Mutual nearest-neighbour matches between two descriptor sets (L2).

    Returns ``(i, j)`` index pairs where ``a[i]``'s nearest in B is ``b[j]`` and
    vice-versa.
    """
    a = np.asarray(desc_a, dtype=np.float32)
    b = np.asarray(desc_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return []
    # pairwise squared L2 distances (na, nb)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    nn_ab = d.argmin(axis=1)  # best B for each A
    nn_ba = d.argmin(axis=0)  # best A for each B
    return [(int(i), int(j)) for i, j in enumerate(nn_ab) if nn_ba[j] == i]


def estimate_homography(pts_a, pts_b, ransac_thresh: float = 3.0):
    """RANSAC homography from matched point sets (>=4 pairs), or None."""
    a = np.asarray(pts_a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(pts_b, dtype=np.float32).reshape(-1, 2)
    if a.shape[0] < 4 or a.shape != b.shape:
        return None
    H, _ = cv2.findHomography(a, b, cv2.RANSAC, ransac_thresh)
    return H


class XFeatMatcher:
    """XFeat feature extractor + matcher (lazy torch model, or injected for tests)."""

    def __init__(self, top_k: int = 4096, device: str = "", model=None):
        self._top_k = top_k
        self._device = device
        if model is not None:
            self._model = model
        else:
            import torch  # lazy heavy import

            # XFeat ships via torch.hub; kept lazy so import stays light.
            self._model = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (keypoints (N,2), descriptors (N,D)) for a frame."""
        out = self._model.detectAndCompute(frame, top_k=self._top_k)[0]
        return np.asarray(out["keypoints"]), np.asarray(out["descriptors"])

    def match(self, frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return matched point arrays (pts_a, pts_b) between two frames."""
        kp_a, desc_a = self.detect(frame_a)
        kp_b, desc_b = self.detect(frame_b)
        pairs = mutual_nearest_matches(desc_a, desc_b)
        if not pairs:
            return np.empty((0, 2)), np.empty((0, 2))
        ia, ib = zip(*pairs)
        return kp_a[list(ia)], kp_b[list(ib)]
