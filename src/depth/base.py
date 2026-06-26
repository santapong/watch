"""Monocular depth estimation: interfaces and dependency-light helpers.

``BaseDepthEstimator`` is the model interface. ``percentile_normalize`` /
``sample_depth`` / ``annotate_depth`` are pure numpy helpers that turn a raw depth
map into robust per-detection depth values. Concrete ONNX estimators live in
``onnx_estimator.py`` and lazy-import their heavy runtime, so importing this module
never requires onnxruntime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models.base import Detection


class BaseDepthEstimator(ABC):
    """Interface for monocular depth estimators."""

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Return a HxW float depth map aligned to ``frame`` (relative depth).

        Convention: larger value = nearer (inverse depth), matching Depth Anything /
        MiDaS outputs. Use :func:`percentile_normalize` for a robust [0, 1] map.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the loaded depth model."""

    @property
    def units(self) -> str:
        """Scale of ``estimate`` output: ``'relative'`` (inverse depth, larger = nearer)
        or ``'metric'`` (meters, smaller = nearer). Metric estimators override this."""
        return "relative"

    def reset(self) -> None:
        """Reset any per-stream state (no-op for stateless estimators; overridden by
        temporal/streaming estimators)."""


def percentile_normalize(depth: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Robustly normalize a depth map to [0, 1] via percentile clipping.

    Returns an all-zero map when the percentile range is degenerate.
    """
    d = np.asarray(depth, dtype=np.float32)
    lo, hi = np.percentile(d, [low, high])
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


def sample_depth(
    depth_map: np.ndarray,
    bbox: tuple[float, float, float, float],
    shrink: float = 0.2,
    use_mad: bool = True,
) -> float | None:
    """Robust per-detection depth from a depth map.

    Shrinks the bbox toward its centre (to avoid background leak), optionally rejects
    outliers with a MAD filter, and returns the median. Returns None when the region
    is empty/degenerate.

    Args:
        depth_map: HxW depth map.
        bbox: (x1, y1, x2, y2) in pixel coords.
        shrink: fraction of width/height trimmed from the bbox (0 = no shrink).
        use_mad: apply a 3-sigma MAD outlier rejection before the median.
    """
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox
    dx = (x2 - x1) * shrink / 2.0
    dy = (y2 - y1) * shrink / 2.0
    xi1, yi1 = max(0, int(round(x1 + dx))), max(0, int(round(y1 + dy)))
    xi2, yi2 = min(w, int(round(x2 - dx))), min(h, int(round(y2 - dy)))
    if xi2 - xi1 < 1 or yi2 - yi1 < 1:
        return None
    roi = np.asarray(depth_map[yi1:yi2, xi1:xi2], dtype=np.float32).ravel()
    if roi.size == 0:
        return None
    if use_mad and roi.size >= 4:
        med = float(np.median(roi))
        mad = float(np.median(np.abs(roi - med)))
        if mad > 1e-6:
            keep = np.abs(roi - med) <= 3.0 * 1.4826 * mad
            if keep.any():
                roi = roi[keep]
    return float(np.median(roi))


def is_too_close(depth: float | None, threshold: float, units: str = "relative") -> bool:
    """Proximity test that respects the depth convention.

    - ``relative`` (inverse depth, larger = nearer): too close when ``depth >= threshold``.
    - ``metric`` (meters, smaller = nearer): too close when ``depth <= threshold``.

    Returns ``False`` for a missing depth. Raises ``ValueError`` on an unknown ``units``
    so a sloppily-set unit can't silently flip the comparison.
    """
    if depth is None:
        return False
    if units == "metric":
        return depth <= threshold
    if units == "relative":
        return depth >= threshold
    raise ValueError(f"unknown depth units {units!r}; expected 'relative' or 'metric'")


def prepare_depth_map(raw_map: np.ndarray, units: str = "relative") -> np.ndarray:
    """Turn a raw estimator output into a per-detection-ready depth map.

    Metric maps are already in meters and returned unchanged; relative/inverse maps are
    robustly normalized to [0, 1] via :func:`percentile_normalize`. Raises ``ValueError``
    on an unknown ``units``.
    """
    if units == "metric":
        return np.asarray(raw_map, dtype=np.float32)
    if units == "relative":
        return percentile_normalize(raw_map)
    raise ValueError(f"unknown depth units {units!r}; expected 'relative' or 'metric'")


def annotate_depth(
    detections: list[Detection],
    depth_map: np.ndarray,
    shrink: float = 0.2,
    units: str | None = None,
) -> list[Detection]:
    """Populate each detection's ``depth`` (in place) by sampling the depth map.

    If ``units`` is given (``'relative'``|``'metric'``), it is stamped on
    ``detection.depth_units`` so downstream consumers know the scale.
    """
    for det in detections:
        det.depth = sample_depth(depth_map, det.bbox, shrink=shrink)
        if units is not None:
            det.depth_units = units
    return detections
