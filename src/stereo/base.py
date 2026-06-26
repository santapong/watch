"""Stereo matching interfaces + pure disparity->depth math (LAZY SCAFFOLD).

Disparity from a rectified stereo pair converts to metric depth by
``Z = fx * baseline / disparity``. The neural matcher (e.g. ESMStereo) lazy-loads its
runtime in ``onnx_matcher.py``; everything here is pure numpy/cv2 and unit-tested. A real
stereo rig + weights are required for inference, so this subsystem is NOT validated here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class BaseStereoMatcher(ABC):
    """Interface for stereo disparity estimators."""

    @abstractmethod
    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return a HxW float disparity map (pixels) for a rectified stereo pair."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the loaded stereo model."""


@dataclass
class StereoRig:
    """Calibrated stereo geometry. Only ``fx`` and ``baseline`` drive depth; the rest are
    carried for rectification/back-projection by callers."""

    fx: float
    baseline: float  # metres between the two camera centres
    cx: float | None = None
    cy: float | None = None
    fy: float | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "StereoRig | None":
        """Build from ``{"fx":..., "baseline":...}``; None if either is missing."""
        cfg = cfg or {}
        if cfg.get("fx") is None or cfg.get("baseline") is None:
            return None
        return cls(fx=float(cfg["fx"]), baseline=float(cfg["baseline"]),
                   cx=cfg.get("cx"), cy=cfg.get("cy"), fy=cfg.get("fy"))


def disparity_to_depth(disparity: float, fx: float, baseline: float,
                       *, min_disparity: float = 1e-6) -> float:
    """Metric depth (m) from a single disparity. +inf when disparity <= min_disparity."""
    d = float(disparity)
    if d <= min_disparity:
        return float("inf")
    return float(fx * baseline / d)


def disparity_to_depth_map(disparity: np.ndarray, fx: float, baseline: float,
                           *, min_disparity: float = 1e-6) -> np.ndarray:
    """Vectorized disparity->depth (m). Invalid pixels (disparity <= min) map to +inf, so a
    robust sampler must ignore non-finite values rather than median over them."""
    disp = np.asarray(disparity, dtype=np.float64)
    out = np.full(disp.shape, np.inf, dtype=np.float64)
    valid = disp > min_disparity
    out[valid] = fx * baseline / disp[valid]
    return out


def rectify_stereo_pair(left, right, map_left, map_right):
    """Apply precomputed rectification maps (``cv2.remap``) to a stereo pair."""
    import cv2

    rl = cv2.remap(left, map_left[0], map_left[1], cv2.INTER_LINEAR)
    rr = cv2.remap(right, map_right[0], map_right[1], cv2.INTER_LINEAR)
    return rl, rr
