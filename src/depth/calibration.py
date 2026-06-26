"""Relative-depth -> metric scale calibration (pure, dependency-light).

Monocular relative depth (Depth Anything / MiDaS) is affine-ambiguous *inverse* depth:
for a fixed camera + model, the raw estimator output ``d_rel`` (larger = nearer) relates to
metric distance ``Z`` (meters, smaller = nearer) approximately by

    1 / Z = a * d_rel + b      =>      Z = 1 / (a * d_rel + b)

Given a few references (a relative sample observed at a known distance),
``DepthScaleCalibrator`` fits ``(a, b)`` by least squares and converts relative depth to
approximate meters — letting the existing *relative* backend report meters without swapping
to a metric model.

IMPORTANT: fit and apply on the RAW estimator output (``estimator.estimate(frame)``), not on
``percentile_normalize``'d values — per-frame normalization rescales every frame and would
make a fixed calibration meaningless.
"""

from __future__ import annotations

import numpy as np


class DepthScaleCalibrator:
    """Affine-in-inverse-depth map from relative depth to metric meters."""

    def __init__(self, a: float | None = None, b: float | None = None):
        self._a = a
        self._b = b

    @property
    def is_fitted(self) -> bool:
        return self._a is not None and self._b is not None

    @property
    def params(self) -> tuple[float, float]:
        if not self.is_fitted:
            raise RuntimeError("DepthScaleCalibrator is not fitted")
        return self._a, self._b

    def fit(self, rel_samples, known_meters) -> "DepthScaleCalibrator":
        """Least-squares fit of ``1/Z = a*d_rel + b`` from matched references.

        Args:
            rel_samples: raw relative-depth samples (one per reference).
            known_meters: the true distance in meters for each sample (> 0).

        Raises:
            ValueError: fewer than 2 pairs, length mismatch, or a non-positive distance.
        """
        d = np.asarray(rel_samples, dtype=np.float64).ravel()
        z = np.asarray(known_meters, dtype=np.float64).ravel()
        if d.size < 2 or d.size != z.size:
            raise ValueError("need >=2 matched (rel_sample, known_meters) pairs")
        if np.any(z <= 0):
            raise ValueError("known_meters must be positive")
        inv_z = 1.0 / z
        a_mat = np.vstack([d, np.ones_like(d)]).T
        (a, b), *_ = np.linalg.lstsq(a_mat, inv_z, rcond=None)
        self._a, self._b = float(a), float(b)
        return self

    def to_meters(self, d_rel: float) -> float:
        """Convert a single relative-depth value to meters (inf if non-positive denom)."""
        a, b = self.params
        denom = a * float(d_rel) + b
        return float("inf") if denom <= 1e-9 else 1.0 / denom

    def to_meters_array(self, d_rel) -> np.ndarray:
        """Vectorized :meth:`to_meters`; non-positive denominators map to ``inf``."""
        a, b = self.params
        d = np.asarray(d_rel, dtype=np.float64)
        denom = a * d + b
        out = np.full(denom.shape, np.inf, dtype=np.float64)
        valid = denom > 1e-9
        out[valid] = 1.0 / denom[valid]
        return out

    @classmethod
    def from_config(cls, cfg: dict) -> "DepthScaleCalibrator | None":
        """Build from ``{"a": float, "b": float}``; returns None if either is missing."""
        cfg = cfg or {}
        if cfg.get("a") is None or cfg.get("b") is None:
            return None
        return cls(float(cfg["a"]), float(cfg["b"]))
