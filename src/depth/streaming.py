"""Temporally-consistent streaming depth (LAZY SCAFFOLD).

``TemporalDepthEstimator`` wraps any :class:`BaseDepthEstimator` and blends consecutive
depth maps (EMA, optional scale-alignment) for steadier, less-flickery depth across a video
stream. The blend / scale-align / reset logic is pure and unit-tested.

This is a lightweight, model-agnostic stand-in for a true streaming-depth network such as
**oVDA (Online Video Depth Anything)**. A real oVDA backend (recurrent state across frames)
would be plugged in as the ``inner`` estimator; its weights are **non-commercial** and need a
GPU/ONNX box, so that model itself is deferred and not validated here.
"""

from __future__ import annotations

import numpy as np

from src.depth.base import BaseDepthEstimator


def blend_depth(prev, curr, alpha: float) -> np.ndarray:
    """EMA blend ``alpha*curr + (1-alpha)*prev``.

    Returns ``curr`` unchanged when ``prev`` is None or the shapes differ (a reseed). Note:
    a persistent shape change therefore disables smoothing for the stream — callers that
    resize frames mid-stream should call :meth:`TemporalDepthEstimator.reset`.
    """
    curr = np.asarray(curr, dtype=np.float32)
    if prev is None or np.asarray(prev).shape != curr.shape:
        return curr
    return (alpha * curr + (1.0 - alpha) * np.asarray(prev, dtype=np.float32)).astype(np.float32)


def align_scale_to(curr, ref, eps: float = 1e-6) -> np.ndarray:
    """Rescale ``curr`` so its median matches ``ref``'s (tames relative-depth scale flicker).

    Only meaningful for relative/inverse depth — never apply to metric meters.
    """
    curr = np.asarray(curr, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    mc = float(np.median(curr))
    if abs(mc) < eps:
        return curr
    return (curr * (float(np.median(ref)) / mc)).astype(np.float32)


class TemporalDepthEstimator(BaseDepthEstimator):
    """Stateful temporal smoother around any depth estimator."""

    def __init__(self, inner: BaseDepthEstimator, alpha: float = 0.5,
                 align_scale: bool = False, name: str | None = None):
        if align_scale and getattr(inner, "units", "relative") == "metric":
            raise ValueError("align_scale would corrupt metric depth; use it only on relative depth")
        self._inner = inner
        self._alpha = float(alpha)
        self._align = bool(align_scale)
        self._prev: np.ndarray | None = None
        self._name = name or f"temporal({inner.model_name})"

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        curr = np.asarray(self._inner.estimate(frame), dtype=np.float32)
        if self._prev is not None and self._align and self._prev.shape == curr.shape:
            curr = align_scale_to(curr, self._prev)
        blended = blend_depth(self._prev, curr, self._alpha)
        self._prev = blended
        return blended

    def reset(self) -> None:
        """Drop the temporal state (call on scene cuts / resolution changes)."""
        self._prev = None

    @property
    def units(self) -> str:
        return getattr(self._inner, "units", "relative")

    @property
    def model_name(self) -> str:
        return self._name
