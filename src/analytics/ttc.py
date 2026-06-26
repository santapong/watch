"""Time-to-collision / time-to-contact estimation (pure numpy).

Two cues, per tracked object:

- **Looming** (always available, unit-free): an approaching object's image footprint
  grows. From the linear-size ratio across frames, TTC = dt / (size_ratio - 1).
- **Range** (metric only): TTC = distance / closing_speed, using a metric distance and its
  rate (e.g. from :class:`src.tracking.range_filter.RangeTracker`).

The range cue is **only** used when ``depth_units == "metric"`` — relative/inverse depth is
not a metric distance, so ``Z / (dZ/dt)`` would be meaningless (and sign-ambiguous) on it.
The combined TTC is the soonest (min) finite cue. Pure numpy; no heavy deps — any
``RangeTracker`` is dependency-injected by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.models.base import Detection


def bbox_scale(bbox: tuple[float, float, float, float], mode: str = "area") -> float:
    """Linear size proxy for a bbox: width, height, or sqrt(area) (default)."""
    x1, y1, x2, y2 = bbox
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    if mode == "width":
        return w
    if mode == "height":
        return h
    return math.sqrt(w * h)


def ttc_from_scale(prev_size: float, curr_size: float, dt: float, deadband: float = 0.02) -> float:
    """Looming TTC from linear-size growth: dt / (curr/prev - 1).

    Returns +inf when not growing beyond ``deadband`` (static or receding) or on bad input.
    """
    if prev_size <= 0 or curr_size <= 0 or dt <= 0:
        return math.inf
    growth = curr_size / prev_size - 1.0
    if growth <= deadband:
        return math.inf
    return dt / growth


def ttc_from_range(distance: float | None, rate: float, deadband: float = 1e-3) -> float:
    """Metric TTC: distance / closing_speed (closing = -rate, positive when approaching).

    Caller must ensure ``distance`` is in meters. Returns +inf when not approaching or on
    bad input.
    """
    if distance is None or distance <= 0:
        return math.inf
    closing = -rate
    if closing <= deadband:
        return math.inf
    return distance / closing


@dataclass
class TTCResult:
    track_id: int
    ttc: float          # soonest finite cue (seconds), or +inf
    ttc_scale: float    # looming cue
    ttc_range: float    # metric range cue (inf unless metric depth + range tracker)


class TTCEstimator:
    """Per-track TTC with EMA-smoothed looming and an optional metric range cue."""

    def __init__(self, dt: float = 1.0, ema_alpha: float = 0.5, deadband: float = 0.02,
                 scale_mode: str = "area", depth_units: str = "relative", range_tracker=None):
        self._dt = dt
        self._alpha = ema_alpha
        self._deadband = deadband
        self._mode = scale_mode
        self._units = depth_units
        self._rt = range_tracker
        self._size: dict[int, float] = {}      # smoothed size per track
        self._last: dict[int, TTCResult] = {}

    def update(self, track_id: int, bbox, depth: float | None = None,
               dt: float | None = None) -> TTCResult:
        dt = self._dt if dt is None else dt
        size = bbox_scale(bbox, self._mode)
        prev = self._size.get(track_id)
        smoothed = size if prev is None else self._alpha * size + (1.0 - self._alpha) * prev
        ttc_scale = math.inf if prev is None else ttc_from_scale(prev, smoothed, dt, self._deadband)
        self._size[track_id] = smoothed

        # Range cue: metric units only (relative inverse-depth is not a metric distance).
        ttc_range = math.inf
        if self._units == "metric" and depth is not None and self._rt is not None:
            smoothed_z, rate = self._rt.update(track_id, depth, dt)
            ttc_range = ttc_from_range(smoothed_z, rate)

        result = TTCResult(track_id=track_id, ttc=min(ttc_scale, ttc_range),
                           ttc_scale=ttc_scale, ttc_range=ttc_range)
        self._last[track_id] = result
        return result

    def update_batch(self, detections: list[Detection], dt: float | None = None) -> dict[int, TTCResult]:
        """Update every tracked detection; returns {track_id: TTCResult} (skips untracked)."""
        out: dict[int, TTCResult] = {}
        for det in detections:
            if det.track_id is None:
                continue
            out[det.track_id] = self.update(det.track_id, det.bbox, det.depth, dt)
        return out

    def min_ttc(self) -> float:
        """Soonest TTC across the most recent batch (+inf if none)."""
        return min((r.ttc for r in self._last.values()), default=math.inf)

    def drop(self, track_id: int) -> None:
        self._size.pop(track_id, None)
        self._last.pop(track_id, None)
        if self._rt is not None:
            self._rt.drop(track_id)


def build_ttc_estimator(cfg: dict, range_tracker=None, depth_units: str = "relative") -> TTCEstimator:
    """Build a TTCEstimator from a ``ttc`` config block."""
    cfg = dict(cfg or {})
    return TTCEstimator(
        dt=cfg.get("dt", 1.0),
        ema_alpha=cfg.get("ema_alpha", 0.5),
        deadband=cfg.get("deadband", 0.02),
        scale_mode=cfg.get("scale_mode", "area"),
        depth_units=depth_units,
        range_tracker=range_tracker,
    )
