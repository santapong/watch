"""Single-view ground-plane distance estimation (pure, CPU-only).

For a fixed camera viewing flat ground, the distance to an object follows from where its
foot-point (bbox bottom-centre) lands on the ground plane — no neural depth model needed.
Two parameterizations:

- ``PinholeGroundRanger``: from intrinsics (fx, fy, cx, cy) + camera height + pitch,
  intersect the back-projected foot ray with the ground plane (closed form).
- ``GroundPlaneHomographyRanger``: from a calibrated image->ground homography (fit from
  >=4 known ground points), map the foot-point to ground coords and measure distance.

Both report **meters** (``units == "metric"``, smaller = nearer), so results plug straight
into :func:`src.depth.base.is_too_close` / ``Detection.depth``. Uses only numpy + cv2 (slim
deps, same homography math as ``src.multicam.geometry`` but without its scipy import), so the
whole module is import-safe and fully unit-testable on CPU.

Range conventions (resolves the euclidean/forward ambiguity explicitly):
- Pinhole ``report="euclidean"`` (default): straight-line camera->point distance (includes
  camera height). ``report="forward"``: horizontal distance along the ground.
- Homography: horizontal ground distance from the camera's ground origin — a single,
  well-defined quantity (the homography maps onto the ground plane, which has no height).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.models.base import Detection


def _foot_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-centre of a bbox — where an upright object meets the ground."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


class PinholeGroundRanger:
    """Closed-form ground-plane range from intrinsics + camera height + pitch."""

    def __init__(self, fx, fy, cx, cy, height_m, pitch_rad,
                 report: str = "euclidean", name: str = "ground_pinhole"):
        if report not in ("euclidean", "forward"):
            raise ValueError("report must be 'euclidean' or 'forward'")
        self._fx, self._fy, self._cx, self._cy = float(fx), float(fy), float(cx), float(cy)
        self._h, self._theta = float(height_m), float(pitch_rad)
        self._report = report
        self._name = name

    def foot_to_meters(self, u: float, v: float) -> float | None:
        """Range (m) to the ground point imaged at pixel (u, v); None if at/above horizon."""
        # Camera ray (x right, y down, z forward), then pitch down about the x-axis.
        x = (u - self._cx) / self._fx
        y = (v - self._cy) / self._fy
        z = 1.0
        ct, st = math.cos(self._theta), math.sin(self._theta)
        wx = x
        wy = y * ct + z * st          # world "down" component
        wz = -y * st + z * ct         # world "forward" component
        if not np.isfinite(wy) or wy <= 1e-9:
            return None               # ray is parallel to / above the ground plane
        t = self._h / wy              # scale to hit ground at depth-down = height
        if not np.isfinite(t) or t <= 0:
            return None
        if self._report == "forward":
            return float(math.hypot(t * wx, t * wz))
        return float(t * math.sqrt(wx * wx + wy * wy + wz * wz))  # euclidean

    def detection_to_meters(self, det: Detection) -> float | None:
        u, v = _foot_point(det.bbox)
        return self.foot_to_meters(u, v)

    @property
    def units(self) -> str:
        return "metric"

    @property
    def model_name(self) -> str:
        return self._name


class GroundPlaneHomographyRanger:
    """Image->ground homography ranger; reports horizontal ground distance (meters)."""

    def __init__(self, homography, origin=(0.0, 0.0), name: str = "ground_homography"):
        self._H = np.asarray(homography, dtype=np.float64).reshape(3, 3)
        self._ox, self._oy = float(origin[0]), float(origin[1])
        self._name = name

    @classmethod
    def from_points(cls, image_points, ground_points, origin=(0.0, 0.0)):
        """Fit the image->ground homography from >=4 matched (pixel, ground-meter) pairs."""
        img = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
        gnd = np.asarray(ground_points, dtype=np.float32).reshape(-1, 1, 2)
        if img.shape[0] < 4 or img.shape[0] != gnd.shape[0]:
            raise ValueError("need >=4 matched image/ground point pairs")
        h_mat, _ = cv2.findHomography(img, gnd)
        if h_mat is None:
            raise ValueError("homography fit failed")
        return cls(h_mat, origin=origin)

    def foot_to_meters(self, u: float, v: float) -> float | None:
        pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
        g = cv2.perspectiveTransform(pt, self._H)[0, 0]
        gx, gy = float(g[0]), float(g[1])
        if not (np.isfinite(gx) and np.isfinite(gy)):
            return None
        return float(math.hypot(gx - self._ox, gy - self._oy))

    def detection_to_meters(self, det: Detection) -> float | None:
        u, v = _foot_point(det.bbox)
        return self.foot_to_meters(u, v)

    @property
    def units(self) -> str:
        return "metric"

    @property
    def model_name(self) -> str:
        return self._name


def annotate_ground_range(detections: list[Detection], ranger) -> list[Detection]:
    """Set ``det.depth`` (meters) + ``det.depth_units='metric'`` from a ground ranger.

    Foot-points above the horizon / non-finite yield None and the detection is left
    untouched (so a downstream RangeTracker is never fed a None range).
    """
    for det in detections:
        dist = ranger.detection_to_meters(det)
        if dist is not None:
            det.depth = dist
            det.depth_units = "metric"
    return detections


def build_ground_ranger(cfg: dict):
    """Build a ground-plane ranger from config, or None if disabled.

    cfg: ``{enabled, mode: 'pinhole'|'homography', report, fx, fy, cx, cy, height_m,
    pitch_deg, image_points, ground_points, homography, origin}``.
    """
    cfg = dict(cfg or {})
    if not cfg.get("enabled"):
        return None
    mode = (cfg.get("mode") or "pinhole").strip().lower()
    if mode == "pinhole":
        required = ("fx", "fy", "cx", "cy", "height_m")
        if any(cfg.get(k) is None for k in required):
            raise ValueError(f"ground_plane pinhole mode requires {required}")
        return PinholeGroundRanger(
            fx=cfg["fx"], fy=cfg["fy"], cx=cfg["cx"], cy=cfg["cy"],
            height_m=cfg["height_m"], pitch_rad=math.radians(float(cfg.get("pitch_deg", 0.0))),
            report=cfg.get("report", "euclidean"),
        )
    if mode == "homography":
        origin = cfg.get("origin", (0.0, 0.0))
        if cfg.get("image_points") and cfg.get("ground_points"):
            return GroundPlaneHomographyRanger.from_points(
                cfg["image_points"], cfg["ground_points"], origin=origin)
        if cfg.get("homography") is not None:
            return GroundPlaneHomographyRanger(cfg["homography"], origin=origin)
        raise ValueError("ground_plane homography mode requires image_points+ground_points "
                         "or a homography matrix")
    raise ValueError(f"unknown ground_plane mode '{mode}'; use 'pinhole' or 'homography'")
