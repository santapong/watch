"""Per-track distance smoothing via a 1D constant-velocity Kalman filter (pure numpy).

Raw per-frame depth is jittery and drops out when a detection is missed. ``RangeKalman1D``
smooths a single object's distance and estimates its range-rate (closing speed, dZ/dt);
``RangeTracker`` keeps one filter per ``track_id`` and can predict through gaps. Distance is
in whatever unit you feed it — meters if the depth backend is metric/calibrated, otherwise
relative units. No heavy dependencies.
"""

from __future__ import annotations

import numpy as np


class RangeKalman1D:
    """Constant-velocity Kalman filter on a single object's distance.

    State is ``[distance, rate]``; the measurement is the distance only.
    """

    def __init__(self, z0: float, process_var: float = 1.0, meas_var: float = 1.0,
                 initial_var: float = 10.0):
        self.x = np.array([float(z0), 0.0], dtype=np.float64)
        self.P = np.diag([float(initial_var), float(initial_var)])
        self._q = float(process_var)
        self._r = float(meas_var)

    def predict(self, dt: float = 1.0) -> float:
        dt = float(dt)
        f_mat = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = f_mat @ self.x
        q_mat = self._q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                                    [dt ** 2 / 2.0, dt]])
        self.P = f_mat @ self.P @ f_mat.T + q_mat
        return self.value

    def update(self, z: float) -> float:
        h_mat = np.array([[1.0, 0.0]])
        y = float(z) - (h_mat @ self.x)[0]
        s = (h_mat @ self.P @ h_mat.T)[0, 0] + self._r
        k = (self.P @ h_mat.T)[:, 0] / s
        self.x = self.x + k * y
        self.P = (np.eye(2) - np.outer(k, h_mat[0])) @ self.P
        return self.value

    @property
    def value(self) -> float:
        """Smoothed distance estimate."""
        return float(self.x[0])

    @property
    def rate(self) -> float:
        """Estimated range-rate (distance change per unit time; negative = approaching)."""
        return float(self.x[1])


class RangeTracker:
    """Manage one :class:`RangeKalman1D` per ``track_id``."""

    def __init__(self, process_var: float = 1.0, meas_var: float = 1.0):
        self._q = process_var
        self._r = meas_var
        self._filters: dict[int, RangeKalman1D] = {}

    def update(self, track_id: int, z: float, dt: float = 1.0) -> tuple[float, float]:
        """Feed a new distance for ``track_id``; returns (smoothed_distance, rate)."""
        f = self._filters.get(track_id)
        if f is None:
            f = RangeKalman1D(z, self._q, self._r)
            self._filters[track_id] = f
        else:
            f.predict(dt)
            f.update(z)
        return f.value, f.rate

    def predict_only(self, track_id: int, dt: float = 1.0) -> tuple[float, float] | None:
        """Advance a track through a detection gap (no measurement). None if unknown."""
        f = self._filters.get(track_id)
        if f is None:
            return None
        f.predict(dt)
        return f.value, f.rate

    def get(self, track_id: int) -> tuple[float, float] | None:
        f = self._filters.get(track_id)
        return (f.value, f.rate) if f is not None else None

    def drop(self, track_id: int) -> None:
        self._filters.pop(track_id, None)

    def __len__(self) -> int:
        return len(self._filters)
