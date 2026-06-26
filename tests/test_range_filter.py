"""Tests for per-track distance Kalman smoothing (pure numpy)."""

import numpy as np
import pytest

from src.tracking.range_filter import RangeKalman1D, RangeTracker


class TestRangeKalman1D:
    def test_constant_distance_converges(self):
        kf = RangeKalman1D(5.0, process_var=0.1, meas_var=1.0)
        for _ in range(30):
            kf.predict(1.0)
            kf.update(5.0)
        assert kf.value == pytest.approx(5.0, abs=0.3)
        assert kf.rate == pytest.approx(0.0, abs=0.1)

    def test_constant_approach_estimates_rate(self):
        kf = RangeKalman1D(20.0, process_var=0.5, meas_var=0.5)
        z = 20.0
        for _ in range(25):
            z -= 1.0  # approaching 1 unit / step
            kf.predict(1.0)
            kf.update(z)
        assert kf.rate == pytest.approx(-1.0, abs=0.2)
        assert kf.value == pytest.approx(z, abs=0.7)

    def test_smooths_noise(self):
        kf = RangeKalman1D(5.0, process_var=0.1, meas_var=1.0)
        raw = [5.0 + (1.0 if i % 2 else -1.0) for i in range(30)]  # alternating +/-1
        smoothed = []
        for z in raw:
            kf.predict(1.0)
            smoothed.append(kf.update(z))
        # second-half smoothed variance is well below the raw variance
        assert np.var(smoothed[10:]) < np.var(raw[10:])


class TestRangeTracker:
    def test_creates_and_updates_per_track(self):
        rt = RangeTracker(process_var=0.5, meas_var=0.5)
        v0, r0 = rt.update(1, 10.0)
        assert v0 == pytest.approx(10.0) and r0 == pytest.approx(0.0)  # first obs
        rt.update(1, 9.0)
        rt.update(1, 8.0)
        v, r = rt.get(1)
        assert r < 0  # approaching
        assert len(rt) == 1

    def test_predict_only_advances_through_gap(self):
        rt = RangeTracker(process_var=0.5, meas_var=0.5)
        for z in (20.0, 19.0, 18.0, 17.0):
            rt.update(1, z)
        before, rate = rt.get(1)
        after, _ = rt.predict_only(1, dt=1.0)
        assert after < before  # keeps closing during the gap
        assert rt.predict_only(999) is None  # unknown track

    def test_independent_tracks_and_drop(self):
        rt = RangeTracker()
        rt.update(1, 5.0)
        rt.update(2, 50.0)
        assert rt.get(1)[0] == pytest.approx(5.0)
        assert rt.get(2)[0] == pytest.approx(50.0)
        rt.drop(1)
        assert rt.get(1) is None and len(rt) == 1
