"""Tests for the relative->metric depth scale calibrator (pure numpy)."""

import numpy as np
import pytest

from src.depth.calibration import DepthScaleCalibrator


def _samples(a, b, z):
    """Relative samples that exactly satisfy 1/Z = a*d_rel + b -> d_rel = (1/Z - b)/a."""
    z = np.asarray(z, dtype=np.float64)
    return (1.0 / z - b) / a


class TestFit:
    def test_recovers_known_params(self):
        a, b = 0.3, 0.05
        z = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        cal = DepthScaleCalibrator().fit(_samples(a, b, z), z)
        ra, rb = cal.params
        assert ra == pytest.approx(a, rel=1e-6)
        assert rb == pytest.approx(b, abs=1e-6)

    def test_round_trips_to_meters(self):
        a, b = 0.25, 0.1
        z = np.array([1.5, 3.0, 6.0, 12.0])
        d = _samples(a, b, z)
        cal = DepthScaleCalibrator().fit(d, z)
        for di, zi in zip(d, z):
            assert cal.to_meters(di) == pytest.approx(zi, rel=1e-4)

    def test_nearer_is_fewer_meters(self):
        a, b = 0.3, 0.05
        z = np.array([1.0, 10.0])
        cal = DepthScaleCalibrator().fit(_samples(a, b, z), z)
        near = cal.to_meters(_samples(a, b, np.array([1.0]))[0])   # larger d_rel
        far = cal.to_meters(_samples(a, b, np.array([10.0]))[0])   # smaller d_rel
        assert near < far  # larger relative depth = nearer = fewer meters

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            DepthScaleCalibrator().fit([0.5], [2.0])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            DepthScaleCalibrator().fit([0.5, 0.6], [2.0])

    def test_nonpositive_meters_raises(self):
        with pytest.raises(ValueError):
            DepthScaleCalibrator().fit([0.5, 0.6], [2.0, 0.0])


class TestApply:
    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            DepthScaleCalibrator().to_meters(0.5)

    def test_array_handles_nonpositive_denominator(self):
        cal = DepthScaleCalibrator(a=1.0, b=0.0)
        out = cal.to_meters_array(np.array([1.0, 0.0, -1.0]))
        assert out[0] == pytest.approx(1.0)
        assert np.isinf(out[1]) and np.isinf(out[2])  # denom <= 0 -> inf

    def test_from_config(self):
        assert DepthScaleCalibrator.from_config({}) is None
        assert DepthScaleCalibrator.from_config({"a": 0.3}) is None
        cal = DepthScaleCalibrator.from_config({"a": 0.3, "b": 0.05})
        assert cal is not None and cal.params == (0.3, 0.05)
