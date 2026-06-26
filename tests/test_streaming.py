"""Tests for temporal streaming-depth smoothing (pure; inner estimator faked)."""

import sys
import types

import numpy as np
import pytest

from src.depth.base import BaseDepthEstimator
from src.depth.onnx_estimator import build_depth_estimator
from src.depth.streaming import TemporalDepthEstimator, align_scale_to, blend_depth


class _FakeInner(BaseDepthEstimator):
    def __init__(self, maps, units="relative"):
        self._maps = [np.asarray(m, dtype=np.float32) for m in maps]
        self._i = 0
        self._units = units

    def estimate(self, frame):
        m = self._maps[min(self._i, len(self._maps) - 1)]
        self._i += 1
        return m

    @property
    def model_name(self):
        return "fake"

    @property
    def units(self):
        return self._units


class TestPure:
    def test_blend_none_returns_curr(self):
        curr = np.full((4, 4), 5.0, dtype=np.float32)
        assert np.array_equal(blend_depth(None, curr, 0.5), curr)

    def test_blend_ema(self):
        prev = np.full((2, 2), 10.0, dtype=np.float32)
        curr = np.full((2, 2), 20.0, dtype=np.float32)
        assert np.allclose(blend_depth(prev, curr, 0.5), 15.0)

    def test_blend_shape_mismatch_reseeds(self):
        prev = np.zeros((2, 2), dtype=np.float32)
        curr = np.ones((3, 3), dtype=np.float32)
        assert np.array_equal(blend_depth(prev, curr, 0.5), curr)

    def test_align_scale_matches_median(self):
        curr = np.array([5.0, 10.0, 15.0], dtype=np.float32)   # median 10
        ref = np.array([20.0, 20.0, 20.0], dtype=np.float32)   # median 20
        out = align_scale_to(curr, ref)
        assert np.allclose(out, [10.0, 20.0, 30.0])


class TestTemporalEstimator:
    def test_blends_across_frames(self):
        inner = _FakeInner([np.full((4, 4), 10.0), np.full((4, 4), 20.0)])
        est = TemporalDepthEstimator(inner, alpha=0.5)
        assert np.allclose(est.estimate(None), 10.0)   # first frame passes through
        assert np.allclose(est.estimate(None), 15.0)   # 0.5*20 + 0.5*10

    def test_reset_drops_state(self):
        inner = _FakeInner([np.full((2, 2), 10.0), np.full((2, 2), 20.0)])
        est = TemporalDepthEstimator(inner, alpha=0.5)
        est.estimate(None)
        est.reset()
        assert np.allclose(est.estimate(None), 20.0)  # next frame passes through again

    def test_units_delegated_and_name(self):
        est = TemporalDepthEstimator(_FakeInner([np.zeros((2, 2))], units="metric"), align_scale=False)
        assert est.units == "metric"
        assert est.model_name == "temporal(fake)"

    def test_align_scale_rejected_for_metric(self):
        with pytest.raises(ValueError):
            TemporalDepthEstimator(_FakeInner([np.zeros((2, 2))], units="metric"), align_scale=True)


def _install_fake_onnxruntime(monkeypatch):
    class _Sess:
        def __init__(self, path, providers=None):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="input")]

        def run(self, _o, _f):
            return [np.zeros((1, 1, 32, 32), dtype=np.float32)]

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = _Sess
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)


def test_factory_wraps_when_streaming_enabled(monkeypatch):
    _install_fake_onnxruntime(monkeypatch)
    est = build_depth_estimator({
        "backend": "depth_anything", "model_path": "m.onnx",
        "streaming": {"enabled": True, "alpha": 0.5},
    })
    assert isinstance(est, TemporalDepthEstimator)
    assert est.units == "relative"
    out = est.estimate(np.zeros((40, 40, 3), dtype=np.uint8))
    assert out.shape == (40, 40)
    est.reset()  # reset is available on the wrapped estimator


def test_factory_no_wrap_by_default(monkeypatch):
    _install_fake_onnxruntime(monkeypatch)
    est = build_depth_estimator({"backend": "depth_anything", "model_path": "m.onnx"})
    assert not isinstance(est, TemporalDepthEstimator)  # off by default
