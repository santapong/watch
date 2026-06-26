"""Tests for the monocular depth subsystem (pure core; ONNX session is mocked)."""

import sys
import types

import numpy as np
import pytest

from src.depth.base import (
    annotate_depth,
    is_too_close,
    percentile_normalize,
    prepare_depth_map,
    sample_depth,
)
from src.depth.onnx_estimator import (
    DepthAnythingV2,
    DepthAnythingV2Metric,
    MidasONNX,
    build_depth_estimator,
    postprocess,
    preprocess,
)
from src.models.base import Detection


def _det(x1, y1, x2, y2):
    return Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                     confidence=0.9, class_id=0, class_name="person")


class TestSampling:
    def test_median_over_uniform_region(self):
        dm = np.full((100, 100), 5.0, dtype=np.float32)
        assert sample_depth(dm, (10, 10, 50, 50), shrink=0.0) == pytest.approx(5.0)

    def test_robust_to_outlier_minority(self):
        dm = np.full((100, 100), 5.0, dtype=np.float32)
        dm[0:10, :] = 999.0  # 10% outliers
        assert sample_depth(dm, (0, 0, 100, 100), shrink=0.0) == pytest.approx(5.0)

    def test_shrink_focuses_on_object_centre(self):
        dm = np.ones((100, 100), dtype=np.float32)  # background = 1.0
        dm[25:75, 25:75] = 5.0                       # object in the centre
        full = sample_depth(dm, (0, 0, 100, 100), shrink=0.0, use_mad=False)
        focused = sample_depth(dm, (0, 0, 100, 100), shrink=0.6, use_mad=False)
        assert full == pytest.approx(1.0)   # background dominates the whole box
        assert focused == pytest.approx(5.0)  # shrink isolates the object

    def test_empty_region_returns_none(self):
        dm = np.zeros((10, 10), dtype=np.float32)
        assert sample_depth(dm, (5, 5, 5, 5), shrink=0.5) is None

    def test_clips_to_frame_bounds(self):
        dm = np.arange(100, dtype=np.float32).reshape(10, 10)
        assert sample_depth(dm, (-20, -20, 5, 5), shrink=0.0) is not None


class TestNormalize:
    def test_range(self):
        n = percentile_normalize(np.arange(100, dtype=np.float32).reshape(10, 10))
        assert n.shape == (10, 10)
        assert n.min() >= 0.0 and n.max() <= 1.0

    def test_degenerate_is_zero(self):
        assert np.all(percentile_normalize(np.full((10, 10), 7.0, dtype=np.float32)) == 0.0)


class TestAnnotate:
    def test_sets_depth_in_place(self):
        dm = np.full((100, 100), 0.7, dtype=np.float32)
        dets = [_det(10, 10, 50, 50), _det(60, 60, 90, 90)]
        annotate_depth(dets, dm, shrink=0.0)
        assert all(d.depth == pytest.approx(0.7) for d in dets)

    def test_detection_depth_defaults_none(self):
        assert _det(0, 0, 10, 10).depth is None


class TestProximityConvention:
    def test_relative_larger_is_nearer(self):
        assert is_too_close(0.9, 0.8, "relative") is True
        assert is_too_close(0.7, 0.8, "relative") is False

    def test_metric_smaller_is_nearer(self):
        assert is_too_close(1.5, 2.0, "metric") is True   # 1.5 m -> too close
        assert is_too_close(2.5, 2.0, "metric") is False  # 2.5 m -> safe

    def test_none_is_never_close(self):
        assert is_too_close(None, 0.8, "relative") is False
        assert is_too_close(None, 2.0, "metric") is False

    def test_default_units_relative(self):
        assert is_too_close(0.9, 0.8) is True

    def test_unknown_units_raises(self):
        with pytest.raises(ValueError):
            is_too_close(0.5, 0.8, "meters")  # typo'd unit must not silently pass


class TestPrepareDepthMap:
    def test_metric_passthrough_unchanged(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        out = prepare_depth_map(raw, "metric")
        assert np.array_equal(out, raw)  # meters left as-is

    def test_relative_is_normalized(self):
        raw = np.arange(100, dtype=np.float32).reshape(10, 10)
        out = prepare_depth_map(raw, "relative")
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_unknown_units_raises(self):
        with pytest.raises(ValueError):
            prepare_depth_map(np.zeros((4, 4), dtype=np.float32), "meters")


class TestAnnotateUnits:
    def test_stamps_units(self):
        dm = np.full((50, 50), 3.0, dtype=np.float32)
        dets = [_det(10, 10, 40, 40)]
        annotate_depth(dets, dm, shrink=0.0, units="metric")
        assert dets[0].depth == pytest.approx(3.0)
        assert dets[0].depth_units == "metric"

    def test_no_units_leaves_default_none(self):
        dm = np.full((50, 50), 3.0, dtype=np.float32)
        d = _det(10, 10, 40, 40)
        annotate_depth([d], dm, shrink=0.0)
        assert d.depth_units is None


class TestOnnxPrePost:
    def test_preprocess_shape_and_dtype(self):
        t = preprocess(np.zeros((480, 640, 3), dtype=np.uint8), (518, 518))
        assert t.shape == (1, 3, 518, 518)
        assert t.dtype == np.float32

    def test_postprocess_resizes_to_frame(self):
        out = postprocess(np.zeros((1, 1, 100, 120), dtype=np.float32), (480, 640))
        assert out.shape == (480, 640)


def _install_fake_onnxruntime(monkeypatch, out_shape=(1, 1, 64, 64)):
    class _Sess:
        def __init__(self, path, providers=None):
            self.path = path

        def get_inputs(self):
            return [types.SimpleNamespace(name="input")]

        def run(self, _outputs, _feeds):
            return [np.zeros(out_shape, dtype=np.float32)]

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = _Sess
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)


class TestFactory:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            build_depth_estimator({"backend": "bogus", "model_path": "x.onnx"})

    def test_missing_model_path_raises(self):
        with pytest.raises(ValueError):
            build_depth_estimator({"backend": "depth_anything"})

    def test_builds_depth_anything(self, monkeypatch):
        _install_fake_onnxruntime(monkeypatch)
        est = build_depth_estimator({"backend": "depth_anything", "model_path": "m.onnx"})
        assert isinstance(est, DepthAnythingV2)
        assert est.model_name == "depth_anything_v2"

    def test_builds_midas(self, monkeypatch):
        _install_fake_onnxruntime(monkeypatch)
        est = build_depth_estimator({"backend": "midas", "model_path": "m.onnx"})
        assert isinstance(est, MidasONNX)

    def test_relative_backend_units_default(self, monkeypatch):
        _install_fake_onnxruntime(monkeypatch)
        est = build_depth_estimator({"backend": "depth_anything", "model_path": "m.onnx"})
        assert est.units == "relative"

    def test_builds_metric_backend(self, monkeypatch):
        _install_fake_onnxruntime(monkeypatch)
        est = build_depth_estimator({"backend": "depth_anything_metric", "model_path": "m.onnx"})
        assert isinstance(est, DepthAnythingV2Metric)
        assert est.units == "metric"
        assert est.model_name == "depth_anything_v2_metric"

    def test_input_size_passthrough(self, monkeypatch):
        _install_fake_onnxruntime(monkeypatch)
        est = build_depth_estimator(
            {"backend": "depth_anything", "model_path": "m.onnx", "input_size": [320, 320]}
        )
        assert est._input_size == (320, 320)  # the edge FPS/accuracy knob is honored

    def test_estimate_resizes_to_frame(self, monkeypatch):
        _install_fake_onnxruntime(monkeypatch, out_shape=(1, 1, 64, 64))
        est = build_depth_estimator({"backend": "depth_anything", "model_path": "m.onnx"})
        depth = est.estimate(np.zeros((120, 160, 3), dtype=np.uint8))
        assert depth.shape == (120, 160)
