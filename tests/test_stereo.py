"""Tests for the stereo subsystem (pure math + factory; ONNX session injected)."""

import numpy as np
import pytest

from src.stereo import (
    ESMStereo,
    OnnxStereoMatcher,
    StereoRig,
    build_stereo_matcher,
    disparity_to_depth,
    disparity_to_depth_map,
)
from src.stereo.onnx_matcher import postprocess_disparity, preprocess_pair


class TestDisparityMath:
    def test_disparity_to_depth(self):
        # fx=1000 px, baseline=0.1 m, disparity=50 px -> Z = 1000*0.1/50 = 2.0 m
        assert disparity_to_depth(50, 1000, 0.1) == pytest.approx(2.0)

    def test_zero_disparity_is_inf(self):
        assert disparity_to_depth(0, 1000, 0.1) == float("inf")

    def test_depth_map_marks_invalid_inf(self):
        disp = np.array([[50.0, 0.0], [25.0, -1.0]], dtype=np.float32)
        depth = disparity_to_depth_map(disp, 1000, 0.1)
        assert depth[0, 0] == pytest.approx(2.0)
        assert depth[1, 0] == pytest.approx(4.0)
        assert np.isinf(depth[0, 1]) and np.isinf(depth[1, 1])  # invalid disparities


class TestRig:
    def test_from_config_requires_fx_and_baseline(self):
        assert StereoRig.from_config({"fx": 1000}) is None
        assert StereoRig.from_config({"baseline": 0.1}) is None
        rig = StereoRig.from_config({"fx": 1000, "baseline": 0.12})
        assert rig is not None and rig.fx == 1000 and rig.baseline == 0.12


class TestOnnxScaffold:
    def test_preprocess_pair_shape(self):
        t = preprocess_pair(np.zeros((100, 120, 3), np.uint8),
                            np.zeros((100, 120, 3), np.uint8), (256, 192))
        assert t.shape == (2, 3, 192, 256) and t.dtype == np.float32

    def test_postprocess_resizes_and_rescales(self):
        raw = np.full((10, 20), 5.0, dtype=np.float32)   # 5 px disparity at network width 20
        out = postprocess_disparity(raw, (10, 40), input_w=20)
        assert out.shape == (10, 40)
        assert out[0, 0] == pytest.approx(10.0)          # 5 * (40/20)

    def test_build_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            build_stereo_matcher({"backend": "bogus", "model_path": "m.onnx"})

    def test_requires_model_or_session(self):
        with pytest.raises(ValueError):
            build_stereo_matcher({"backend": "esmstereo"})  # no model_path, no session

    def test_compute_disparity_with_injected_session(self):
        class _Sess:
            def get_inputs(self):
                import types
                return [types.SimpleNamespace(name="input")]

            def run(self, _o, _f):
                return [np.ones((1, 1, 48, 64), dtype=np.float32)]

        m = ESMStereo(input_size=(64, 48), session=_Sess())
        disp = m.compute_disparity(np.zeros((20, 30, 3), np.uint8), np.zeros((20, 30, 3), np.uint8))
        assert disp.shape == (20, 30)  # resized back to the left frame
        assert isinstance(m, OnnxStereoMatcher) and m.model_name == "esmstereo"
