"""ONNX stereo matcher (e.g. ESMStereo) — LAZY SCAFFOLD.

``onnxruntime`` is imported lazily inside ``OnnxStereoMatcher.__init__`` (or skipped
entirely when a session is injected for tests). ``preprocess_pair`` / ``postprocess_disparity``
are pure (cv2 + numpy) and unit-tested. Weights are not bundled (requirements-phase2.txt).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.stereo.base import BaseStereoMatcher

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_pair(left, right, input_size, mean=_IMAGENET_MEAN, std=_IMAGENET_STD) -> np.ndarray:
    """Two BGR frames -> stacked, normalized (2, 3, H, W) float32 tensor."""
    def _one(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (int(input_size[0]), int(input_size[1]))).astype(np.float32) / 255.0
        n = (r - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
        return np.transpose(n, (2, 0, 1))
    return np.ascontiguousarray(np.stack([_one(left), _one(right)]), dtype=np.float32)


def postprocess_disparity(raw: np.ndarray, out_hw, input_w: int) -> np.ndarray:
    """Network disparity -> HxW disparity at output resolution.

    Disparity is in *pixels at the network input width* (``input_w``), so after resizing to
    the output width it must be multiplied by ``out_w / input_w`` to stay in output pixels.
    """
    disp = np.asarray(raw, dtype=np.float32).squeeze()
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if disp.shape[:2] != (out_h, out_w):
        disp = cv2.resize(disp, (out_w, out_h))
    disp = disp * (float(out_w) / float(input_w))  # rescale pixel disparity to output width
    return disp.astype(np.float32)


class OnnxStereoMatcher(BaseStereoMatcher):
    """Generic ONNX stereo matcher; lazy onnxruntime or an injected session."""

    def __init__(self, model_path: str = "", input_size=(640, 480), name: str = "onnx-stereo",
                 session=None, providers=None):
        self._input_size = input_size
        self._name = name
        if session is not None:
            self._session = session
        elif model_path:
            import onnxruntime as ort  # lazy heavy import

            self._session = ort.InferenceSession(
                model_path, providers=providers or ["CPUExecutionProvider"])
        else:
            raise ValueError("OnnxStereoMatcher requires model_path or an injected session")
        self._input_name = self._session.get_inputs()[0].name

    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        inp = preprocess_pair(left, right, self._input_size)
        out = self._session.run(None, {self._input_name: inp})[0]
        return postprocess_disparity(out, left.shape[:2], int(self._input_size[0]))

    @property
    def model_name(self) -> str:
        return self._name


class ESMStereo(OnnxStereoMatcher):
    """ESMStereo (ONNX) real-time stereo backend."""

    def __init__(self, model_path: str = "", input_size=(640, 480), session=None, providers=None):
        super().__init__(model_path, input_size=input_size, name="esmstereo",
                         session=session, providers=providers)


def build_stereo_matcher(cfg: dict) -> BaseStereoMatcher:
    """Build a stereo matcher from a config dict (``{backend, model_path, input_size}``)."""
    cfg = dict(cfg or {})
    backend = (cfg.get("backend") or "esmstereo").strip().lower()
    kwargs = {}
    if cfg.get("input_size"):
        kwargs["input_size"] = tuple(cfg["input_size"])
    if backend in ("esmstereo", "esm"):
        return ESMStereo(cfg.get("model_path", ""), **kwargs)
    if backend in ("onnx", "onnx-stereo"):
        return OnnxStereoMatcher(cfg.get("model_path", ""), **kwargs)
    raise ValueError(f"Unknown stereo backend '{backend}'. Use 'esmstereo' or 'onnx'.")
