"""ONNX monocular depth estimators (Depth Anything V2 / MiDaS).

``onnxruntime`` is imported lazily inside ``OnnxDepthEstimator.__init__`` so this
module stays importable without it. ``preprocess`` / ``postprocess`` are pure
(cv2 + numpy) and unit-testable on their own.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.depth.base import BaseDepthEstimator

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(frame, input_size, mean=_IMAGENET_MEAN, std=_IMAGENET_STD) -> np.ndarray:
    """BGR frame -> normalized NCHW float32 tensor for an ONNX depth model."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (int(input_size[0]), int(input_size[1]))).astype(np.float32) / 255.0
    normed = (resized - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    chw = np.transpose(normed, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(chw, dtype=np.float32)


def postprocess(raw, out_hw) -> np.ndarray:
    """Model output -> HxW float32 depth map resized to (out_h, out_w)."""
    depth = np.asarray(raw, dtype=np.float32).squeeze()
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if depth.shape[:2] != (out_h, out_w):
        depth = cv2.resize(depth, (out_w, out_h))
    return depth.astype(np.float32)


class OnnxDepthEstimator(BaseDepthEstimator):
    """Generic ONNX depth estimator; subclasses set input size / normalization."""

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (518, 518),
        mean: tuple[float, float, float] = _IMAGENET_MEAN,
        std: tuple[float, float, float] = _IMAGENET_STD,
        name: str = "onnx-depth",
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort  # lazy heavy import

        self._session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size
        self._mean = mean
        self._std = std
        self._name = name

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        inp = preprocess(frame, self._input_size, self._mean, self._std)
        out = self._session.run(None, {self._input_name: inp})[0]
        return postprocess(out, frame.shape[:2])

    @property
    def model_name(self) -> str:
        return self._name


class DepthAnythingV2(OnnxDepthEstimator):
    """Depth Anything V2 (ONNX). Larger value = nearer (inverse depth)."""

    def __init__(self, model_path: str, input_size: tuple[int, int] = (518, 518), providers=None):
        super().__init__(
            model_path, input_size=input_size, name="depth_anything_v2", providers=providers
        )


class MidasONNX(OnnxDepthEstimator):
    """MiDaS-small (ONNX) fallback."""

    def __init__(self, model_path: str, input_size: tuple[int, int] = (256, 256), providers=None):
        super().__init__(
            model_path, input_size=input_size, name="midas_small", providers=providers
        )


def build_depth_estimator(cfg: dict) -> BaseDepthEstimator:
    """Build a depth estimator from a config dict.

    Args:
        cfg: ``{"backend": "depth_anything"|"midas", "model_path": str,
            "input_size": [w, h]}``.

    Raises:
        ValueError: on an unknown backend or a missing ``model_path``.
    """
    cfg = dict(cfg or {})
    backend = (cfg.get("backend") or "depth_anything").strip().lower()
    model_path = cfg.get("model_path") or ""
    if not model_path:
        raise ValueError("depth.model_path is required to build a depth estimator")
    kwargs = {}
    if cfg.get("input_size"):
        kwargs["input_size"] = tuple(cfg["input_size"])
    if backend in ("depth_anything", "depth-anything", "dav2"):
        return DepthAnythingV2(model_path, **kwargs)
    if backend == "midas":
        return MidasONNX(model_path, **kwargs)
    raise ValueError(
        f"Unknown depth backend '{backend}'. Use 'depth_anything' or 'midas'."
    )
