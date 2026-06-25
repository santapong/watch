"""Pluggable appearance embedders for re-identification.

Backends:
- ``HistogramEmbedder`` — zero-dependency 96-d colour-histogram descriptor (the
  CI-tested default; a verbatim port of the original tracker logic).
- ``OSNetEmbedder`` — deep 512-d OSNet embedding via ``torchreid`` (heavy deps are
  **lazy-imported** so this module stays importable without torch installed).

``build_embedder(backend)`` is the factory: ``"auto"`` uses OSNet when torch +
torchreid are importable and falls back to the histogram otherwise, so the slim,
torch-free test gate keeps working.

Import-safety: top-level imports are limited to ``abc``/``cv2``/``numpy`` — no torch
at module load (``tracker.py`` imports this module and is itself imported by the
torch-free CI suite).
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod

import cv2
import numpy as np

from src.models.base import Detection


class ReIDEmbedder(ABC):
    """Interface for appearance embedders used by the tracker for re-ID."""

    @abstractmethod
    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Return an appearance vector for the detection crop, or None if too small."""

    @staticmethod
    def _crop(frame: np.ndarray, detection: Detection, min_px: int = 10) -> np.ndarray | None:
        """Clamped bbox crop, or None when the crop is smaller than ``min_px``.

        Matches the clamping/min-size rule the tracker has always used.
        """
        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < min_px or y2 - y1 < min_px:
            return None
        return frame[y1:y2, x1:x2]


class HistogramEmbedder(ReIDEmbedder):
    """96-d (3 channels x 32 bins) normalised BGR colour histogram.

    Lightweight and illumination-sensitive, but zero-dependency — the default.
    """

    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        crop = self._crop(frame, detection)
        if crop is None:
            return None
        crop_resized = cv2.resize(crop, (64, 64))
        hist_features = []
        for ch in range(3):
            hist = cv2.calcHist([crop_resized], [ch], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.append(hist)
        return np.concatenate(hist_features)


class OSNetEmbedder(ReIDEmbedder):
    """Deep ~512-d L2-normalised OSNet embedding via ``torchreid``.

    ``torch``/``torchreid`` are imported lazily in ``__init__`` so importing this
    module never requires torch. ``torchreid.utils.FeatureExtractor`` handles the
    resize(256x128) -> ToTensor -> ImageNet-normalise pipeline and runs the model in
    eval/no-grad mode internally.

    Args:
        model_name: torchreid model id (default ``"osnet_x1_0"`` -> 512-d).
        model_path: path to a re-ID ``.pth.tar`` (Market-1501/MSMT17) for real
            accuracy; empty string loads ImageNet-pretrained weights.
        device: ``"cuda"``/``"cpu"`` (auto-detected when None).
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        model_path: str = "",
        device: str | None = None,
    ):
        import torch  # lazy: heavy dependency
        from torchreid.utils import FeatureExtractor  # lazy

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device,
        )

    def embed(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        crop = self._crop(frame, detection)
        if crop is None:
            return None
        # FeatureExtractor uses ToPILImage(), which assumes RGB; OpenCV frames are
        # BGR, so convert or the channels (and thus the embedding) are wrong.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        feats = self._extractor([rgb])  # torch.Tensor, shape (1, 512)
        vec = feats[0].cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / (norm + 1e-6)


def _torch_stack_available() -> bool:
    """True when both torch and torchreid are importable (no import side-effects)."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchreid") is not None
    )


def build_embedder(backend: str = "auto") -> ReIDEmbedder:
    """Construct an embedder by backend name.

    Args:
        backend: ``"auto"`` (OSNet if torch+torchreid present, else histogram;
            also falls back to histogram if OSNet construction fails, e.g. an
            offline weight download), ``"histogram"``, or ``"osnet"``.

    Returns:
        A :class:`ReIDEmbedder`.

    Raises:
        ValueError: on an unknown backend name.
    """
    backend = (backend or "auto").strip().lower()
    if backend == "histogram":
        return HistogramEmbedder()
    if backend == "osnet":
        return OSNetEmbedder()
    if backend == "auto":
        if _torch_stack_available():
            try:
                return OSNetEmbedder()
            except Exception:
                return HistogramEmbedder()  # e.g. weights unavailable / offline
        return HistogramEmbedder()
    raise ValueError(
        f"Unknown reid backend '{backend}'. Use 'auto', 'histogram', or 'osnet'."
    )
