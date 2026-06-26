"""DINOv3 frozen-backbone embedding service (lazy scaffold).

One frozen forward pass yields a general-purpose image/crop embedding usable for
re-ID, anomaly scoring, and retrieval. The backbone is loaded lazily via
``transformers`` (or injected for tests); ``l2_normalize`` and ``cosine_similarity``
are the pure, unit-tested core. Weights are not bundled (see requirements-phase2.txt).
"""

from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Unit-normalize a vector (safe for zero vectors)."""
    v = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / (norm + 1e-6)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / denom)


class DINOv3Backbone:
    """Frozen DINOv3 image embedder (lazy transformers model, or injected for tests)."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        device: str = "cpu",
        model=None,
        processor=None,
    ):
        self._model_name = model_name
        self._device = device
        if model is not None and processor is not None:
            self._model, self._processor = model, processor
        else:
            import torch  # noqa: F401  (lazy; ensures torch present for inference)
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(device).eval()

    def embed(self, frame: np.ndarray) -> np.ndarray:
        """Return an L2-normalized embedding for a BGR frame/crop."""
        import cv2
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state.mean(dim=1)
        return l2_normalize(feat[0].cpu().numpy())

    @property
    def model_name(self) -> str:
        return self._model_name
