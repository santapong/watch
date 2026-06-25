"""P2PNet point-based crowd counting wrapper.

P2PNet predicts one point per head, which counts dense crowds where YOLO boxes merge.
The points convert to tiny ``Detection`` objects so they flow into the existing
heatmap and zone-counter seams unchanged.

The torch model is loaded lazily (or injected for tests); ``predict_points`` does the
forward pass, while ``filter_points`` / ``points_to_detections`` / ``count`` are pure
and unit-tested. P2PNet weights are not bundled (see requirements-phase2.txt).
"""

from __future__ import annotations

import numpy as np

from src.models.base import Detection


def filter_points(points, scores, threshold: float = 0.5) -> list[tuple[float, float]]:
    """Keep (x, y) head points whose score is >= threshold."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    scr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if pts.shape[0] == 0:
        return []
    keep = scr >= threshold
    return [(float(x), float(y)) for x, y in pts[keep]]


def points_to_detections(points, box: int = 8, class_name: str = "head") -> list[Detection]:
    """Convert head points to small square Detections (for heatmap/zone reuse)."""
    half = box / 2.0
    return [
        Detection(
            bbox=(x - half, y - half, x + half, y + half),
            confidence=1.0, class_id=0, class_name=class_name,
        )
        for (x, y) in points
    ]


class P2PNetCounter:
    """Point-based crowd counter (lazy torch model, or an injected model for tests)."""

    def __init__(self, model_path: str = "", device: str = "", threshold: float = 0.5, model=None):
        self._threshold = threshold
        self._device = device
        if model is not None:
            self._model = model
            self._model_name = "injected"
        elif model_path:
            import torch  # lazy heavy import

            self._model = torch.jit.load(model_path, map_location=device or "cpu")
            self._model.eval()
            self._model_name = model_path
        else:
            raise ValueError("P2PNetCounter requires model_path or an injected model")

    def predict_points(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the model and return (coords (N,2), scores (N,)). Lazy torch forward."""
        import torch

        with torch.no_grad():
            t = torch.from_numpy(frame.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
            coords, scores = self._model(t)
        return np.asarray(coords).reshape(-1, 2), np.asarray(scores).reshape(-1)

    def count(self, frame: np.ndarray) -> list[tuple[float, float]]:
        """Return accepted head points for the frame."""
        coords, scores = self.predict_points(frame)
        return filter_points(coords, scores, self._threshold)

    def count_only(self, frame: np.ndarray) -> int:
        return len(self.count(frame))

    @property
    def model_name(self) -> str:
        return self._model_name
