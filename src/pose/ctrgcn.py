"""CTR-GCN skeleton-based action recognition (lazy scaffold).

Consumes a temporal sequence of 17-keypoint COCO skeletons (the YOLO-pose stream)
and classifies the action. The graph-convolutional model is loaded lazily (or
injected for tests); ``normalize_skeleton`` and ``SkeletonSequenceBuffer`` are pure
and unit-tested, and the classifier degrades to ``("unknown", 0.0)`` without a model
(so the rule-based ``ActionClassifier`` stays the default).

Weights are not bundled (see requirements-phase2.txt).
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


def normalize_skeleton(kpts) -> np.ndarray:
    """Translation/scale-invariant skeleton: centre on the hip midpoint, scale by
    the shoulder-hip (torso) length. Accepts (17, 2) or (17, 3); returns (17, 2)."""
    k = np.asarray(kpts, dtype=np.float32)
    xy = k[:, :2].copy()
    hip = (xy[11] + xy[12]) / 2.0
    shoulder = (xy[5] + xy[6]) / 2.0
    scale = float(np.linalg.norm(shoulder - hip)) + 1e-6
    return (xy - hip) / scale


class SkeletonSequenceBuffer:
    """Per-track ring buffer of normalized skeletons."""

    def __init__(self, length: int = 30):
        self._length = length
        self._buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=length))

    def update(self, track_id: int, kpts) -> None:
        self._buffers[track_id].append(normalize_skeleton(kpts))

    def is_ready(self, track_id: int) -> bool:
        return len(self._buffers[track_id]) >= self._length

    def sequence(self, track_id: int) -> np.ndarray | None:
        """(T, 17, 2) array once the buffer is full, else None."""
        if not self.is_ready(track_id):
            return None
        return np.stack(list(self._buffers[track_id]))

    def clear(self) -> None:
        self._buffers.clear()


class CTRGCNActionClassifier:
    """CTR-GCN action classifier over a skeleton sequence (lazy torch model)."""

    def __init__(
        self,
        model_path: str = "",
        labels: list[str] | None = None,
        sequence_length: int = 30,
        device: str = "",
        model=None,
    ):
        self._labels = labels or ["unknown"]
        self._buffer = SkeletonSequenceBuffer(sequence_length)
        self._device = device
        if model is not None:
            self._model = model
        elif model_path:
            import torch  # lazy heavy import

            self._model = torch.jit.load(model_path, map_location=device or "cpu")
            self._model.eval()
        else:
            self._model = None  # fallback: no learned model available

    def classify(self, track_id: int, kpts) -> tuple[str, float]:
        """Buffer the skeleton and classify once the sequence is full."""
        self._buffer.update(track_id, kpts)
        seq = self._buffer.sequence(track_id)
        if self._model is None or seq is None:
            return "unknown", 0.0
        return self._infer(seq)

    def _infer(self, seq: np.ndarray) -> tuple[str, float]:
        import torch
        from torch.nn import functional as F

        with torch.no_grad():
            t = torch.from_numpy(seq[None].astype(np.float32))  # (1, T, 17, 2)
            logits = self._model(t)
            probs = F.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            conf = float(probs[idx].item())
        label = self._labels[idx] if idx < len(self._labels) else str(idx)
        return label, conf
