"""Reconstruction-based anomaly detector (MNAD-style), drop-in for AnomalyDetector.

An autoencoder learns to reconstruct the per-frame scene descriptor of "normal"
footage; reconstruction error is the anomaly score. This is the learned alternative
to the IsolationForest in ``anomaly_detector.py`` and conforms to the SAME interface
(``update`` / ``fit`` / ``check`` returning ``(score, is_anomalous)`` plus the
``is_learning`` / ``is_fitted`` / ``learning_progress`` properties), so it can be
swapped in wherever the IsolationForest detector is used.

``torch`` is imported lazily in ``fit`` so this module stays importable without it;
the learning-phase bookkeeping and the threshold maths are unit-tested torch-free.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.analytics.anomaly_detector import SceneDescriptor
from src.models.base import Detection


def reconstruction_threshold(errors: np.ndarray, contamination: float) -> float:
    """Anomaly cutoff: the (1 - contamination) quantile of training errors."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return float("inf")
    q = float(np.clip(1.0 - contamination, 0.0, 1.0) * 100.0)
    return float(np.percentile(errors, q))


class MNADAnomalyDetector:
    """Autoencoder reconstruction anomaly over scene-descriptor features."""

    def __init__(
        self,
        learning_frames: int = 500,
        contamination: float = 0.05,
        num_classes: int = 80,
        window_size: int = 10,
        hidden_dim: int = 32,
        epochs: int = 30,
    ):
        self._descriptor = SceneDescriptor(num_classes=num_classes)
        self._learning_frames = learning_frames
        self._contamination = contamination
        self._window_size = window_size
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._training: list[np.ndarray] = []
        self._model = None
        self._threshold: float | None = None
        self._scores: deque[float] = deque(maxlen=window_size)

    @property
    def is_learning(self) -> bool:
        return self._model is None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def learning_progress(self) -> float:
        return min(1.0, len(self._training) / max(1, self._learning_frames))

    def update(self, detections: list[Detection], frame_shape: tuple[int, int] = (720, 1280)) -> None:
        """Collect a normal-scene feature while learning; auto-fit when full."""
        if self._model is not None:
            return
        self._training.append(self._descriptor.describe(detections, frame_shape))
        if len(self._training) >= self._learning_frames:
            self.fit()

    def fit(self) -> None:
        """Train the autoencoder on collected features and set the threshold."""
        if len(self._training) < 10:
            raise ValueError("Need at least 10 training samples to fit.")
        import torch  # lazy heavy import
        from torch import nn

        x = torch.tensor(np.asarray(self._training, dtype=np.float32))
        dim = x.shape[1]
        model = nn.Sequential(
            nn.Linear(dim, self._hidden_dim), nn.ReLU(),
            nn.Linear(self._hidden_dim, dim),
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(self._epochs):
            opt.zero_grad()
            loss = loss_fn(model(x), x)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            errs = ((model(x) - x) ** 2).mean(dim=1).cpu().numpy()
        self._threshold = reconstruction_threshold(errs, self._contamination)
        self._model = model

    def check(
        self, detections: list[Detection], frame_shape: tuple[int, int] = (720, 1280)
    ) -> tuple[float, bool]:
        """Return (smoothed reconstruction error, is_anomalous). (0.0, False) while learning."""
        if self._model is None:
            return 0.0, False
        import torch

        feat = self._descriptor.describe(detections, frame_shape)
        with torch.no_grad():
            t = torch.tensor(feat[None, :].astype(np.float32))
            err = float(((self._model(t) - t) ** 2).mean().item())
        self._scores.append(err)
        avg = float(np.mean(self._scores))
        return avg, bool(self._threshold is not None and err > self._threshold)
