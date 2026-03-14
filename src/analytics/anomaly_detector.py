"""Anomaly detection from normal scene patterns.

Learns "normal" scene characteristics from detection metadata and flags
deviations using Isolation Forest. No domain-specific training needed —
uses YOLO detections as input features.
"""

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.base import Detection


class SceneDescriptor:
    """Converts a list of detections into a fixed-size feature vector.

    Features include:
    - Total object count
    - Per-class object counts (top N classes)
    - Average/std of bounding box positions (center x, center y)
    - Average/std of bounding box sizes
    - Average confidence
    - Spatial distribution entropy
    """

    def __init__(self, num_classes: int = 80, grid_size: int = 4):
        """Initialize scene descriptor.

        Args:
            num_classes: Number of object classes to track counts for.
            grid_size: Grid divisions for spatial distribution (grid_size x grid_size).
        """
        self._num_classes = num_classes
        self._grid_size = grid_size

    def describe(
        self, detections: list[Detection], frame_shape: tuple[int, int] = (720, 1280)
    ) -> np.ndarray:
        """Convert detections to a feature vector.

        Args:
            detections: List of Detection objects.
            frame_shape: (height, width) of the frame.

        Returns:
            1D numpy feature vector.
        """
        h, w = frame_shape

        # Object count
        total_count = len(detections)

        # Per-class counts
        class_counts = np.zeros(self._num_classes, dtype=np.float32)
        for det in detections:
            if det.class_id < self._num_classes:
                class_counts[det.class_id] += 1

        if total_count == 0:
            # Return zero vector for empty scenes
            feature_dim = self._num_classes + 1 + 8 + self._grid_size**2
            features = np.zeros(feature_dim, dtype=np.float32)
            features[0] = 0  # count
            return features

        # Position statistics
        centers_x = np.array([d.center[0] / w for d in detections])
        centers_y = np.array([d.center[1] / h for d in detections])
        widths = np.array([d.width / w for d in detections])
        heights = np.array([d.height / h for d in detections])
        confidences = np.array([d.confidence for d in detections])

        position_stats = np.array(
            [
                centers_x.mean(),
                centers_x.std() if len(centers_x) > 1 else 0,
                centers_y.mean(),
                centers_y.std() if len(centers_y) > 1 else 0,
                widths.mean(),
                heights.mean(),
                confidences.mean(),
                confidences.std() if len(confidences) > 1 else 0,
            ],
            dtype=np.float32,
        )

        # Spatial distribution (grid histogram)
        grid_hist = np.zeros(self._grid_size**2, dtype=np.float32)
        for det in detections:
            gx = min(int(det.center[0] / w * self._grid_size), self._grid_size - 1)
            gy = min(int(det.center[1] / h * self._grid_size), self._grid_size - 1)
            grid_hist[gy * self._grid_size + gx] += 1
        if grid_hist.sum() > 0:
            grid_hist /= grid_hist.sum()

        # Combine all features
        features = np.concatenate(
            [
                [total_count],
                class_counts,
                position_stats,
                grid_hist,
            ]
        ).astype(np.float32)

        return features


class AnomalyDetector:
    """Detects anomalies in scenes by learning normal patterns.

    Two modes:
    - learn: Collect scene descriptors to build a baseline of "normal"
    - detect: Score new scenes against the baseline, flag anomalies

    Example:
        anomaly = AnomalyDetector(learning_frames=500)
        # Learning phase
        for frame_detections in learning_data:
            anomaly.update(frame_detections, frame.shape[:2])
        anomaly.fit()
        # Detection phase
        score, is_anomalous = anomaly.check(detections, frame.shape[:2])
    """

    def __init__(
        self,
        learning_frames: int = 500,
        contamination: float = 0.05,
        num_classes: int = 80,
        window_size: int = 10,
    ):
        """Initialize anomaly detector.

        Args:
            learning_frames: Number of frames to collect before fitting.
            contamination: Expected proportion of anomalies (0-0.5).
            num_classes: Number of object classes.
            window_size: Sliding window size for temporal smoothing.
        """
        self._descriptor = SceneDescriptor(num_classes=num_classes)
        self._learning_frames = learning_frames
        self._contamination = contamination
        self._window_size = window_size

        self._training_data: list[np.ndarray] = []
        self._model: IsolationForest | None = None
        self._is_fitted = False
        self._score_history: deque[float] = deque(maxlen=window_size)
        self._frame_count = 0

    @property
    def is_learning(self) -> bool:
        """Whether the detector is still in learning mode."""
        return not self._is_fitted

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def learning_progress(self) -> float:
        """Learning progress as a fraction (0.0 to 1.0)."""
        if self._is_fitted:
            return 1.0
        return min(len(self._training_data) / self._learning_frames, 1.0)

    def update(
        self, detections: list[Detection], frame_shape: tuple[int, int] = (720, 1280)
    ) -> None:
        """Add a frame's detections to the learning buffer.

        Args:
            detections: Detections from the current frame.
            frame_shape: (height, width) of the frame.
        """
        features = self._descriptor.describe(detections, frame_shape)
        self._training_data.append(features)
        self._frame_count += 1

        # Auto-fit when enough data is collected
        if len(self._training_data) >= self._learning_frames and not self._is_fitted:
            self.fit()

    def fit(self) -> None:
        """Fit the anomaly detection model on collected data."""
        if len(self._training_data) < 10:
            raise ValueError("Need at least 10 training samples to fit.")

        X = np.array(self._training_data, dtype=np.float32)
        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=100,
        )
        self._model.fit(X)
        self._is_fitted = True

    def check(
        self, detections: list[Detection], frame_shape: tuple[int, int] = (720, 1280)
    ) -> tuple[float, bool]:
        """Check if the current scene is anomalous.

        Args:
            detections: Detections from the current frame.
            frame_shape: (height, width) of the frame.

        Returns:
            Tuple of (anomaly_score, is_anomalous).
            Score is negative for anomalies, positive for normal.
        """
        if not self._is_fitted:
            # Still learning — collect data and return normal
            self.update(detections, frame_shape)
            return 0.0, False

        features = self._descriptor.describe(detections, frame_shape)
        score = float(self._model.score_samples(features.reshape(1, -1))[0])
        prediction = int(self._model.predict(features.reshape(1, -1))[0])

        self._score_history.append(score)

        # Temporal smoothing: use average score over window
        avg_score = np.mean(list(self._score_history))
        is_anomalous = prediction == -1

        return avg_score, is_anomalous

    def save(self, path: str) -> None:
        """Save the trained model and training data."""
        import pickle

        save_data = {
            "model": self._model,
            "training_data": self._training_data,
            "is_fitted": self._is_fitted,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path: str) -> None:
        """Load a previously trained model."""
        import pickle

        with open(path, "rb") as f:
            save_data = pickle.load(f)

        self._model = save_data["model"]
        self._training_data = save_data["training_data"]
        self._is_fitted = save_data["is_fitted"]
