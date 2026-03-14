"""Abstract base class for all object detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single detected object."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = None
    track_id: int | None = None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


class BaseDetector(ABC):
    """Abstract base class that all model wrappers must implement."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects.
        """

    @abstractmethod
    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        """Run detection + tracking on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects with track_id populated.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the loaded model."""

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """Return mapping of class_id -> class_name."""
