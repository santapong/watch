"""Promptable segmentation interface.

A segmenter takes a frame plus existing detections (as box prompts) and populates
each ``Detection.mask`` with a pixel mask. Concrete models (SAM 2) lazy-import their
heavy runtime in ``sam2_wrapper.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models.base import Detection


class BaseSegmenter(ABC):
    """Interface for box-prompted segmenters."""

    @abstractmethod
    def segment(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Populate ``detection.mask`` for each detection (in place) and return them."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the loaded segmentation model."""
