"""Privacy mode for automatic face and person blurring.

Provides GDPR/CCPA-compliant privacy features by blurring or
pixelating detected faces and/or persons in video frames.
"""

import cv2
import numpy as np

from src.models.base import Detection


class PrivacyFilter:
    """Applies privacy filters (blur/pixelate) to detected persons and faces.

    Supports three modes:
    - "blur": Gaussian blur on detected regions
    - "pixelate": Mosaic pixelation effect
    - "blackout": Solid color overlay

    Example:
        privacy = PrivacyFilter(mode="blur", target="person")
        for frame, detections in stream:
            frame = privacy.apply(frame, detections)
    """

    PERSON_CLASSES = {"person"}
    FACE_CLASSES = {"face"}

    def __init__(
        self,
        mode: str = "blur",
        target: str = "person",
        blur_strength: int = 51,
        pixel_size: int = 15,
    ):
        """Initialize privacy filter.

        Args:
            mode: "blur", "pixelate", or "blackout".
            target: "person" (full body), "face" (faces only), or "all" (both).
            blur_strength: Gaussian blur kernel size (must be odd).
            pixel_size: Mosaic block size for pixelation.
        """
        if mode not in ("blur", "pixelate", "blackout"):
            raise ValueError(f"Invalid mode: {mode}. Use 'blur', 'pixelate', or 'blackout'.")
        if target not in ("person", "face", "all"):
            raise ValueError(f"Invalid target: {target}. Use 'person', 'face', or 'all'.")

        self._mode = mode
        self._target = target
        self._blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        self._pixel_size = pixel_size

    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Apply privacy filter to detected regions.

        Args:
            frame: BGR image to process.
            detections: List of Detection objects.

        Returns:
            Frame with privacy filter applied to matching detections.
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        for det in detections:
            if not self._should_filter(det):
                continue

            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            roi = result[y1:y2, x1:x2]
            filtered = self._filter_roi(roi)

            # If the detection carries a segmentation mask, filter only the masked
            # pixels (pixel-accurate privacy); otherwise filter the whole bbox.
            mask = self._roi_mask(det.mask, x1, y1, x2, y2, (h, w))
            if mask is not None:
                roi[mask] = filtered[mask]
            else:
                result[y1:y2, x1:x2] = filtered

        return result

    def _filter_roi(self, roi: np.ndarray) -> np.ndarray:
        """Return the privacy-filtered version of a region (mode-dependent)."""
        if self._mode == "blur":
            return cv2.GaussianBlur(roi, (self._blur_strength, self._blur_strength), 0)
        if self._mode == "pixelate":
            return self._pixelate(roi)
        return np.zeros_like(roi)  # blackout

    @staticmethod
    def _roi_mask(mask, x1, y1, x2, y2, frame_hw) -> np.ndarray | None:
        """Boolean (roi_h, roi_w) mask for the bbox region, or None if no mask.

        Accepts a full-frame mask (cropped to the bbox) or a bbox-sized mask
        (resized to the clamped ROI). Values are thresholded at 0.5.
        """
        if mask is None:
            return None
        m = np.asarray(mask)
        if m.ndim > 2:
            m = m.squeeze()
        if m.ndim != 2:
            return None
        rh, rw = y2 - y1, x2 - x1
        fh, fw = frame_hw
        if m.shape[:2] == (fh, fw):
            m = m[y1:y2, x1:x2]
        elif m.shape[:2] != (rh, rw):
            m = cv2.resize(m.astype(np.float32), (rw, rh))
        return np.asarray(m) > 0.5

    def _should_filter(self, detection: Detection) -> bool:
        """Check if a detection should be filtered."""
        name = detection.class_name.lower()
        if self._target == "person":
            return name in self.PERSON_CLASSES
        elif self._target == "face":
            return name in self.FACE_CLASSES
        else:  # "all"
            return name in self.PERSON_CLASSES or name in self.FACE_CLASSES

    def _pixelate(self, roi: np.ndarray) -> np.ndarray:
        """Apply mosaic pixelation to a region."""
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // self._pixel_size), max(1, h // self._pixel_size)),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("blur", "pixelate", "blackout"):
            raise ValueError(f"Invalid mode: {value}")
        self._mode = value

    @property
    def target(self) -> str:
        return self._target

    @target.setter
    def target(self, value: str) -> None:
        if value not in ("person", "face", "all"):
            raise ValueError(f"Invalid target: {value}")
        self._target = value
