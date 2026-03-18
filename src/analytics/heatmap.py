"""Heatmap generation from detection and tracking data.

Generates density heatmaps showing where objects/people concentrate
most frequently over time. Supports real-time accumulation and
periodic snapshot export.
"""

from pathlib import Path

import cv2
import numpy as np

from src.models.base import Detection


class HeatmapGenerator:
    """Generates occupancy heatmaps from detection data over time.

    Accumulates object center positions into a density map and renders
    it as a color overlay on the camera frame.

    Example:
        heatmap = HeatmapGenerator(frame_shape=(720, 1280))
        for detections in stream:
            heatmap.update(detections)
            overlay = heatmap.render(frame)
            cv2.imshow("Heatmap", overlay)
    """

    def __init__(
        self,
        frame_shape: tuple[int, int] = (720, 1280),
        decay: float = 0.995,
        radius: int = 40,
        class_filter: list[str] | None = None,
    ):
        """Initialize heatmap generator.

        Args:
            frame_shape: (height, width) of the video frame.
            decay: Exponential decay factor per frame (0-1). Higher = longer memory.
            radius: Gaussian radius for each detection point.
            class_filter: Only count these classes (None = all classes).
        """
        self._h, self._w = frame_shape
        self._decay = decay
        self._radius = radius
        self._class_filter = set(class_filter) if class_filter else None
        self._accumulator = np.zeros((self._h, self._w), dtype=np.float64)
        self._frame_count = 0

    def update(self, detections: list[Detection]) -> None:
        """Accumulate detections into the heatmap.

        Args:
            detections: Detections from the current frame.
        """
        # Apply decay to existing accumulation
        self._accumulator *= self._decay

        for det in detections:
            if self._class_filter and det.class_name not in self._class_filter:
                continue

            cx, cy = det.center
            cx, cy = int(cx), int(cy)

            if 0 <= cx < self._w and 0 <= cy < self._h:
                # Add a Gaussian blob at the detection center
                y_start = max(0, cy - self._radius)
                y_end = min(self._h, cy + self._radius)
                x_start = max(0, cx - self._radius)
                x_end = min(self._w, cx + self._radius)

                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                        if dist_sq < self._radius ** 2:
                            weight = np.exp(-dist_sq / (2 * (self._radius / 3) ** 2))
                            self._accumulator[y, x] += weight

        self._frame_count += 1

    def render(
        self,
        frame: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Render heatmap overlay on frame.

        Args:
            frame: BGR frame to overlay on.
            alpha: Blend factor (0 = frame only, 1 = heatmap only).
            colormap: OpenCV colormap constant.

        Returns:
            Frame with heatmap overlay.
        """
        if self._accumulator.max() > 0:
            normalized = (self._accumulator / self._accumulator.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros((self._h, self._w), dtype=np.uint8)

        heatmap_colored = cv2.applyColorMap(normalized, colormap)

        # Resize if frame shape differs
        fh, fw = frame.shape[:2]
        if (fh, fw) != (self._h, self._w):
            heatmap_colored = cv2.resize(heatmap_colored, (fw, fh))

        # Only overlay where there's actual heat
        mask = normalized > 5
        if mask.ndim == 2 and (fh, fw) != (self._h, self._w):
            mask = cv2.resize(mask.astype(np.uint8), (fw, fh)).astype(bool)

        result = frame.copy()
        if mask.any():
            result[mask] = cv2.addWeighted(
                frame[mask], 1 - alpha, heatmap_colored[mask], alpha, 0
            )

        return result

    def get_raw(self) -> np.ndarray:
        """Get the raw accumulator array."""
        return self._accumulator.copy()

    def save_snapshot(self, path: str, colormap: int = cv2.COLORMAP_JET) -> None:
        """Save the current heatmap as an image file.

        Args:
            path: Output file path (e.g., "heatmap.png").
            colormap: OpenCV colormap constant.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if self._accumulator.max() > 0:
            normalized = (self._accumulator / self._accumulator.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros((self._h, self._w), dtype=np.uint8)

        colored = cv2.applyColorMap(normalized, colormap)
        cv2.imwrite(path, colored)

    def reset(self) -> None:
        """Reset the heatmap accumulator."""
        self._accumulator.fill(0)
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count
