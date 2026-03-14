"""Drawing utilities for visualizing detections on frames."""

import cv2
import numpy as np

from src.models.base import Detection

# Color palette for different classes (BGR format)
COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),  # spring green
    (255, 128, 0),  # light blue
    (0, 128, 255),  # orange
    (128, 0, 255),  # purple
]


def get_color(class_id: int) -> tuple[int, int, int]:
    """Get a consistent color for a class ID."""
    return COLORS[class_id % len(COLORS)]


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    show_confidence: bool = True,
    show_track_id: bool = True,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes and labels on frame.

    Args:
        frame: BGR image to draw on (modified in place).
        detections: List of Detection objects.
        show_confidence: Whether to show confidence percentage.
        show_track_id: Whether to show track ID.
        thickness: Line thickness for bounding boxes.
        font_scale: Font scale for labels.

    Returns:
        The annotated frame (same reference as input).
    """
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = get_color(det.class_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        label_parts = [det.class_name]
        if show_track_id and det.track_id is not None:
            label_parts.insert(0, f"#{det.track_id}")
        if show_confidence:
            label_parts.append(f"{det.confidence:.0%}")
        label = " ".join(label_parts)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Draw label text
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on top-left of frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


def draw_info(frame: np.ndarray, model_name: str, num_detections: int) -> np.ndarray:
    """Draw model info on top-right of frame."""
    h, w = frame.shape[:2]
    text = f"{model_name} | {num_detections} objects"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(
        frame, text, (w - tw - 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return frame
