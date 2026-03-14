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


def draw_tracks(
    frame: np.ndarray,
    trajectories: dict[int, list[tuple[float, float]]],
    thickness: int = 2,
    max_trail: int = 30,
) -> np.ndarray:
    """Draw trajectory trails for tracked objects.

    Args:
        frame: BGR image to draw on.
        trajectories: Dict of track_id -> list of (x, y) positions.
        thickness: Line thickness.
        max_trail: Maximum trail length to draw.

    Returns:
        Annotated frame.
    """
    for track_id, positions in trajectories.items():
        if len(positions) < 2:
            continue

        color = COLORS[track_id % len(COLORS)]
        trail = positions[-max_trail:]

        for i in range(1, len(trail)):
            # Fade trail (more recent = more opaque)
            alpha = i / len(trail)
            pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            line_thickness = max(1, int(thickness * alpha))
            cv2.line(frame, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    return frame


def draw_zones(
    frame: np.ndarray,
    zones: dict[str, np.ndarray],
    counts: dict[str, int] | None = None,
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw polygon zones with optional counts.

    Args:
        frame: BGR image to draw on.
        zones: Dict of zone_name -> polygon vertices (N, 2).
        counts: Optional dict of zone_name -> object count.
        alpha: Transparency for zone overlay.

    Returns:
        Annotated frame.
    """
    overlay = frame.copy()

    for idx, (name, polygon) in enumerate(zones.items()):
        color = COLORS[idx % len(COLORS)]
        pts = polygon.reshape((-1, 1, 2)).astype(np.int32)

        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

        # Draw zone name and count
        centroid = polygon.mean(axis=0).astype(int)
        label = name
        if counts and name in counts:
            label = f"{name}: {counts[name]}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            frame,
            (centroid[0] - tw // 2 - 4, centroid[1] - th // 2 - 4),
            (centroid[0] + tw // 2 + 4, centroid[1] + th // 2 + 4),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (centroid[0] - tw // 2, centroid[1] + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    connections: list[tuple[int, int]] | None = None,
    confidence_threshold: float = 0.5,
    point_radius: int = 4,
    line_thickness: int = 2,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw a human skeleton from keypoints.

    Args:
        frame: BGR image to draw on.
        keypoints: Array of shape (17, 3) with x, y, confidence per keypoint.
        connections: List of (from, to) keypoint index pairs.
        confidence_threshold: Minimum keypoint confidence to draw.
        point_radius: Radius of keypoint circles.
        line_thickness: Thickness of skeleton lines.
        color: BGR color for the skeleton.

    Returns:
        Annotated frame.
    """
    if connections is None:
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]

    # Draw connections
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i, 2] > confidence_threshold and keypoints[j, 2] > confidence_threshold:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(frame, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    # Draw keypoints
    for k in range(len(keypoints)):
        if keypoints[k, 2] > confidence_threshold:
            pt = (int(keypoints[k, 0]), int(keypoints[k, 1]))
            cv2.circle(frame, pt, point_radius, (0, 0, 255), -1, cv2.LINE_AA)

    return frame


def draw_action_label(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    action: str,
    confidence: float,
    color: tuple[int, int, int] = (255, 128, 0),
) -> np.ndarray:
    """Draw an action label below a bounding box.

    Args:
        frame: BGR image to draw on.
        bbox: (x1, y1, x2, y2) bounding box.
        action: Action name string.
        confidence: Action confidence (0-1).
        color: BGR color for the label.

    Returns:
        Annotated frame.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    label = f"{action} {confidence:.0%}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 4, y2 + th + 8), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y2 + th + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_anomaly_alert(
    frame: np.ndarray,
    score: float,
    is_anomalous: bool,
) -> np.ndarray:
    """Draw anomaly detection status on frame.

    Args:
        frame: BGR image to draw on.
        score: Anomaly score.
        is_anomalous: Whether an anomaly was detected.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    if is_anomalous:
        # Red border for anomaly
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        text = f"ANOMALY DETECTED (score: {score:.3f})"
        color = (0, 0, 255)
    else:
        text = f"Normal (score: {score:.3f})"
        color = (0, 200, 0)

    cv2.putText(
        frame, text, (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )
    return frame


def draw_scene_info(
    frame: np.ndarray,
    scene_type: str,
    description: str,
    y_offset: int = 60,
) -> np.ndarray:
    """Draw scene understanding info on frame.

    Args:
        frame: BGR image to draw on.
        scene_type: Classified scene type.
        description: Brief scene description.
        y_offset: Y position offset from bottom.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # Scene type badge
    label = f"Scene: {scene_type}"
    cv2.putText(
        frame, label, (10, h - y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1, cv2.LINE_AA,
    )

    # Truncate description to fit
    max_chars = w // 10
    if len(description) > max_chars:
        description = description[:max_chars] + "..."

    cv2.putText(
        frame, description, (10, h - y_offset + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return frame


def draw_event_log(
    frame: np.ndarray,
    events: list[dict],
    max_events: int = 5,
) -> np.ndarray:
    """Draw temporal event log on the right side of frame.

    Args:
        frame: BGR image to draw on.
        events: List of event dicts with 'description' and 'severity'.
        max_events: Maximum events to display.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    severity_colors = {
        "info": (200, 200, 200),
        "warning": (0, 200, 255),
        "alert": (0, 0, 255),
    }

    recent = events[-max_events:]
    for i, event in enumerate(recent):
        y = 60 + i * 22
        color = severity_colors.get(event.get("severity", "info"), (200, 200, 200))
        text = event.get("description", "")[:60]
        cv2.putText(
            frame, text, (w - 500, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    return frame
