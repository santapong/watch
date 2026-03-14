"""Pose detection and action recognition using YOLOv8-pose.

Extracts human skeletons and classifies actions based on pose sequences.
Supports: standing, sitting, walking, running, falling, raising hand.
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from .base import BaseDetector, Detection


# COCO keypoint indices
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # Head
    (5, 6),                                  # Shoulders
    (5, 7), (7, 9),                          # Left arm
    (6, 8), (8, 10),                         # Right arm
    (5, 11), (6, 12),                        # Torso
    (11, 12),                                # Hips
    (11, 13), (13, 15),                      # Left leg
    (12, 14), (14, 16),                      # Right leg
]


@dataclass
class PoseResult:
    """Result from pose detection for a single person."""

    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray  # Shape (17, 3) — x, y, confidence per keypoint
    track_id: int | None = None
    action: str | None = None
    action_confidence: float = 0.0


class PoseDetector(BaseDetector):
    """YOLOv8-pose wrapper for skeleton extraction.

    Detects human poses with 17 COCO keypoints per person.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        confidence: float = 0.25,
        device: str = "",
    ):
        """Initialize pose detector.

        Args:
            model_name: YOLOv8-pose model file.
            confidence: Minimum confidence threshold.
            device: Device string.
        """
        self._model = YOLO(model_name)
        self._model_name = model_name
        self._confidence = confidence
        self._device = device

    def detect_poses(self, frame: np.ndarray) -> list[PoseResult]:
        """Detect human poses in a frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of PoseResult objects with keypoints.
        """
        results = self._model.predict(
            frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        poses = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes
            keypoints = result.keypoints

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                kpts = keypoints.data[i].cpu().numpy()  # (17, 3)
                track_id = int(boxes.id[i].cpu()) if boxes.id is not None else None

                poses.append(
                    PoseResult(
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        keypoints=kpts,
                        track_id=track_id,
                    )
                )

        return poses

    def detect_poses_and_track(self, frame: np.ndarray) -> list[PoseResult]:
        """Detect poses with tracking enabled."""
        results = self._model.track(
            frame,
            conf=self._confidence,
            device=self._device,
            persist=True,
            verbose=False,
        )

        poses = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            boxes = result.boxes
            keypoints = result.keypoints

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                kpts = keypoints.data[i].cpu().numpy()
                track_id = int(boxes.id[i].cpu()) if boxes.id is not None else None

                poses.append(
                    PoseResult(
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        keypoints=kpts,
                        track_id=track_id,
                    )
                )

        return poses

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Standard detect interface — returns person detections."""
        poses = self.detect_poses(frame)
        return [
            Detection(
                bbox=p.bbox,
                confidence=p.confidence,
                class_id=0,
                class_name="person",
                track_id=p.track_id,
            )
            for p in poses
        ]

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        """Standard detect+track interface."""
        poses = self.detect_poses_and_track(frame)
        return [
            Detection(
                bbox=p.bbox,
                confidence=p.confidence,
                class_id=0,
                class_name="person",
                track_id=p.track_id,
            )
            for p in poses
        ]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "person"}


class ActionClassifier:
    """Classifies human actions from pose sequences.

    Uses rule-based classification on keypoint geometry and temporal patterns.
    Supports: standing, sitting, walking, running, falling, raising_hand.

    Example:
        classifier = ActionClassifier(sequence_length=15)
        poses = pose_detector.detect_poses(frame)
        for pose in poses:
            action, conf = classifier.classify(pose)
    """

    def __init__(self, sequence_length: int = 15):
        """Initialize action classifier.

        Args:
            sequence_length: Number of frames to consider for temporal actions.
        """
        self._sequence_length = sequence_length
        self._pose_history: dict[int, deque] = {}
        self._frame_counter = 0

    def classify(self, pose: PoseResult) -> tuple[str, float]:
        """Classify the action of a detected person.

        Args:
            pose: PoseResult with keypoints.

        Returns:
            Tuple of (action_name, confidence).
        """
        kpts = pose.keypoints  # (17, 3)

        # Store in history for temporal analysis
        track_id = pose.track_id or 0
        if track_id not in self._pose_history:
            self._pose_history[track_id] = deque(maxlen=self._sequence_length)
        self._pose_history[track_id].append(kpts)

        # Get visible keypoints (confidence > 0.5)
        visible = kpts[:, 2] > 0.5

        # Need at least shoulders and hips for classification
        if not (visible[5] and visible[6] and visible[11] and visible[12]):
            return "unknown", 0.0

        # Extract key measurements
        left_shoulder = kpts[5, :2]
        right_shoulder = kpts[6, :2]
        left_hip = kpts[11, :2]
        right_hip = kpts[12, :2]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        torso_height = np.linalg.norm(shoulder_center - hip_center)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        if torso_height < 1:
            return "unknown", 0.0

        # Aspect ratio of torso
        torso_ratio = torso_height / (shoulder_width + 1e-6)

        # Check for falling (torso nearly horizontal)
        torso_angle = abs(np.arctan2(
            shoulder_center[1] - hip_center[1],
            shoulder_center[0] - hip_center[0],
        ))
        torso_angle_deg = np.degrees(torso_angle)

        if torso_angle_deg < 30:
            return "falling", 0.8

        # Check for raising hand
        if visible[9] or visible[10]:  # Wrists visible
            left_wrist_up = visible[9] and kpts[9, 1] < kpts[5, 1]  # Wrist above shoulder
            right_wrist_up = visible[10] and kpts[10, 1] < kpts[6, 1]
            if left_wrist_up or right_wrist_up:
                return "raising_hand", 0.7

        # Check for sitting (hips close to knees vertically)
        if visible[13] and visible[14]:  # Knees visible
            left_knee = kpts[13, :2]
            right_knee = kpts[14, :2]
            knee_center = (left_knee + right_knee) / 2

            hip_knee_vertical = abs(hip_center[1] - knee_center[1])
            hip_knee_horizontal = abs(hip_center[0] - knee_center[0])

            if hip_knee_vertical < torso_height * 0.3:
                return "sitting", 0.7

        # Check for walking/running using temporal motion
        history = self._pose_history.get(track_id, deque())
        if len(history) >= 5:
            # Calculate hip movement speed
            recent_hips = []
            for h in list(history)[-5:]:
                if h[11, 2] > 0.5 and h[12, 2] > 0.5:
                    hc = (h[11, :2] + h[12, :2]) / 2
                    recent_hips.append(hc)

            if len(recent_hips) >= 3:
                movements = [
                    np.linalg.norm(recent_hips[i] - recent_hips[i - 1])
                    for i in range(1, len(recent_hips))
                ]
                avg_movement = np.mean(movements)

                if avg_movement > 15:
                    return "running", 0.7
                elif avg_movement > 5:
                    return "walking", 0.7

        # Default: standing
        if torso_ratio > 1.2:
            return "standing", 0.6

        return "standing", 0.5

    def classify_batch(self, poses: list[PoseResult]) -> list[PoseResult]:
        """Classify actions for all detected poses.

        Args:
            poses: List of PoseResult objects.

        Returns:
            Same list with action and action_confidence fields populated.
        """
        for pose in poses:
            action, confidence = self.classify(pose)
            pose.action = action
            pose.action_confidence = confidence
        return poses

    def clear_history(self) -> None:
        """Clear pose history for all tracks."""
        self._pose_history.clear()
