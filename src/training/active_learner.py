"""Active learning pipeline for efficient model improvement.

Identifies low-confidence detections, exports them for human labeling,
and manages the label-retrain loop to minimize annotation effort.
"""

import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.models.base import Detection


@dataclass
class UncertainSample:
    """A frame/detection flagged for human review."""

    frame_path: str
    detections: list[Detection]
    uncertainty_score: float
    timestamp: float
    frame_index: int
    reason: str  # "low_confidence", "conflicting_classes", "boundary_case"


class UncertaintySampler:
    """Identifies frames with uncertain detections for human labeling.

    Uncertainty criteria:
    - Low confidence detections (near threshold)
    - Conflicting predictions (multiple classes for overlapping boxes)
    - High-density regions where detections may be missed
    - Novel-looking detections (unusual size/position)

    Example:
        sampler = UncertaintySampler(confidence_range=(0.25, 0.6))
        frames_to_review = sampler.evaluate(detections, frame)
    """

    def __init__(
        self,
        confidence_range: tuple[float, float] = (0.2, 0.6),
        max_queue_size: int = 1000,
        sampling_interval: int = 30,
        output_dir: str = "labeling_queue",
    ):
        """Initialize uncertainty sampler.

        Args:
            confidence_range: (min, max) confidence range considered uncertain.
            max_queue_size: Maximum number of samples to keep in queue.
            sampling_interval: Minimum frames between samples.
            output_dir: Directory to save uncertain frames.
        """
        self._conf_min, self._conf_max = confidence_range
        self._max_queue = max_queue_size
        self._sampling_interval = sampling_interval
        self._output_dir = Path(output_dir)
        self._queue: list[UncertainSample] = []
        self._frame_counter = 0
        self._last_sample_frame = -sampling_interval

    def evaluate(
        self, detections: list[Detection], frame: np.ndarray
    ) -> UncertainSample | None:
        """Evaluate a frame for uncertainty and potentially queue it.

        Args:
            detections: Detections from the current frame.
            frame: The BGR frame image.

        Returns:
            UncertainSample if the frame was queued, None otherwise.
        """
        self._frame_counter += 1

        # Respect sampling interval
        if self._frame_counter - self._last_sample_frame < self._sampling_interval:
            return None

        # Calculate uncertainty score
        score, reason = self._compute_uncertainty(detections)

        if score > 0.3:  # Threshold for "interesting" frames
            sample = self._save_sample(frame, detections, score, reason)
            if sample:
                self._last_sample_frame = self._frame_counter
                return sample

        return None

    def _compute_uncertainty(
        self, detections: list[Detection]
    ) -> tuple[float, str]:
        """Compute uncertainty score for a set of detections.

        Returns:
            Tuple of (score, reason) where score is 0-1.
        """
        if not detections:
            return 0.1, "no_detections"

        scores = []
        reasons = []

        # Check for low-confidence detections
        uncertain_dets = [
            d for d in detections if self._conf_min <= d.confidence <= self._conf_max
        ]
        if uncertain_dets:
            avg_uncertain_conf = np.mean([d.confidence for d in uncertain_dets])
            conf_score = 1.0 - avg_uncertain_conf
            scores.append(conf_score * 0.6)
            reasons.append("low_confidence")

        # Check for conflicting classes (overlapping boxes, different classes)
        conflict_score = self._check_conflicts(detections)
        if conflict_score > 0:
            scores.append(conflict_score * 0.3)
            reasons.append("conflicting_classes")

        # Check for boundary cases (detections at frame edges)
        boundary_count = sum(
            1
            for d in detections
            if d.bbox[0] < 10
            or d.bbox[1] < 10
            or d.bbox[2] > 1270
            or d.bbox[3] > 710
        )
        if boundary_count > 0:
            scores.append(boundary_count / len(detections) * 0.1)
            reasons.append("boundary_case")

        total_score = min(sum(scores), 1.0) if scores else 0.0
        main_reason = reasons[0] if reasons else "none"

        return total_score, main_reason

    def _check_conflicts(self, detections: list[Detection]) -> float:
        """Check for conflicting class predictions on overlapping boxes."""
        conflicts = 0
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                iou = self._compute_iou(detections[i].bbox, detections[j].bbox)
                if iou > 0.5 and detections[i].class_id != detections[j].class_id:
                    conflicts += 1

        return min(conflicts / max(len(detections), 1), 1.0)

    @staticmethod
    def _compute_iou(
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _save_sample(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        score: float,
        reason: str,
    ) -> UncertainSample | None:
        """Save uncertain frame and metadata to disk."""
        if len(self._queue) >= self._max_queue:
            # Remove least uncertain sample
            self._queue.sort(key=lambda s: s.uncertainty_score)
            if self._queue and self._queue[0].uncertainty_score >= score:
                return None
            removed = self._queue.pop(0)
            Path(removed.frame_path).unlink(missing_ok=True)

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Save frame
        filename = f"uncertain_{self._frame_counter:06d}_{score:.2f}.jpg"
        frame_path = str(self._output_dir / filename)
        cv2.imwrite(frame_path, frame)

        # Save detection metadata
        meta_path = frame_path.replace(".jpg", ".json")
        meta = {
            "frame_index": self._frame_counter,
            "uncertainty_score": score,
            "reason": reason,
            "detections": [
                {
                    "bbox": list(d.bbox),
                    "confidence": d.confidence,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
                for d in detections
            ],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        sample = UncertainSample(
            frame_path=frame_path,
            detections=detections,
            uncertainty_score=score,
            timestamp=time.time(),
            frame_index=self._frame_counter,
            reason=reason,
        )
        self._queue.append(sample)
        return sample

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def queue(self) -> list[UncertainSample]:
        return list(self._queue)

    def clear_queue(self) -> None:
        """Clear all queued samples and delete saved files."""
        for sample in self._queue:
            Path(sample.frame_path).unlink(missing_ok=True)
            meta_path = sample.frame_path.replace(".jpg", ".json")
            Path(meta_path).unlink(missing_ok=True)
        self._queue.clear()


class ActiveLearner:
    """Manages the active learning loop: detect → sample → label → retrain.

    Workflow:
    1. Run detection on video stream
    2. UncertaintySampler flags uncertain frames
    3. Human reviews and labels corrections
    4. Model is fine-tuned on corrected labels
    5. Repeat

    Example:
        learner = ActiveLearner(output_dir="active_learning")
        learner.start_session("session_001")

        for frame in video:
            detections = detector.detect(frame)
            sample = learner.evaluate(detections, frame)
            if sample:
                print(f"Queued frame {sample.frame_index} (score: {sample.uncertainty_score:.2f})")

        learner.export_for_labeling()
    """

    def __init__(
        self,
        output_dir: str = "active_learning",
        confidence_range: tuple[float, float] = (0.2, 0.6),
        max_samples: int = 500,
        sampling_interval: int = 30,
    ):
        """Initialize active learner.

        Args:
            output_dir: Root directory for active learning data.
            confidence_range: Uncertainty confidence range.
            max_samples: Maximum samples per session.
            sampling_interval: Minimum frames between samples.
        """
        self._output_dir = Path(output_dir)
        self._sampler = UncertaintySampler(
            confidence_range=confidence_range,
            max_queue_size=max_samples,
            sampling_interval=sampling_interval,
            output_dir=str(self._output_dir / "queue"),
        )
        self._session_name: str | None = None
        self._stats = defaultdict(int)

    def start_session(self, session_name: str) -> None:
        """Start a new active learning session.

        Args:
            session_name: Unique session identifier.
        """
        self._session_name = session_name
        session_dir = self._output_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        self._sampler._output_dir = session_dir / "queue"
        self._stats = defaultdict(int)

    def evaluate(
        self, detections: list[Detection], frame: np.ndarray
    ) -> UncertainSample | None:
        """Evaluate frame and potentially queue for labeling.

        Args:
            detections: Current frame detections.
            frame: BGR frame image.

        Returns:
            UncertainSample if queued, None otherwise.
        """
        self._stats["frames_processed"] += 1
        self._stats["total_detections"] += len(detections)

        sample = self._sampler.evaluate(detections, frame)
        if sample:
            self._stats["samples_queued"] += 1
        return sample

    def export_for_labeling(self, format: str = "yolo") -> str:
        """Export queued samples in a format suitable for labeling tools.

        Args:
            format: Export format ("yolo", "coco", "voc").

        Returns:
            Path to the exported dataset.
        """
        export_dir = self._output_dir / (self._session_name or "default") / "export"
        images_dir = export_dir / "images"
        labels_dir = export_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for sample in self._sampler.queue:
            src_path = Path(sample.frame_path)
            if not src_path.exists():
                continue

            # Copy image
            dst_image = images_dir / src_path.name
            shutil.copy2(src_path, dst_image)

            # Create pre-annotation file (YOLO format)
            if format == "yolo":
                label_path = labels_dir / src_path.with_suffix(".txt").name
                with open(label_path, "w") as f:
                    for det in sample.detections:
                        # YOLO format: class_id cx cy w h (normalized)
                        cx = (det.bbox[0] + det.bbox[2]) / 2
                        cy = (det.bbox[1] + det.bbox[3]) / 2
                        w = det.bbox[2] - det.bbox[0]
                        h = det.bbox[3] - det.bbox[1]
                        # Note: these need to be normalized by frame dimensions
                        f.write(f"{det.class_id} {cx} {cy} {w} {h}\n")

        return str(export_dir)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def queue_size(self) -> int:
        return self._sampler.queue_size
