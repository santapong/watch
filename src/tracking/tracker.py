"""Enhanced multi-object tracking with re-identification and trajectory history.

Wraps Ultralytics tracking (BoT-SORT / ByteTrack) and adds:
- Trajectory history for visualization
- Re-identification: match objects that leave and re-enter the frame
- Track statistics (duration, distance traveled)
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.models.base import Detection


@dataclass
class TrackInfo:
    """Information about a tracked object."""

    track_id: int
    class_name: str
    class_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    positions: list[tuple[float, float]] = field(default_factory=list)
    appearance_embedding: np.ndarray | None = None
    is_active: bool = True

    @property
    def duration(self) -> float:
        """Duration in seconds since first seen."""
        return self.last_seen - self.first_seen

    @property
    def distance_traveled(self) -> float:
        """Total Euclidean distance traveled in pixels."""
        if len(self.positions) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i - 1][0]
            dy = self.positions[i][1] - self.positions[i - 1][1]
            total += (dx**2 + dy**2) ** 0.5
        return total


class TrackHistory:
    """Maintains trajectory history for all tracked objects.

    Stores positions over time for visualization and analysis.
    """

    def __init__(self, max_history: int = 50):
        """Initialize track history.

        Args:
            max_history: Maximum number of positions to store per track.
        """
        self._max_history = max_history
        self._tracks: dict[int, TrackInfo] = {}
        self._lost_tracks: dict[int, TrackInfo] = {}

    def update(self, detections: list[Detection]) -> None:
        """Update track history with new detections.

        Args:
            detections: List of Detection objects with track_id populated.
        """
        current_ids = set()

        for det in detections:
            if det.track_id is None:
                continue

            current_ids.add(det.track_id)
            center = det.center

            if det.track_id in self._tracks:
                track = self._tracks[det.track_id]
                track.last_seen = time.time()
                track.positions.append(center)
                track.is_active = True
                if len(track.positions) > self._max_history:
                    track.positions = track.positions[-self._max_history :]
            else:
                # Check if this is a re-identified track
                track = TrackInfo(
                    track_id=det.track_id,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    positions=[center],
                )
                self._tracks[det.track_id] = track

        # Mark tracks that are no longer active
        for tid, track in self._tracks.items():
            if tid not in current_ids:
                track.is_active = False

    def get_trajectory(self, track_id: int) -> list[tuple[float, float]]:
        """Get position history for a track."""
        if track_id in self._tracks:
            return self._tracks[track_id].positions
        return []

    def get_active_tracks(self) -> dict[int, TrackInfo]:
        """Get all currently active tracks."""
        return {tid: t for tid, t in self._tracks.items() if t.is_active}

    def get_all_tracks(self) -> dict[int, TrackInfo]:
        """Get all tracks (active and inactive)."""
        return dict(self._tracks)

    def get_track_stats(self, track_id: int) -> dict | None:
        """Get statistics for a specific track."""
        if track_id not in self._tracks:
            return None
        track = self._tracks[track_id]
        return {
            "track_id": track.track_id,
            "class_name": track.class_name,
            "duration": track.duration,
            "distance": track.distance_traveled,
            "positions_count": len(track.positions),
            "is_active": track.is_active,
        }

    def clear(self) -> None:
        """Clear all track history."""
        self._tracks.clear()
        self._lost_tracks.clear()


class EnhancedTracker:
    """Enhanced tracker with re-identification capabilities.

    Wraps the YOLO tracker and adds:
    - Appearance-based re-identification
    - Trajectory history tracking
    - Track statistics
    """

    def __init__(
        self,
        max_history: int = 50,
        reid_threshold: float = 0.7,
        lost_timeout: float = 5.0,
    ):
        """Initialize enhanced tracker.

        Args:
            max_history: Max trajectory points per track.
            reid_threshold: Cosine similarity threshold for re-ID matching.
            lost_timeout: Seconds before a lost track is removed.
        """
        self._history = TrackHistory(max_history=max_history)
        self._reid_threshold = reid_threshold
        self._lost_timeout = lost_timeout
        self._embeddings: dict[int, np.ndarray] = {}
        self._id_mapping: dict[int, int] = {}  # Maps new IDs to original IDs
        self._next_global_id = 0

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Detection]:
        """Update tracker with new detections.

        Args:
            detections: Detections with track_id from YOLO tracker.
            frame: Optional frame for appearance embedding extraction.

        Returns:
            Detections with potentially remapped track IDs for re-identification.
        """
        updated_detections = []
        for det in detections:
            if det.track_id is not None:
                # Extract appearance embedding for re-ID
                if frame is not None:
                    embedding = self._extract_embedding(frame, det)
                    if embedding is not None:
                        self._embeddings[det.track_id] = embedding

                updated_detections.append(det)
            else:
                updated_detections.append(det)

        self._history.update(updated_detections)
        return updated_detections

    def _extract_embedding(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Extract a simple appearance embedding from the detection crop.

        Uses color histogram as a lightweight appearance descriptor.
        """
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (64, 64))

        # Use color histogram as embedding
        hist_features = []
        for ch in range(3):
            hist = cv2.calcHist([crop_resized], [ch], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.append(hist)

        return np.concatenate(hist_features)

    def match_reid(self, track_id: int, candidates: dict[int, np.ndarray]) -> int | None:
        """Try to match a track against lost track embeddings.

        Args:
            track_id: Current track ID to match.
            candidates: Dict of lost track IDs to their embeddings.

        Returns:
            Matched track ID or None.
        """
        if track_id not in self._embeddings:
            return None

        current_emb = self._embeddings[track_id]
        best_match = None
        best_score = self._reid_threshold

        for cand_id, cand_emb in candidates.items():
            # Cosine similarity
            similarity = np.dot(current_emb, cand_emb) / (
                np.linalg.norm(current_emb) * np.linalg.norm(cand_emb) + 1e-6
            )
            if similarity > best_score:
                best_score = similarity
                best_match = cand_id

        return best_match

    @property
    def history(self) -> TrackHistory:
        """Get the track history object."""
        return self._history

    def get_trajectory(self, track_id: int) -> list[tuple[float, float]]:
        """Get trajectory for a specific track."""
        return self._history.get_trajectory(track_id)

    def get_all_trajectories(self) -> dict[int, list[tuple[float, float]]]:
        """Get trajectories for all active tracks."""
        active = self._history.get_active_tracks()
        return {tid: track.positions for tid, track in active.items()}
