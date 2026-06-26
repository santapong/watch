"""Enhanced multi-object tracking with re-identification and trajectory history.

Wraps Ultralytics tracking (BoT-SORT / ByteTrack) and adds:
- Trajectory history for visualization
- Re-identification: match objects that leave and re-enter the frame
- Track statistics (duration, distance traveled)
"""

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from src.models.base import Detection
from src.tracking.reid import ReIDEmbedder, build_embedder


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
        reid_backend: str = "auto",
        embedder: ReIDEmbedder | None = None,
    ):
        """Initialize enhanced tracker.

        Args:
            max_history: Max trajectory points per track.
            reid_threshold: Cosine similarity threshold for re-ID matching.
            lost_timeout: Seconds a vanished track stays re-identifiable.
            reid_backend: Embedder backend ("auto" | "histogram" | "osnet"); only
                used when ``embedder`` is not supplied.
            embedder: Explicit embedder instance (overrides ``reid_backend``).
        """
        self._history = TrackHistory(max_history=max_history)
        self._reid_threshold = reid_threshold
        self._lost_timeout = lost_timeout
        self._embedder = embedder if embedder is not None else build_embedder(reid_backend)
        self._embeddings: dict[int, np.ndarray] = {}   # raw track_id -> latest embedding
        self._id_mapping: dict[int, int] = {}          # raw id -> earlier (canonical) id
        self._lost: dict[int, dict] = {}               # canonical id -> {embedding, lost_at, class_id}
        self._seen_ids: set[int] = set()               # every raw id ever observed
        self._class_of: dict[int, int] = {}            # canonical id -> class_id
        self._last_raw_of: dict[int, int] = {}         # canonical id -> latest raw id seen
        self._active: set[int] = set()                 # canonical ids present last frame

    def _resolve(self, track_id: int) -> int:
        """Follow id-mapping chains (a -> b -> c) to the canonical id, cycle-safe."""
        seen: set[int] = set()
        while track_id in self._id_mapping and track_id not in seen:
            seen.add(track_id)
            nxt = self._id_mapping[track_id]
            if nxt == track_id:
                break
            track_id = nxt
        return track_id

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> list[Detection]:
        """Update tracker with new detections and re-identify re-entering tracks.

        Detections arrive with raw track IDs from the YOLO tracker. When a track
        leaves and a *new* raw ID re-enters with a matching appearance, its ID is
        remapped back to the original so the identity (and trajectory) persists.

        Args:
            detections: Detections with track_id from the YOLO tracker.
            frame: Optional frame for appearance-embedding extraction. Without it,
                re-ID is skipped (existing remaps still apply) and detections pass
                through.

        Returns:
            Detections with re-identified track IDs (same length as the input).
        """
        now = time.time()

        # 1) Evict lost identities older than the timeout.
        for cid in list(self._lost):
            if now - self._lost[cid]["lost_at"] > self._lost_timeout:
                del self._lost[cid]

        # 2) Embed current detections; record presence by canonical id.
        present: set[int] = set()
        current_raw: set[int] = set()
        for det in detections:
            if det.track_id is None:
                continue
            raw = det.track_id
            current_raw.add(raw)
            if frame is not None:
                emb = self._embedder.embed(frame, det)
                if emb is not None:
                    self._embeddings[raw] = emb
            canon = self._resolve(raw)
            present.add(canon)
            self._class_of[canon] = det.class_id
            self._last_raw_of[canon] = raw

        # 3) Re-identify each NEW raw id against the lost pool (same class only).
        for raw in current_raw:
            if raw in self._seen_ids or raw in self._id_mapping:
                continue
            if raw not in self._embeddings:
                continue
            cls = self._class_of.get(raw)
            candidates = {
                cid: info["embedding"]
                for cid, info in self._lost.items()
                if info["class_id"] == cls
            }
            if not candidates:
                continue
            match = self.match_reid(raw, candidates)
            if match is not None:
                self._id_mapping[raw] = match
                del self._lost[match]
                present.discard(raw)
                present.add(match)
                self._last_raw_of[match] = raw

        self._seen_ids.update(current_raw)

        # 4) Remap detections to canonical ids (non-destructive copy).
        remapped: list[Detection] = []
        for det in detections:
            if det.track_id is not None and det.track_id in self._id_mapping:
                remapped.append(dataclasses.replace(det, track_id=self._resolve(det.track_id)))
            else:
                remapped.append(det)

        # 5) Move JUST-vanished identities (present last frame, absent now) into the
        #    lost pool — exactly once per disappearance, so a timed-out identity is
        #    not silently re-added every subsequent frame.
        for canon in self._active - present:
            if canon in self._lost:
                continue
            emb = self._embeddings.get(self._last_raw_of.get(canon, canon))
            if emb is not None:
                self._lost[canon] = {
                    "embedding": emb,
                    "lost_at": now,
                    "class_id": self._class_of.get(canon, -1),
                }
        self._active = present

        # 6) Feed history with remapped detections so trajectories continue.
        self._history.update(remapped)
        return remapped

    def _extract_embedding(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Extract an appearance embedding via the configured embedder.

        Thin delegate kept for backward compatibility; the actual descriptor is
        produced by ``self._embedder`` (histogram by default, OSNet when enabled).
        """
        return self._embedder.embed(frame, detection)

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
