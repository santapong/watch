"""Temporal detection and video understanding.

Detects events that only make sense over time:
- Object left unattended
- Loitering detection
- Object appears/disappears
- Crowd formation/dispersal
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from src.models.base import Detection


class EventType(Enum):
    """Types of temporal events."""

    OBJECT_APPEARED = "object_appeared"
    OBJECT_DISAPPEARED = "object_disappeared"
    LOITERING = "loitering"
    ABANDONED_OBJECT = "abandoned_object"
    CROWD_FORMING = "crowd_forming"
    CROWD_DISPERSING = "crowd_dispersing"
    SPEED_ANOMALY = "speed_anomaly"


@dataclass
class TemporalEvent:
    """A detected temporal event."""

    event_type: EventType
    timestamp: float
    description: str
    location: tuple[float, float] | None = None  # Center point
    track_id: int | None = None
    class_name: str | None = None
    severity: str = "info"  # "info", "warning", "alert"


@dataclass
class TrackedObjectState:
    """State of a tracked object over time."""

    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    positions: list[tuple[float, float]] = field(default_factory=list)
    stationary_since: float | None = None
    is_active: bool = True


class TemporalBuffer:
    """Maintains a sliding window of detection history.

    Stores per-frame detection snapshots for temporal analysis.
    """

    def __init__(self, max_frames: int = 300):
        """Initialize temporal buffer.

        Args:
            max_frames: Maximum number of frames to store.
        """
        self._max_frames = max_frames
        self._frames: deque[tuple[float, list[Detection]]] = deque(
            maxlen=max_frames
        )
        self._object_states: dict[int, TrackedObjectState] = {}

    def add(self, detections: list[Detection], timestamp: float | None = None) -> None:
        """Add a frame's detections to the buffer.

        Args:
            detections: Detections from the current frame.
            timestamp: Optional timestamp (defaults to current time).
        """
        ts = timestamp or time.time()
        self._frames.append((ts, detections))

        # Update object states
        current_ids = set()
        for det in detections:
            if det.track_id is None:
                continue

            current_ids.add(det.track_id)
            center = det.center

            if det.track_id in self._object_states:
                state = self._object_states[det.track_id]
                state.last_seen = ts
                state.positions.append(center)
                state.is_active = True

                # Keep positions bounded
                if len(state.positions) > self._max_frames:
                    state.positions = state.positions[-self._max_frames :]
            else:
                self._object_states[det.track_id] = TrackedObjectState(
                    track_id=det.track_id,
                    class_name=det.class_name,
                    first_seen=ts,
                    last_seen=ts,
                    positions=[center],
                )

        # Mark inactive objects
        for tid, state in self._object_states.items():
            if tid not in current_ids:
                state.is_active = False

    def get_recent(self, seconds: float = 5.0) -> list[tuple[float, list[Detection]]]:
        """Get detections from the last N seconds."""
        now = time.time()
        cutoff = now - seconds
        return [(ts, dets) for ts, dets in self._frames if ts >= cutoff]

    def get_object_duration(self, track_id: int) -> float:
        """Get how long an object has been tracked (in seconds)."""
        if track_id not in self._object_states:
            return 0.0
        state = self._object_states[track_id]
        return state.last_seen - state.first_seen

    @property
    def object_states(self) -> dict[int, TrackedObjectState]:
        return dict(self._object_states)

    def clear(self) -> None:
        self._frames.clear()
        self._object_states.clear()


class EventDetector:
    """Detects time-based events from temporal detection history.

    Events detected:
    - Loitering: Object stationary in an area for too long
    - Abandoned object: Non-person object stationary for extended time
    - Object appeared/disappeared: Track lifecycle events
    - Crowd formation: Sudden increase in person count
    - Speed anomaly: Object moving unusually fast or slow

    Example:
        buffer = TemporalBuffer()
        detector = EventDetector(loiter_seconds=30)

        for frame in video:
            detections = yolo.detect_and_track(frame)
            buffer.add(detections)
            events = detector.check(buffer)
            for event in events:
                print(f"[{event.severity}] {event.description}")
    """

    def __init__(
        self,
        loiter_seconds: float = 30.0,
        abandoned_seconds: float = 60.0,
        stationary_threshold: float = 20.0,
        crowd_threshold: int = 10,
        crowd_change_rate: int = 5,
        speed_threshold: float = 100.0,
    ):
        """Initialize event detector.

        Args:
            loiter_seconds: Seconds before flagging loitering.
            abandoned_seconds: Seconds before flagging abandoned object.
            stationary_threshold: Pixel movement threshold for "stationary".
            crowd_threshold: Person count to trigger crowd alert.
            crowd_change_rate: Persons/second change rate for crowd events.
            speed_threshold: Pixel/frame speed for anomaly.
        """
        self._loiter_seconds = loiter_seconds
        self._abandoned_seconds = abandoned_seconds
        self._stationary_threshold = stationary_threshold
        self._crowd_threshold = crowd_threshold
        self._crowd_change_rate = crowd_change_rate
        self._speed_threshold = speed_threshold

        self._known_tracks: set[int] = set()
        self._person_counts: deque[tuple[float, int]] = deque(maxlen=100)
        self._fired_events: set[str] = set()  # Prevent duplicate events

    def check(self, buffer: TemporalBuffer) -> list[TemporalEvent]:
        """Check for temporal events based on current buffer state.

        Args:
            buffer: TemporalBuffer with recent detection history.

        Returns:
            List of newly detected events.
        """
        events = []
        now = time.time()

        events.extend(self._check_appearances(buffer))
        events.extend(self._check_loitering(buffer, now))
        events.extend(self._check_abandoned(buffer, now))
        events.extend(self._check_crowd(buffer, now))
        events.extend(self._check_speed(buffer))

        return events

    def _check_appearances(self, buffer: TemporalBuffer) -> list[TemporalEvent]:
        """Check for new objects appearing or disappearing."""
        events = []
        current_ids = {
            tid
            for tid, state in buffer.object_states.items()
            if state.is_active
        }

        # New appearances
        new_ids = current_ids - self._known_tracks
        for tid in new_ids:
            state = buffer.object_states[tid]
            event_key = f"appear_{tid}"
            if event_key not in self._fired_events:
                events.append(
                    TemporalEvent(
                        event_type=EventType.OBJECT_APPEARED,
                        timestamp=time.time(),
                        description=f"New {state.class_name} appeared (track #{tid})",
                        location=state.positions[-1] if state.positions else None,
                        track_id=tid,
                        class_name=state.class_name,
                    )
                )
                self._fired_events.add(event_key)

        # Disappearances
        lost_ids = self._known_tracks - current_ids
        for tid in lost_ids:
            if tid in buffer.object_states:
                state = buffer.object_states[tid]
                event_key = f"disappear_{tid}"
                if event_key not in self._fired_events:
                    duration = state.last_seen - state.first_seen
                    events.append(
                        TemporalEvent(
                            event_type=EventType.OBJECT_DISAPPEARED,
                            timestamp=time.time(),
                            description=f"{state.class_name} disappeared (track #{tid}, was present {duration:.1f}s)",
                            location=state.positions[-1] if state.positions else None,
                            track_id=tid,
                            class_name=state.class_name,
                        )
                    )
                    self._fired_events.add(event_key)

        self._known_tracks = current_ids
        return events

    def _check_loitering(
        self, buffer: TemporalBuffer, now: float
    ) -> list[TemporalEvent]:
        """Check for persons loitering in one area."""
        events = []
        for tid, state in buffer.object_states.items():
            if not state.is_active or state.class_name != "person":
                continue

            if len(state.positions) < 10:
                continue

            # Check if person has been mostly stationary
            recent_positions = state.positions[-30:]
            if len(recent_positions) < 5:
                continue

            max_displacement = max(
                ((p[0] - recent_positions[0][0]) ** 2 + (p[1] - recent_positions[0][1]) ** 2) ** 0.5
                for p in recent_positions
            )

            duration = state.last_seen - state.first_seen

            if max_displacement < self._stationary_threshold and duration > self._loiter_seconds:
                event_key = f"loiter_{tid}"
                if event_key not in self._fired_events:
                    events.append(
                        TemporalEvent(
                            event_type=EventType.LOITERING,
                            timestamp=now,
                            description=f"Person loitering for {duration:.0f}s (track #{tid})",
                            location=state.positions[-1],
                            track_id=tid,
                            class_name="person",
                            severity="warning",
                        )
                    )
                    self._fired_events.add(event_key)

        return events

    def _check_abandoned(
        self, buffer: TemporalBuffer, now: float
    ) -> list[TemporalEvent]:
        """Check for abandoned objects (non-person, stationary for long time)."""
        events = []
        non_person_classes = {"backpack", "suitcase", "handbag", "umbrella", "bag"}

        for tid, state in buffer.object_states.items():
            if not state.is_active:
                continue
            if state.class_name not in non_person_classes:
                continue

            if len(state.positions) < 5:
                continue

            # Check if stationary
            recent = state.positions[-20:]
            if len(recent) < 3:
                continue

            max_displacement = max(
                ((p[0] - recent[0][0]) ** 2 + (p[1] - recent[0][1]) ** 2) ** 0.5
                for p in recent
            )

            duration = state.last_seen - state.first_seen

            if max_displacement < self._stationary_threshold and duration > self._abandoned_seconds:
                event_key = f"abandoned_{tid}"
                if event_key not in self._fired_events:
                    events.append(
                        TemporalEvent(
                            event_type=EventType.ABANDONED_OBJECT,
                            timestamp=now,
                            description=f"Possible abandoned {state.class_name} for {duration:.0f}s (track #{tid})",
                            location=state.positions[-1],
                            track_id=tid,
                            class_name=state.class_name,
                            severity="alert",
                        )
                    )
                    self._fired_events.add(event_key)

        return events

    def _check_crowd(
        self, buffer: TemporalBuffer, now: float
    ) -> list[TemporalEvent]:
        """Check for crowd formation or dispersal."""
        events = []

        # Count current persons
        person_count = sum(
            1
            for state in buffer.object_states.values()
            if state.is_active and state.class_name == "person"
        )
        self._person_counts.append((now, person_count))

        if person_count >= self._crowd_threshold:
            event_key = f"crowd_{person_count // 5 * 5}"
            if event_key not in self._fired_events:
                events.append(
                    TemporalEvent(
                        event_type=EventType.CROWD_FORMING,
                        timestamp=now,
                        description=f"Crowd detected: {person_count} persons",
                        severity="warning",
                    )
                )
                self._fired_events.add(event_key)

        # Check for rapid changes
        if len(self._person_counts) >= 10:
            old_ts, old_count = self._person_counts[-10]
            time_diff = now - old_ts
            if time_diff > 0:
                change_rate = (person_count - old_count) / time_diff
                if change_rate > self._crowd_change_rate:
                    event_key = f"crowd_form_{int(now)}"
                    if event_key not in self._fired_events:
                        events.append(
                            TemporalEvent(
                                event_type=EventType.CROWD_FORMING,
                                timestamp=now,
                                description=f"Rapid crowd formation: +{person_count - old_count} persons in {time_diff:.1f}s",
                                severity="alert",
                            )
                        )
                        self._fired_events.add(event_key)
                elif change_rate < -self._crowd_change_rate:
                    event_key = f"crowd_disp_{int(now)}"
                    if event_key not in self._fired_events:
                        events.append(
                            TemporalEvent(
                                event_type=EventType.CROWD_DISPERSING,
                                timestamp=now,
                                description=f"Crowd dispersing: {old_count - person_count} persons left in {time_diff:.1f}s",
                                severity="warning",
                            )
                        )
                        self._fired_events.add(event_key)

        return events

    def _check_speed(self, buffer: TemporalBuffer) -> list[TemporalEvent]:
        """Check for unusual movement speeds."""
        events = []

        for tid, state in buffer.object_states.items():
            if not state.is_active or len(state.positions) < 3:
                continue

            # Calculate recent speed
            recent = state.positions[-5:]
            speeds = []
            for i in range(1, len(recent)):
                dx = recent[i][0] - recent[i - 1][0]
                dy = recent[i][1] - recent[i - 1][1]
                speeds.append((dx**2 + dy**2) ** 0.5)

            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                if avg_speed > self._speed_threshold:
                    event_key = f"speed_{tid}_{int(time.time())}"
                    if event_key not in self._fired_events:
                        events.append(
                            TemporalEvent(
                                event_type=EventType.SPEED_ANOMALY,
                                timestamp=time.time(),
                                description=f"Fast movement: {state.class_name} (track #{tid}) at {avg_speed:.0f} px/frame",
                                location=state.positions[-1],
                                track_id=tid,
                                class_name=state.class_name,
                                severity="info",
                            )
                        )
                        self._fired_events.add(event_key)

        return events

    def reset(self) -> None:
        """Reset all event tracking state."""
        self._known_tracks.clear()
        self._person_counts.clear()
        self._fired_events.clear()
