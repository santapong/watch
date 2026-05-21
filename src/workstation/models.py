"""Data model for workstation activity monitoring.

Five activity states form taxonomy v1. POLISHING / FILING / TOOL_CHANGE are
the work states the CEO report cares about; IDLE and UNKNOWN are required
negative classes so utilization percentages are meaningful (UNKNOWN covers
the gap before Phase B's trained classifier replaces the placeholder).
"""

from dataclasses import dataclass
from enum import Enum


class ActivityState(str, Enum):
    POLISHING = "polishing"
    FILING = "filing"
    TOOL_CHANGE = "tool_change"
    IDLE = "idle"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, value: str) -> "ActivityState":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"Unknown activity state {value!r}. "
                f"Expected one of: {[s.value for s in cls]}"
            ) from exc


@dataclass
class Station:
    name: str
    camera_index: int
    location: str | None = None
    id: int | None = None


@dataclass
class Employee:
    name: str
    badge_id: str | None = None
    consent: bool = False
    id: int | None = None


@dataclass
class Shift:
    employee_id: int
    station_id: int
    start_time: float
    end_time: float | None = None
    id: int | None = None

    @property
    def is_open(self) -> bool:
        return self.end_time is None

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class ActivityEvent:
    """A contiguous interval during which the station was in one state."""

    shift_id: int
    employee_id: int
    station_id: int
    state: ActivityState
    start_time: float
    end_time: float
    id: int | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError(
                f"ActivityEvent end_time ({self.end_time}) precedes "
                f"start_time ({self.start_time})"
            )
        if isinstance(self.state, str) and not isinstance(self.state, ActivityState):
            self.state = ActivityState.from_str(self.state)
