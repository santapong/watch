"""Tests for workstation data classes."""

import pytest

from src.workstation.models import (
    ActivityEvent,
    ActivityState,
    Employee,
    Shift,
    Station,
)


class TestActivityState:
    def test_from_str_known(self):
        assert ActivityState.from_str("polishing") is ActivityState.POLISHING
        assert ActivityState.from_str("tool_change") is ActivityState.TOOL_CHANGE
        assert ActivityState.from_str("idle") is ActivityState.IDLE

    def test_from_str_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown activity state"):
            ActivityState.from_str("dancing")

    def test_taxonomy_v1_complete(self):
        # Phase A taxonomy must cover these five states.
        values = {s.value for s in ActivityState}
        assert values == {"polishing", "filing", "tool_change", "idle", "unknown"}


class TestStation:
    def test_minimal(self):
        s = Station(name="bench-1", camera_index=0)
        assert s.id is None
        assert s.location is None
        assert s.camera_index == 0


class TestEmployee:
    def test_defaults(self):
        e = Employee(name="Ana")
        assert e.consent is False
        assert e.badge_id is None
        assert e.id is None


class TestShift:
    def test_open_shift_has_no_duration(self):
        s = Shift(employee_id=1, station_id=1, start_time=100.0)
        assert s.is_open is True
        assert s.duration_seconds is None

    def test_closed_shift_duration(self):
        s = Shift(employee_id=1, station_id=1, start_time=100.0, end_time=130.0)
        assert s.is_open is False
        assert s.duration_seconds == pytest.approx(30.0)


class TestActivityEvent:
    def test_duration(self):
        ev = ActivityEvent(
            shift_id=1,
            employee_id=1,
            station_id=1,
            state=ActivityState.FILING,
            start_time=10.0,
            end_time=25.5,
        )
        assert ev.duration_seconds == pytest.approx(15.5)

    def test_string_state_is_coerced(self):
        ev = ActivityEvent(
            shift_id=1,
            employee_id=1,
            station_id=1,
            state="polishing",  # type: ignore[arg-type]
            start_time=0.0,
            end_time=1.0,
        )
        assert ev.state is ActivityState.POLISHING

    def test_inverted_times_rejected(self):
        with pytest.raises(ValueError, match="precedes"):
            ActivityEvent(
                shift_id=1,
                employee_id=1,
                station_id=1,
                state=ActivityState.IDLE,
                start_time=10.0,
                end_time=5.0,
            )

    def test_zero_duration_allowed(self):
        # Edge case: a transition recorded at the same instant.
        ev = ActivityEvent(
            shift_id=1,
            employee_id=1,
            station_id=1,
            state=ActivityState.UNKNOWN,
            start_time=10.0,
            end_time=10.0,
        )
        assert ev.duration_seconds == 0.0
