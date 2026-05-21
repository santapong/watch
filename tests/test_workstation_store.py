"""Tests for ActivityStore (SQLite-backed)."""

import pytest

from src.workstation.models import (
    ActivityEvent,
    ActivityState,
    Employee,
    Station,
)
from src.workstation.store import ActivityStore


@pytest.fixture
def store():
    s = ActivityStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def seeded(store):
    station = store.upsert_station(Station(name="bench-1", camera_index=0, location="floor"))
    employee = store.upsert_employee(Employee(name="Ana", badge_id="EMP001", consent=True))
    return store, station, employee


class TestStations:
    def test_upsert_assigns_id(self, store):
        st = store.upsert_station(Station(name="bench-1", camera_index=0))
        assert st.id is not None

    def test_upsert_is_idempotent_on_name(self, store):
        a = store.upsert_station(Station(name="bench-1", camera_index=0))
        b = store.upsert_station(Station(name="bench-1", camera_index=2, location="updated"))
        assert a.id == b.id
        round_trip = store.get_station(a.id)
        assert round_trip is not None
        assert round_trip.camera_index == 2
        assert round_trip.location == "updated"

    def test_get_by_name(self, store):
        store.upsert_station(Station(name="bench-1", camera_index=0))
        assert store.get_station_by_name("bench-1") is not None
        assert store.get_station_by_name("nonexistent") is None


class TestEmployees:
    def test_upsert_with_badge(self, store):
        e = store.upsert_employee(Employee(name="Ana", badge_id="EMP001"))
        assert e.id is not None

    def test_upsert_idempotent_on_badge(self, store):
        a = store.upsert_employee(Employee(name="Ana", badge_id="EMP001", consent=False))
        b = store.upsert_employee(Employee(name="Ana B.", badge_id="EMP001", consent=True))
        assert a.id == b.id
        round_trip = store.get_employee(a.id)
        assert round_trip is not None
        assert round_trip.name == "Ana B."
        assert round_trip.consent is True

    def test_upsert_without_badge_inserts_each_time(self, store):
        a = store.upsert_employee(Employee(name="Anon"))
        b = store.upsert_employee(Employee(name="Anon"))
        assert a.id != b.id

    def test_get_by_badge(self, store):
        store.upsert_employee(Employee(name="Ana", badge_id="EMP001"))
        assert store.get_employee_by_badge("EMP001") is not None
        assert store.get_employee_by_badge("NOPE") is None


class TestShifts:
    def test_start_shift(self, seeded):
        store, station, employee = seeded
        shift = store.start_shift(employee.id, station.id, start_time=100.0)
        assert shift.id is not None
        assert shift.is_open

    def test_end_shift(self, seeded):
        store, station, employee = seeded
        shift = store.start_shift(employee.id, station.id, start_time=100.0)
        store.end_shift(shift.id, end_time=130.0)
        round_trip = store.get_shift(shift.id)
        assert round_trip is not None
        assert round_trip.end_time == pytest.approx(130.0)


class TestActivityEvents:
    def test_append_and_roundtrip(self, seeded):
        store, station, employee = seeded
        shift = store.start_shift(employee.id, station.id, start_time=0.0)
        store.append_event(
            ActivityEvent(
                shift_id=shift.id,
                employee_id=employee.id,
                station_id=station.id,
                state=ActivityState.FILING,
                start_time=0.0,
                end_time=10.0,
            )
        )
        store.append_event(
            ActivityEvent(
                shift_id=shift.id,
                employee_id=employee.id,
                station_id=station.id,
                state=ActivityState.POLISHING,
                start_time=10.0,
                end_time=25.0,
            )
        )
        events = store.events_for_shift(shift.id)
        assert [e.state for e in events] == [ActivityState.FILING, ActivityState.POLISHING]
        assert events[0].duration_seconds == pytest.approx(10.0)
        assert events[1].duration_seconds == pytest.approx(15.0)

    def test_range_query_filters_correctly(self, seeded):
        store, station, employee = seeded
        shift = store.start_shift(employee.id, station.id, start_time=0.0)

        def _ev(state, start, end):
            return ActivityEvent(
                shift_id=shift.id,
                employee_id=employee.id,
                station_id=station.id,
                state=state,
                start_time=start,
                end_time=end,
            )

        store.append_event(_ev(ActivityState.FILING, 0.0, 10.0))
        store.append_event(_ev(ActivityState.POLISHING, 10.0, 20.0))
        store.append_event(_ev(ActivityState.TOOL_CHANGE, 20.0, 22.0))
        store.append_event(_ev(ActivityState.IDLE, 22.0, 60.0))

        # Range overlaps first two events
        events = store.events_in_range(5.0, 15.0)
        states = [e.state for e in events]
        assert ActivityState.FILING in states
        assert ActivityState.POLISHING in states
        assert ActivityState.IDLE not in states

        # State filter
        polishing = store.events_in_range(0.0, 100.0, state=ActivityState.POLISHING)
        assert len(polishing) == 1
        assert polishing[0].state is ActivityState.POLISHING

        # Employee/station filter
        for_emp = store.events_in_range(0.0, 100.0, employee_id=employee.id)
        assert len(for_emp) == 4
        for_other_station = store.events_in_range(0.0, 100.0, station_id=station.id + 999)
        assert for_other_station == []

    def test_range_query_excludes_touching_boundaries(self, seeded):
        # An event from [10, 20) should not appear in range [20, 30).
        store, station, employee = seeded
        shift = store.start_shift(employee.id, station.id, start_time=0.0)
        store.append_event(
            ActivityEvent(
                shift_id=shift.id,
                employee_id=employee.id,
                station_id=station.id,
                state=ActivityState.FILING,
                start_time=10.0,
                end_time=20.0,
            )
        )
        assert store.events_in_range(20.0, 30.0) == []
        assert len(store.events_in_range(15.0, 25.0)) == 1


class TestPersistence:
    def test_disk_roundtrip(self, tmp_path):
        path = tmp_path / "activity.db"
        with ActivityStore(path) as s:
            station = s.upsert_station(Station(name="bench-1", camera_index=0))
            employee = s.upsert_employee(Employee(name="Ana", badge_id="EMP001"))
            shift = s.start_shift(employee.id, station.id, start_time=0.0)
            s.append_event(
                ActivityEvent(
                    shift_id=shift.id,
                    employee_id=employee.id,
                    station_id=station.id,
                    state=ActivityState.POLISHING,
                    start_time=0.0,
                    end_time=5.0,
                )
            )

        with ActivityStore(path) as s2:
            assert s2.get_station_by_name("bench-1") is not None
            assert s2.get_employee_by_badge("EMP001") is not None
            events = s2.events_in_range(0.0, 10.0)
            assert len(events) == 1
            assert events[0].state is ActivityState.POLISHING

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "subdir" / "activity.db"
        with ActivityStore(path) as s:
            s.upsert_station(Station(name="bench-1", camera_index=0))
        assert path.exists()
