"""Tests for ActivityLedger: hysteresis, idle timeout, transition accounting."""

import pytest

from src.workstation.ledger import ActivityLedger
from src.workstation.models import ActivityState, Employee, Station
from src.workstation.store import ActivityStore


@pytest.fixture
def env():
    store = ActivityStore(":memory:")
    station = store.upsert_station(Station(name="bench-1", camera_index=0))
    employee = store.upsert_employee(Employee(name="Ana", badge_id="EMP001"))
    shift = store.start_shift(employee.id, station.id, start_time=0.0)
    yield store, station, employee, shift
    store.close()


def _open(store, hysteresis=2.0, idle_timeout=10.0, initial=ActivityState.UNKNOWN):
    return ActivityLedger(
        store,
        hysteresis_seconds=hysteresis,
        idle_timeout_seconds=idle_timeout,
        initial_state=initial,
    )


class TestHysteresis:
    def test_no_commit_within_window(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, hysteresis=2.0)
        ledger.open_shift(shift, timestamp=0.0)

        assert ledger.observe(shift.id, ActivityState.FILING, timestamp=0.5) is None
        assert ledger.observe(shift.id, ActivityState.FILING, timestamp=1.0) is None
        # Still within 2.0s window from pending_since=0.5
        assert ledger.observe(shift.id, ActivityState.FILING, timestamp=2.0) is None
        assert ledger.current_state(shift.id) is ActivityState.UNKNOWN
        assert store.events_for_shift(shift.id) == []

    def test_commit_after_window(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, hysteresis=2.0)
        ledger.open_shift(shift, timestamp=0.0)

        ledger.observe(shift.id, ActivityState.FILING, timestamp=0.5)
        event = ledger.observe(shift.id, ActivityState.FILING, timestamp=3.0)
        assert event is not None
        assert event.state is ActivityState.UNKNOWN
        # The committed UNKNOWN event runs from 0.0 to pending_since (0.5).
        assert event.start_time == pytest.approx(0.0)
        assert event.end_time == pytest.approx(0.5)
        assert ledger.current_state(shift.id) is ActivityState.FILING

    def test_flicker_does_not_commit(self, env):
        # Classifier oscillates FILING <-> POLISHING within hysteresis window.
        # Neither candidate accumulates enough continuous time to fire.
        store, _station, _emp, shift = env
        ledger = _open(store, hysteresis=2.0)
        ledger.open_shift(shift, timestamp=0.0)

        for i, state in enumerate(
            [
                ActivityState.FILING,
                ActivityState.POLISHING,
                ActivityState.FILING,
                ActivityState.POLISHING,
                ActivityState.FILING,
            ]
        ):
            ledger.observe(shift.id, state, timestamp=0.5 + i * 0.3)

        assert ledger.current_state(shift.id) is ActivityState.UNKNOWN
        assert store.events_for_shift(shift.id) == []

    def test_returning_to_current_clears_pending(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, hysteresis=2.0, initial=ActivityState.POLISHING)
        ledger.open_shift(shift, timestamp=0.0)

        ledger.observe(shift.id, ActivityState.FILING, timestamp=0.5)
        ledger.observe(shift.id, ActivityState.POLISHING, timestamp=1.0)
        # Even after 2s the FILING candidate must not fire.
        event = ledger.observe(shift.id, ActivityState.FILING, timestamp=10.0)
        assert event is None
        assert ledger.current_state(shift.id) is ActivityState.POLISHING

    def test_zero_hysteresis_commits_immediately(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, hysteresis=0.0)
        ledger.open_shift(shift, timestamp=0.0)

        # First observation establishes pending.
        ledger.observe(shift.id, ActivityState.FILING, timestamp=1.0)
        # Second observation of same state -> hysteresis (0) cleared, commit.
        event = ledger.observe(shift.id, ActivityState.FILING, timestamp=1.0)
        assert event is not None
        assert ledger.current_state(shift.id) is ActivityState.FILING


class TestTransitionAccounting:
    def test_three_state_sequence(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, hysteresis=1.0)
        ledger.open_shift(shift, timestamp=0.0)

        # UNKNOWN -> FILING at t=2
        ledger.observe(shift.id, ActivityState.FILING, timestamp=2.0)
        ledger.observe(shift.id, ActivityState.FILING, timestamp=3.5)
        # FILING -> POLISHING at t=10
        ledger.observe(shift.id, ActivityState.POLISHING, timestamp=10.0)
        ledger.observe(shift.id, ActivityState.POLISHING, timestamp=12.0)
        # POLISHING -> TOOL_CHANGE at t=20
        ledger.observe(shift.id, ActivityState.TOOL_CHANGE, timestamp=20.0)
        ledger.observe(shift.id, ActivityState.TOOL_CHANGE, timestamp=22.0)
        ledger.close_shift(shift.id, timestamp=30.0)

        events = store.events_for_shift(shift.id)
        assert [e.state for e in events] == [
            ActivityState.UNKNOWN,
            ActivityState.FILING,
            ActivityState.POLISHING,
            ActivityState.TOOL_CHANGE,
        ]
        # Durations sum back to total elapsed shift time.
        total = sum(e.duration_seconds for e in events)
        assert total == pytest.approx(30.0)

    def test_close_shift_finalizes_current_event(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, hysteresis=0.0)
        ledger.open_shift(shift, timestamp=0.0)

        # No transitions — closing should still emit the trailing UNKNOWN.
        event = ledger.close_shift(shift.id, timestamp=42.0)
        assert event is not None
        assert event.state is ActivityState.UNKNOWN
        assert event.duration_seconds == pytest.approx(42.0)
        assert store.get_shift(shift.id).end_time == pytest.approx(42.0)

    def test_close_unknown_shift_returns_none(self, env):
        store, _station, _emp, _shift = env
        ledger = _open(store)
        assert ledger.close_shift(9999, timestamp=10.0) is None

    def test_observe_unknown_shift_raises(self, env):
        store, *_ = env
        ledger = _open(store)
        with pytest.raises(KeyError):
            ledger.observe(9999, ActivityState.FILING, timestamp=1.0)


class TestIdleTimeout:
    def test_auto_idle_after_silence(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, hysteresis=0.5, idle_timeout=5.0, initial=ActivityState.POLISHING)
        ledger.open_shift(shift, timestamp=0.0)

        ledger.observe(shift.id, ActivityState.POLISHING, timestamp=1.0)
        ledger.observe(shift.id, ActivityState.POLISHING, timestamp=2.0)
        # 6s after last observation -> idle timeout fires.
        committed = ledger.flush_idle(timestamp=8.0)

        assert len(committed) == 1
        assert committed[0].state is ActivityState.POLISHING
        # Idle event starts at last_observation (2.0), not at flush time.
        assert committed[0].end_time == pytest.approx(2.0)
        assert ledger.current_state(shift.id) is ActivityState.IDLE

    def test_flush_idle_noop_when_fresh(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, idle_timeout=10.0)
        ledger.open_shift(shift, timestamp=0.0)

        ledger.observe(shift.id, ActivityState.UNKNOWN, timestamp=1.0)
        assert ledger.flush_idle(timestamp=2.0) == []

    def test_flush_idle_does_not_re_idle(self, env):
        store, _station, _emp, shift = env
        ledger = _open(store, idle_timeout=5.0, initial=ActivityState.IDLE)
        ledger.open_shift(shift, timestamp=0.0)

        # Already idle and silent — should not commit a new event.
        assert ledger.flush_idle(timestamp=100.0) == []


class TestLedgerLifecycle:
    def test_open_requires_persisted_shift(self, env):
        store, station, employee, _shift = env
        from src.workstation.models import Shift

        ledger = _open(store)
        with pytest.raises(ValueError, match="must have an id"):
            ledger.open_shift(
                Shift(employee_id=employee.id, station_id=station.id, start_time=0.0)
            )

    def test_negative_hysteresis_rejected(self, env):
        store, *_ = env
        with pytest.raises(ValueError):
            ActivityLedger(store, hysteresis_seconds=-1.0)

    def test_zero_idle_timeout_rejected(self, env):
        store, *_ = env
        with pytest.raises(ValueError):
            ActivityLedger(store, idle_timeout_seconds=0.0)
