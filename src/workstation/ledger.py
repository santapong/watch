"""Activity ledger: consume state observations, write contiguous intervals.

The ledger sits between the classifier (Phase B) and the store. It buffers
per-shift state and only commits a transition to the store once a candidate
new state has been observed continuously for ``hysteresis_seconds`` — without
this debounce the table would churn on single-frame classifier flicker
between (say) FILING and POLISHING.

Idle timeout: if a shift goes silent for ``idle_timeout_seconds`` (e.g. the
employee walks away and the classifier emits nothing), the ledger
auto-transitions to IDLE at the last-observation timestamp so utilization
math stays honest.
"""

import time
from dataclasses import dataclass

from .models import ActivityEvent, ActivityState
from .store import ActivityStore


@dataclass
class _ShiftState:
    shift_id: int
    employee_id: int
    station_id: int
    current_state: ActivityState
    current_event_start: float
    last_observation: float
    pending_state: ActivityState | None = None
    pending_since: float | None = None


class ActivityLedger:
    """Buffer state observations, debounce, and persist intervals.

    Example:
        store = ActivityStore("data/activity.db")
        ledger = ActivityLedger(store, hysteresis_seconds=2.0)
        shift = store.start_shift(employee_id=1, station_id=1, start_time=time.time())
        ledger.open_shift(shift)

        # Per-frame loop, classifier emits states:
        for state in classifier_stream:
            ledger.observe(shift.id, state)

        ledger.close_shift(shift.id)
    """

    def __init__(
        self,
        store: ActivityStore,
        hysteresis_seconds: float = 2.0,
        idle_timeout_seconds: float = 10.0,
        initial_state: ActivityState = ActivityState.UNKNOWN,
    ):
        if hysteresis_seconds < 0:
            raise ValueError("hysteresis_seconds must be >= 0")
        if idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be > 0")
        self._store = store
        self._hysteresis = hysteresis_seconds
        self._idle_timeout = idle_timeout_seconds
        self._initial_state = initial_state
        self._shifts: dict[int, _ShiftState] = {}

    def open_shift(self, shift, timestamp: float | None = None) -> None:
        """Begin tracking a shift. The first event starts in ``initial_state``."""
        if shift.id is None:
            raise ValueError("shift must have an id; persist via store.start_shift first")
        ts = timestamp if timestamp is not None else shift.start_time
        self._shifts[shift.id] = _ShiftState(
            shift_id=shift.id,
            employee_id=shift.employee_id,
            station_id=shift.station_id,
            current_state=self._initial_state,
            current_event_start=ts,
            last_observation=ts,
        )

    def observe(
        self,
        shift_id: int,
        state: ActivityState,
        timestamp: float | None = None,
    ) -> ActivityEvent | None:
        """Record one observation. Returns the committed event if a transition fired."""
        ts = timestamp if timestamp is not None else time.time()
        ss = self._shifts.get(shift_id)
        if ss is None:
            raise KeyError(f"shift {shift_id} is not open in ledger")

        ss.last_observation = ts

        if state == ss.current_state:
            ss.pending_state = None
            ss.pending_since = None
            return None

        if ss.pending_state != state:
            ss.pending_state = state
            ss.pending_since = ts
            return None

        # Same candidate as last time — check whether hysteresis has cleared.
        assert ss.pending_since is not None
        if ts - ss.pending_since < self._hysteresis:
            return None

        transition_ts = ss.pending_since
        event = self._commit(ss, transition_ts)
        ss.current_state = state
        ss.current_event_start = transition_ts
        ss.pending_state = None
        ss.pending_since = None
        return event

    def flush_idle(self, timestamp: float | None = None) -> list[ActivityEvent]:
        """Auto-transition stale shifts to IDLE. Returns committed events."""
        ts = timestamp if timestamp is not None else time.time()
        committed: list[ActivityEvent] = []
        for ss in self._shifts.values():
            if ss.current_state == ActivityState.IDLE:
                continue
            if ts - ss.last_observation < self._idle_timeout:
                continue
            transition_ts = ss.last_observation
            event = self._commit(ss, transition_ts)
            committed.append(event)
            ss.current_state = ActivityState.IDLE
            ss.current_event_start = transition_ts
            ss.pending_state = None
            ss.pending_since = None
        return committed

    def close_shift(
        self,
        shift_id: int,
        timestamp: float | None = None,
    ) -> ActivityEvent | None:
        """Finalize the in-flight event and mark the shift ended in the store."""
        ts = timestamp if timestamp is not None else time.time()
        ss = self._shifts.pop(shift_id, None)
        if ss is None:
            return None
        event = self._commit(ss, ts)
        self._store.end_shift(shift_id, ts)
        return event

    def current_state(self, shift_id: int) -> ActivityState | None:
        ss = self._shifts.get(shift_id)
        return ss.current_state if ss else None

    def _commit(self, ss: _ShiftState, end_time: float) -> ActivityEvent:
        if end_time < ss.current_event_start:
            end_time = ss.current_event_start
        event = ActivityEvent(
            shift_id=ss.shift_id,
            employee_id=ss.employee_id,
            station_id=ss.station_id,
            state=ss.current_state,
            start_time=ss.current_event_start,
            end_time=end_time,
        )
        return self._store.append_event(event)
