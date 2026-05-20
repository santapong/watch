#!/usr/bin/env python3
"""Run the workstation activity ledger end-to-end (Phase A scaffold).

Loads stations, employees, and assignments from ``configs/default.yaml``,
opens shifts in the SQLite store, and drives the ledger with a placeholder
classifier that always emits ``ActivityState.UNKNOWN``. Phase B replaces the
classifier without changing this runner.

The ``--demo`` flag substitutes a deterministic round-robin classifier so the
ledger can be exercised without a camera — useful for smoke-testing the
pipeline and inspecting what ends up in the events table.

Usage:
    # Real run (requires camera + workstation.enabled: true in config):
    python scripts/run_workstation.py

    # Offline demo (no camera, cycles through states for ~30s):
    python scripts/run_workstation.py --demo --duration 30

Inspect results:
    sqlite3 data/activity.db "SELECT state, start_time, end_time \
        FROM activity_events ORDER BY id;"
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workstation import (
    ActivityLedger,
    ActivityState,
    ActivityStore,
    Employee,
    StaticAssignment,
    Station,
)


_DEMO_CYCLE = [
    (ActivityState.TOOL_CHANGE, 3.0),
    (ActivityState.FILING, 8.0),
    (ActivityState.TOOL_CHANGE, 2.0),
    (ActivityState.POLISHING, 10.0),
    (ActivityState.IDLE, 5.0),
]


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def bootstrap(store: ActivityStore, ws_cfg: dict) -> tuple[list[Station], dict[int, int]]:
    """Persist stations + employees from config, return (stations, station_id→employee_id)."""
    stations: list[Station] = []
    for s in ws_cfg.get("stations") or []:
        stations.append(
            store.upsert_station(
                Station(
                    name=s["name"],
                    camera_index=int(s.get("camera_index", 0)),
                    location=s.get("location"),
                )
            )
        )

    badge_to_emp: dict[str, int] = {}
    for e in ws_cfg.get("employees") or []:
        emp = store.upsert_employee(
            Employee(
                name=e["name"],
                badge_id=e.get("badge_id"),
                consent=bool(e.get("consent", False)),
            )
        )
        if emp.badge_id:
            badge_to_emp[emp.badge_id] = emp.id  # type: ignore[assignment]

    assignments: dict[int, int] = {}
    name_to_station = {st.name: st for st in stations}
    for station_name, badge in (ws_cfg.get("assignments") or {}).items():
        station = name_to_station.get(station_name)
        emp_id = badge_to_emp.get(badge)
        if station and station.id is not None and emp_id is not None:
            assignments[station.id] = emp_id

    return stations, assignments


def _classifier_placeholder(_station_id: int, _now: float) -> ActivityState:
    return ActivityState.UNKNOWN


def _classifier_demo(start: float):
    cycle_total = sum(d for _, d in _DEMO_CYCLE)

    def _classify(_station_id: int, now: float) -> ActivityState:
        elapsed = (now - start) % cycle_total
        cursor = 0.0
        for state, duration in _DEMO_CYCLE:
            cursor += duration
            if elapsed < cursor:
                return state
        return ActivityState.UNKNOWN

    return _classify


def main() -> int:
    parser = argparse.ArgumentParser(description="Workstation activity ledger runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run an offline round-robin classifier instead of UNKNOWN",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Auto-exit after N seconds (0 = run until SIGINT)",
    )
    parser.add_argument(
        "--tick",
        type=float,
        default=0.5,
        help="Seconds between classifier observations",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ws_cfg = cfg.get("workstation") or {}
    if not args.demo and not ws_cfg.get("enabled", False):
        print(
            "workstation.enabled is false in config. Pass --demo to run the "
            "offline pipeline, or enable workstation in configs/default.yaml.",
            file=sys.stderr,
        )
        return 2

    db_path = ws_cfg.get("db_path", "data/activity.db")
    hysteresis = float(ws_cfg.get("hysteresis_seconds", 2.0))
    idle_timeout = float(ws_cfg.get("idle_timeout_seconds", 10.0))

    if args.demo and not ws_cfg.get("stations"):
        ws_cfg = {
            **ws_cfg,
            "stations": [{"name": "demo-station", "camera_index": 0}],
            "employees": [{"name": "Demo Employee", "badge_id": "DEMO", "consent": True}],
            "assignments": {"demo-station": "DEMO"},
        }

    store = ActivityStore(db_path)
    stations, assignments_map = bootstrap(store, ws_cfg)
    if not stations:
        print("No stations configured. Add entries under workstation.stations.", file=sys.stderr)
        store.close()
        return 2

    resolver = StaticAssignment(assignments_map)
    ledger = ActivityLedger(
        store,
        hysteresis_seconds=hysteresis,
        idle_timeout_seconds=idle_timeout,
    )

    shift_start = time.time()
    open_shifts: list[int] = []
    for station in stations:
        emp_id = resolver.resolve(station.id) if station.id is not None else None
        if emp_id is None:
            print(f"[skip] no employee assigned to station {station.name!r}")
            continue
        shift = store.start_shift(emp_id, station.id, shift_start)  # type: ignore[arg-type]
        ledger.open_shift(shift)
        open_shifts.append(shift.id)  # type: ignore[arg-type]
        print(f"[open] shift={shift.id} station={station.name} employee={emp_id}")

    if not open_shifts:
        print("No shifts opened (check workstation.assignments).", file=sys.stderr)
        store.close()
        return 2

    classify = _classifier_demo(shift_start) if args.demo else _classifier_placeholder

    stop = {"flag": False}

    def _handle_sigint(_signo, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print("Running. Ctrl-C to stop.")
    try:
        while not stop["flag"]:
            now = time.time()
            if args.duration and (now - shift_start) >= args.duration:
                break
            for shift_id, station in zip(open_shifts, stations):
                state = classify(station.id, now)  # type: ignore[arg-type]
                event = ledger.observe(shift_id, state, timestamp=now)
                if event is not None:
                    print(
                        f"[event] shift={event.shift_id} state={event.state.value} "
                        f"duration={event.duration_seconds:.2f}s"
                    )
            ledger.flush_idle(now)
            time.sleep(args.tick)
    finally:
        end_ts = time.time()
        for shift_id in open_shifts:
            event = ledger.close_shift(shift_id, timestamp=end_ts)
            if event is not None:
                print(
                    f"[close] shift={event.shift_id} final_state={event.state.value} "
                    f"duration={event.duration_seconds:.2f}s"
                )
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
