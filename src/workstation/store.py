"""SQLite-backed store for workstation activity data.

Schema is created on first use. Range queries are indexed by
(employee_id, start_time), (station_id, start_time), and (state, start_time)
so the Phase D report generator can scan a date range without a full table
scan.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import ActivityEvent, ActivityState, Employee, Shift, Station

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    camera_index INTEGER NOT NULL,
    location TEXT
);

CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    badge_id TEXT UNIQUE,
    consent INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shifts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    station_id INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    FOREIGN KEY (employee_id) REFERENCES employees(id),
    FOREIGN KEY (station_id) REFERENCES stations(id)
);
CREATE INDEX IF NOT EXISTS idx_shifts_employee ON shifts(employee_id, start_time);
CREATE INDEX IF NOT EXISTS idx_shifts_station ON shifts(station_id, start_time);

CREATE TABLE IF NOT EXISTS activity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shift_id INTEGER NOT NULL,
    employee_id INTEGER NOT NULL,
    station_id INTEGER NOT NULL,
    state TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    FOREIGN KEY (shift_id) REFERENCES shifts(id)
);
CREATE INDEX IF NOT EXISTS idx_events_shift ON activity_events(shift_id, start_time);
CREATE INDEX IF NOT EXISTS idx_events_employee_time ON activity_events(employee_id, start_time);
CREATE INDEX IF NOT EXISTS idx_events_station_time ON activity_events(station_id, start_time);
CREATE INDEX IF NOT EXISTS idx_events_state ON activity_events(state, start_time);
"""


class ActivityStore:
    """SQLite store for stations, employees, shifts, and activity events.

    Pass ``:memory:`` as the path for an in-process database (used by tests).
    Parent directories are created on disk-backed paths.
    """

    def __init__(self, db_path: str | Path = "data/activity.db"):
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ActivityStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ── Stations ─────────────────────────────────────────────────

    def upsert_station(self, station: Station) -> Station:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO stations (name, camera_index, location) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "  camera_index = excluded.camera_index, "
                "  location = excluded.location",
                (station.name, station.camera_index, station.location),
            )
            cur.execute("SELECT id FROM stations WHERE name = ?", (station.name,))
            row = cur.fetchone()
        station.id = row["id"]
        return station

    def get_station(self, station_id: int) -> Station | None:
        row = self._conn.execute(
            "SELECT id, name, camera_index, location FROM stations WHERE id = ?",
            (station_id,),
        ).fetchone()
        if row is None:
            return None
        return Station(
            id=row["id"],
            name=row["name"],
            camera_index=row["camera_index"],
            location=row["location"],
        )

    def get_station_by_name(self, name: str) -> Station | None:
        row = self._conn.execute(
            "SELECT id, name, camera_index, location FROM stations WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return Station(
            id=row["id"],
            name=row["name"],
            camera_index=row["camera_index"],
            location=row["location"],
        )

    # ── Employees ────────────────────────────────────────────────

    def upsert_employee(self, employee: Employee) -> Employee:
        with self._cursor() as cur:
            if employee.badge_id is not None:
                cur.execute(
                    "INSERT INTO employees (name, badge_id, consent) VALUES (?, ?, ?) "
                    "ON CONFLICT(badge_id) DO UPDATE SET "
                    "  name = excluded.name, "
                    "  consent = excluded.consent",
                    (employee.name, employee.badge_id, int(employee.consent)),
                )
                cur.execute(
                    "SELECT id FROM employees WHERE badge_id = ?",
                    (employee.badge_id,),
                )
            else:
                cur.execute(
                    "INSERT INTO employees (name, badge_id, consent) VALUES (?, ?, ?)",
                    (employee.name, None, int(employee.consent)),
                )
                cur.execute("SELECT last_insert_rowid() AS id")
            row = cur.fetchone()
        employee.id = row["id"]
        return employee

    def get_employee(self, employee_id: int) -> Employee | None:
        row = self._conn.execute(
            "SELECT id, name, badge_id, consent FROM employees WHERE id = ?",
            (employee_id,),
        ).fetchone()
        if row is None:
            return None
        return Employee(
            id=row["id"],
            name=row["name"],
            badge_id=row["badge_id"],
            consent=bool(row["consent"]),
        )

    def get_employee_by_badge(self, badge_id: str) -> Employee | None:
        row = self._conn.execute(
            "SELECT id, name, badge_id, consent FROM employees WHERE badge_id = ?",
            (badge_id,),
        ).fetchone()
        if row is None:
            return None
        return Employee(
            id=row["id"],
            name=row["name"],
            badge_id=row["badge_id"],
            consent=bool(row["consent"]),
        )

    # ── Shifts ───────────────────────────────────────────────────

    def start_shift(self, employee_id: int, station_id: int, start_time: float) -> Shift:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO shifts (employee_id, station_id, start_time) "
                "VALUES (?, ?, ?)",
                (employee_id, station_id, start_time),
            )
            shift_id = cur.lastrowid
        return Shift(
            id=shift_id,
            employee_id=employee_id,
            station_id=station_id,
            start_time=start_time,
        )

    def end_shift(self, shift_id: int, end_time: float) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE shifts SET end_time = ? WHERE id = ?",
                (end_time, shift_id),
            )

    def get_shift(self, shift_id: int) -> Shift | None:
        row = self._conn.execute(
            "SELECT id, employee_id, station_id, start_time, end_time "
            "FROM shifts WHERE id = ?",
            (shift_id,),
        ).fetchone()
        if row is None:
            return None
        return Shift(
            id=row["id"],
            employee_id=row["employee_id"],
            station_id=row["station_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
        )

    # ── Activity events ──────────────────────────────────────────

    def append_event(self, event: ActivityEvent) -> ActivityEvent:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO activity_events "
                "(shift_id, employee_id, station_id, state, start_time, end_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.shift_id,
                    event.employee_id,
                    event.station_id,
                    event.state.value,
                    event.start_time,
                    event.end_time,
                ),
            )
            event.id = cur.lastrowid
        return event

    def events_for_shift(self, shift_id: int) -> list[ActivityEvent]:
        rows = self._conn.execute(
            "SELECT id, shift_id, employee_id, station_id, state, start_time, end_time "
            "FROM activity_events WHERE shift_id = ? ORDER BY start_time ASC",
            (shift_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def events_in_range(
        self,
        start_time: float,
        end_time: float,
        employee_id: int | None = None,
        station_id: int | None = None,
        state: ActivityState | None = None,
    ) -> list[ActivityEvent]:
        clauses = ["start_time < ?", "end_time > ?"]
        params: list[object] = [end_time, start_time]
        if employee_id is not None:
            clauses.append("employee_id = ?")
            params.append(employee_id)
        if station_id is not None:
            clauses.append("station_id = ?")
            params.append(station_id)
        if state is not None:
            clauses.append("state = ?")
            params.append(state.value)
        sql = (
            "SELECT id, shift_id, employee_id, station_id, state, start_time, end_time "
            "FROM activity_events WHERE " + " AND ".join(clauses) +
            " ORDER BY start_time ASC"
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> ActivityEvent:
        return ActivityEvent(
            id=row["id"],
            shift_id=row["shift_id"],
            employee_id=row["employee_id"],
            station_id=row["station_id"],
            state=ActivityState.from_str(row["state"]),
            start_time=row["start_time"],
            end_time=row["end_time"],
        )
