"""Workstation activity monitoring.

Data model + persistent ledger for per-employee, per-station time-in-state
tracking. Consumes activity-state observations (from a classifier, or a
placeholder for Phase A) and writes contiguous state intervals to SQLite.

Phase B replaces the placeholder classifier with a trained model that emits
ActivityState labels (polishing, filing, tool_change, ...). The ledger and
schema below stay unchanged.
"""

from .models import (
    ActivityEvent,
    ActivityState,
    Employee,
    Shift,
    Station,
)
from .store import ActivityStore
from .ledger import ActivityLedger
from .assignment import IdentityResolver, StaticAssignment

__all__ = [
    "ActivityEvent",
    "ActivityLedger",
    "ActivityState",
    "ActivityStore",
    "Employee",
    "IdentityResolver",
    "Shift",
    "StaticAssignment",
    "Station",
]
