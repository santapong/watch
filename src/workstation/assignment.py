"""Camera → station → employee assignment.

Phase A uses a static map loaded from config: camera N is station N, and
each station has a single employee on shift. Phase B+ swaps in a real
``IdentityResolver`` (face/badge re-ID using the existing ReID embeddings
in src/tracking/tracker.py) without changing the ledger contract.
"""

from abc import ABC, abstractmethod

import numpy as np


class IdentityResolver(ABC):
    """Map a station observation to the employee currently working there."""

    @abstractmethod
    def resolve(
        self,
        station_id: int,
        frame: np.ndarray | None = None,
    ) -> int | None:
        """Return the employee_id at ``station_id``, or None if unknown."""


class StaticAssignment(IdentityResolver):
    """Fixed station_id → employee_id map. Default for Phase A."""

    def __init__(self, mapping: dict[int, int]):
        self._mapping = dict(mapping)

    def resolve(
        self,
        station_id: int,
        frame: np.ndarray | None = None,
    ) -> int | None:
        return self._mapping.get(station_id)

    def assign(self, station_id: int, employee_id: int) -> None:
        self._mapping[station_id] = employee_id

    def unassign(self, station_id: int) -> None:
        self._mapping.pop(station_id, None)

    def __contains__(self, station_id: int) -> bool:
        return station_id in self._mapping

    def __len__(self) -> int:
        return len(self._mapping)
