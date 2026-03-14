"""FPS counter utility."""

import time
from collections import deque


class FPSCounter:
    """Sliding-window FPS counter for smooth display."""

    def __init__(self, window_size: int = 30):
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self):
        """Call once per frame."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
