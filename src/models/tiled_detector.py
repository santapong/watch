"""SAHI-style tiled inference wrapper.

``TiledDetector`` wraps any :class:`BaseDetector` and runs it over overlapping
tiles of the frame, offsets each tile's boxes back to full-frame coordinates, and
merges them with class-aware NMS. This recovers small / long-range objects a single
full-frame pass misses, at the cost of N forward passes — so it is opt-in
(``model.tiled``) and off by default.

Tracking is delegated to the wrapped detector on the full frame (per-tile track IDs
would not be consistent), so only ``detect`` is tiled.
"""

from collections import defaultdict

import numpy as np

from .base import BaseDetector, Detection


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TiledDetector(BaseDetector):
    """Slice-aided detector wrapper around another :class:`BaseDetector`.

    Args:
        base: The detector to run on each tile.
        tile_size: Square tile size in pixels.
        overlap: Fractional overlap between adjacent tiles (0-1).
        iou_threshold: IoU above which overlapping same-class boxes are merged.
    """

    def __init__(
        self,
        base: BaseDetector,
        tile_size: int = 640,
        overlap: float = 0.2,
        iou_threshold: float = 0.5,
    ):
        self._base = base
        self._tile = max(1, int(tile_size))
        self._overlap = min(max(float(overlap), 0.0), 0.9)
        self._iou = iou_threshold

    def _starts(self, length: int) -> list[int]:
        """Tile start offsets along one axis, ensuring the far edge is covered."""
        if length <= self._tile:
            return [0]
        step = max(1, int(self._tile * (1 - self._overlap)))
        starts = list(range(0, length - self._tile + 1, step))
        if not starts or starts[-1] != length - self._tile:
            starts.append(length - self._tile)
        return starts

    def _tiles(self, h: int, w: int):
        for y in self._starts(h):
            for x in self._starts(w):
                yield x, y, min(x + self._tile, w), min(y + self._tile, h)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        merged: list[Detection] = []
        for x1, y1, x2, y2 in self._tiles(h, w):
            tile = frame[y1:y2, x1:x2]
            for d in self._base.detect(tile):
                bx1, by1, bx2, by2 = d.bbox
                shifted = Detection(
                    bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    mask=d.mask,
                    track_id=d.track_id,
                )
                merged.append(shifted)
        return self._merge(merged)

    def _merge(self, detections: list[Detection]) -> list[Detection]:
        """Class-aware greedy NMS over the offset detections."""
        kept: list[Detection] = []
        by_class: dict[int, list[Detection]] = defaultdict(list)
        for d in detections:
            by_class[d.class_id].append(d)
        for group in by_class.values():
            group.sort(key=lambda d: d.confidence, reverse=True)
            class_kept: list[Detection] = []
            for d in group:
                if all(_iou(d.bbox, k.bbox) <= self._iou for k in class_kept):
                    class_kept.append(d)
            kept.extend(class_kept)
        return kept

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        # Tracking is delegated to the wrapped detector on the full frame; per-tile
        # track IDs would not be consistent across tiles.
        return self._base.detect_and_track(frame)

    @property
    def model_name(self) -> str:
        return f"tiled({self._base.model_name})"

    @property
    def class_names(self) -> dict[int, str]:
        return self._base.class_names
