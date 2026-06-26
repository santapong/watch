"""Tests for SAHI-style tiled inference (TiledDetector)."""

import numpy as np

from src.models.base import BaseDetector, Detection
from src.models.tiled_detector import TiledDetector


class _PatchDetector(BaseDetector):
    """Mock detector: returns one box around the bright pixels in the given tile."""

    def detect(self, frame):
        ys, xs = np.where(frame[:, :, 0] > 127)
        if len(xs) == 0:
            return []
        return [Detection(
            bbox=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
            confidence=0.9, class_id=0, class_name="obj",
        )]

    def detect_and_track(self, frame):
        return self.detect(frame)

    @property
    def model_name(self):
        return "mock"

    @property
    def class_names(self):
        return {0: "obj"}


def _frame_with_squares(squares, shape=(200, 200, 3)):
    f = np.zeros(shape, dtype=np.uint8)
    for (x1, y1, x2, y2) in squares:
        f[y1:y2, x1:x2] = 255
    return f


class TestTiledDetector:
    def test_offsets_boxes_to_global_coords(self):
        det = TiledDetector(_PatchDetector(), tile_size=120, overlap=0.5)
        frame = _frame_with_squares([(100, 100, 120, 120)])
        out = det.detect(frame)
        assert len(out) == 1  # deduped across the overlapping tiles that saw it
        assert out[0].bbox == (100.0, 100.0, 120.0, 120.0)

    def test_overlapping_duplicates_are_merged(self):
        det = TiledDetector(_PatchDetector(), tile_size=120, overlap=0.5)
        # This square is covered by all 9 tiles; without NMS we'd get 9 boxes.
        frame = _frame_with_squares([(100, 100, 120, 120)])
        raw = []
        for x1, y1, x2, y2 in det._tiles(200, 200):
            tile = frame[y1:y2, x1:x2]
            raw.extend(det._base.detect(tile))
        assert len(raw) > 1            # multiple tiles detect it
        assert len(det.detect(frame)) == 1  # merged to one

    def test_multiple_distinct_objects_kept(self):
        det = TiledDetector(_PatchDetector(), tile_size=120, overlap=0.5)
        frame = _frame_with_squares([(20, 20, 40, 40), (150, 150, 170, 170)])
        out = det.detect(frame)
        assert len(out) == 2
        boxes = sorted(d.bbox for d in out)
        assert boxes == [(20.0, 20.0, 40.0, 40.0), (150.0, 150.0, 170.0, 170.0)]

    def test_small_frame_single_tile(self):
        det = TiledDetector(_PatchDetector(), tile_size=120, overlap=0.2)
        frame = _frame_with_squares([(30, 30, 50, 50)], shape=(100, 100, 3))
        out = det.detect(frame)
        assert len(out) == 1
        assert out[0].bbox == (30.0, 30.0, 50.0, 50.0)

    def test_delegation(self):
        base = _PatchDetector()
        det = TiledDetector(base, tile_size=120)
        assert det.model_name == "tiled(mock)"
        assert det.class_names == {0: "obj"}
        # detect_and_track delegates to the base on the full frame.
        frame = _frame_with_squares([(100, 100, 120, 120)])
        assert len(det.detect_and_track(frame)) == 1


class TestRegistryTiling:
    def test_build_detector_wraps_when_tiled(self, monkeypatch):
        from src.models.registry import build_detector
        from src.models.yolo_wrapper import YOLODetector

        class _Fake:
            names = {0: "person"}

        monkeypatch.setattr(YOLODetector, "_load_model", lambda self, n: _Fake())
        det = build_detector({"name": "yolov8n.pt", "tiled": True, "tile_size": 320})
        assert isinstance(det, TiledDetector)
        assert isinstance(det._base, YOLODetector)

    def test_build_detector_not_wrapped_by_default(self, monkeypatch):
        from src.models.registry import build_detector
        from src.models.yolo_wrapper import YOLODetector

        class _Fake:
            names = {0: "person"}

        monkeypatch.setattr(YOLODetector, "_load_model", lambda self, n: _Fake())
        det = build_detector({"name": "yolov8n.pt"})
        assert isinstance(det, YOLODetector)
