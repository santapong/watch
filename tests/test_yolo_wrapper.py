"""Tests for YOLODetector result conversion (_results_to_detections).

Covers the Ultralytics-results -> Detection mapping with fake tensor-like objects,
so no real model or torch is needed.
"""

import numpy as np
import pytest

from src.models.yolo_wrapper import YOLODetector


class _T:
    """Minimal tensor-like supporting .cpu()/.numpy()/float()/int()."""

    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


class _Boxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy = [_T(b) for b in xyxy]
        self.conf = [_T(c) for c in conf]
        self.cls = [_T(c) for c in cls]
        self.id = None if ids is None else [_T(i) for i in ids]

    def __len__(self):
        return len(self.xyxy)


class _Result:
    def __init__(self, boxes, names, masks=None):
        self.boxes = boxes
        self.names = names
        self.masks = masks


class _Detector(YOLODetector):
    """YOLODetector with the model loader stubbed out (no ultralytics needed)."""

    def _load_model(self, model_name):
        class _M:
            names = {0: "person", 2: "car"}

        return _M()


def test_results_to_detections_basic():
    det = _Detector(model_name="x.pt")
    boxes = _Boxes(xyxy=[[10, 20, 110, 220]], conf=[0.9], cls=[0])
    result = _Result(boxes, names={0: "person"})

    out = det._results_to_detections([result])

    assert len(out) == 1
    d = out[0]
    assert d.bbox == (10.0, 20.0, 110.0, 220.0)
    assert d.confidence == pytest.approx(0.9)
    assert d.class_id == 0
    assert d.class_name == "person"
    assert d.track_id is None
    assert d.mask is None


def test_results_to_detections_with_track_id():
    det = _Detector(model_name="x.pt")
    boxes = _Boxes(xyxy=[[0, 0, 10, 10]], conf=[0.5], cls=[2], ids=[7])
    result = _Result(boxes, names={2: "car"})

    out = det._results_to_detections([result])

    assert out[0].track_id == 7
    assert out[0].class_name == "car"


def test_results_to_detections_handles_no_boxes():
    det = _Detector(model_name="x.pt")
    result = _Result(boxes=None, names={})

    assert det._results_to_detections([result]) == []


def test_results_to_detections_unknown_class_falls_back_to_id():
    det = _Detector(model_name="x.pt")
    boxes = _Boxes(xyxy=[[1, 2, 3, 4]], conf=[0.7], cls=[99])
    result = _Result(boxes, names={0: "person"})  # 99 not in names

    out = det._results_to_detections([result])

    assert out[0].class_name == "99"
