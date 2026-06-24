"""Tests for the detector registry / factory (src/models/registry.py)."""

import pytest

from src.models.registry import (
    _infer_family,
    available_families,
    build_detector,
    build_detector_from_config,
)
from src.models.rtdetr_wrapper import RTDETRDetector
from src.models.yolo_wrapper import YOLODetector


class _FakeModel:
    def __init__(self, name):
        self.name = name
        self.names = {0: "person"}


@pytest.fixture
def patch_loaders(monkeypatch):
    """Patch both wrappers' _load_model so no real model is ever downloaded."""
    loaded = {}

    def fake_yolo(self, model_name):
        loaded["yolo"] = model_name
        return _FakeModel(model_name)

    def fake_rtdetr(self, model_name):
        loaded["rtdetr"] = model_name
        return _FakeModel(model_name)

    monkeypatch.setattr(YOLODetector, "_load_model", fake_yolo)
    monkeypatch.setattr(RTDETRDetector, "_load_model", fake_rtdetr)
    return loaded


def test_infer_family_yolo():
    assert _infer_family("yolov8n.pt") == "yolo"
    assert _infer_family("yolo11n.pt") == "yolo"
    assert _infer_family("yolo26n.pt") == "yolo"


def test_infer_family_rtdetr():
    assert _infer_family("rtdetr-l.pt") == "rtdetr"
    assert _infer_family("RTDETR-X.pt") == "rtdetr"


def test_available_families_includes_builtins():
    fams = available_families()
    assert "yolo" in fams
    assert "rtdetr" in fams


def test_default_is_yolo_yolov8n(patch_loaders):
    det = build_detector({})
    assert isinstance(det, YOLODetector)
    assert det.model_name == "yolov8n.pt"


def test_build_forwards_kwargs(patch_loaders):
    det = build_detector(
        {
            "name": "yolo11n.pt",
            "confidence": 0.4,
            "iou_threshold": 0.6,
            "classes": [0],
            "device": "cpu",
        }
    )
    assert isinstance(det, YOLODetector)
    assert det.model_name == "yolo11n.pt"
    assert det._confidence == 0.4
    assert det._iou_threshold == 0.6
    assert det._classes == [0]
    assert det._device == "cpu"


def test_rtdetr_uses_rtdetr_wrapper(patch_loaders):
    det = build_detector({"name": "rtdetr-l.pt"})
    assert isinstance(det, RTDETRDetector)
    assert patch_loaders.get("rtdetr") == "rtdetr-l.pt"
    assert patch_loaders.get("yolo") is None  # the YOLO loader was not used


def test_explicit_backend_overrides_inference(patch_loaders):
    det = build_detector({"backend": "rtdetr", "name": "weird-name.pt"})
    assert isinstance(det, RTDETRDetector)


def test_unknown_backend_raises():
    with pytest.raises(ValueError) as exc:
        build_detector({"backend": "does-not-exist"})
    assert "does-not-exist" in str(exc.value)


def test_build_from_config_cli_overrides(patch_loaders):
    config = {"model": {"name": "yolov8n.pt", "confidence": 0.25}}
    det = build_detector_from_config(config, model_name="yolo11n.pt", confidence=0.5)
    assert det.model_name == "yolo11n.pt"
    assert det._confidence == 0.5


def test_build_from_config_empty(patch_loaders):
    det = build_detector_from_config({}, model_name="yolov8n.pt")
    assert isinstance(det, YOLODetector)
    assert det.model_name == "yolov8n.pt"
