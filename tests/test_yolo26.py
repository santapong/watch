"""Regression tests pinning the YOLO26 detector option (ultralytics mocked).

YOLO26 loads through Ultralytics' single ``YOLO`` class, so it reuses ``YOLODetector``;
these tests guard that the ``yolo26`` alias + factory routing stay intact. The general
factory contract lives in tests/test_registry.py — this file only pins YOLO26 specifics.
"""

import pytest

from src.models.registry import available_families, build_detector
from src.models.yolo_wrapper import YOLODetector


class _FakeModel:
    def __init__(self, name):
        self.name = name
        self.names = {0: "person"}


@pytest.fixture
def patch_yolo_loader(monkeypatch):
    """Mock the lazy ultralytics loader so no weights download and torch is never imported."""
    loaded = {}

    def fake_yolo(self, model_name):
        loaded["name"] = model_name
        return _FakeModel(model_name)

    monkeypatch.setattr(YOLODetector, "_load_model", fake_yolo)
    return loaded


def test_yolo26_in_available_families():
    assert "yolo26" in available_families()  # public API, not the private registry dict


def test_build_yolo26_by_name(patch_yolo_loader):
    det = build_detector({"name": "yolo26n.pt", "confidence": 0.3})
    assert isinstance(det, YOLODetector)
    assert det.model_name == "yolo26n.pt"
    assert det._confidence == 0.3
    assert patch_yolo_loader["name"] == "yolo26n.pt"  # the YOLO loader received the weights


def test_explicit_yolo26_backend_routes_to_yolo(patch_yolo_loader):
    # the explicit backend alias resolves even when the weights name lacks "yolo26"
    det = build_detector({"backend": "yolo26", "name": "custom-weights.pt"})
    assert isinstance(det, YOLODetector)
    assert det.model_name == "custom-weights.pt"
