"""Tests for INT8 / quantization argument plumbing in ModelExporter.

These never run a real export — a fake ``ultralytics`` module records the kwargs
passed to ``model.export`` so we can assert the contract.
"""

import sys
import types

from src.deployment.exporter import ModelExporter


def _install_fake_ultralytics(monkeypatch, export_path):
    """Install a fake ultralytics whose model.export records its kwargs."""
    calls = {}

    class FakeModel:
        def __init__(self, path):
            self.path = path
            self.names = {}
            self.task = "detect"

        def export(self, **kwargs):
            calls["export_kwargs"] = kwargs
            return export_path

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeModel
    fake.RTDETR = FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    return calls


def test_export_passes_int8_and_data(monkeypatch, tmp_path):
    out = tmp_path / "m.onnx"
    out.write_bytes(b"x" * 2048)
    calls = _install_fake_ultralytics(monkeypatch, str(out))

    result = ModelExporter("yolov8n.pt").export(format="onnx", int8=True, data="coco128.yaml")

    assert result.success
    assert calls["export_kwargs"]["int8"] is True
    assert calls["export_kwargs"]["data"] == "coco128.yaml"
    assert result.precision == "int8"
    assert result.calibration_data == "coco128.yaml"


def test_export_int8_without_data_omits_data_kwarg(monkeypatch, tmp_path):
    out = tmp_path / "m.onnx"
    out.write_bytes(b"x" * 16)
    calls = _install_fake_ultralytics(monkeypatch, str(out))

    result = ModelExporter("yolov8n.pt").export(format="onnx", int8=True)

    assert result.success
    assert calls["export_kwargs"]["int8"] is True
    assert "data" not in calls["export_kwargs"]  # omitted when None


def test_half_and_int8_mutually_exclusive(monkeypatch, tmp_path):
    out = tmp_path / "m.onnx"
    out.write_bytes(b"x")
    calls = _install_fake_ultralytics(monkeypatch, str(out))

    result = ModelExporter("yolov8n.pt").export(format="onnx", half=True, int8=True)

    assert result.success is False
    assert "mutually exclusive" in result.error_message
    assert "export_kwargs" not in calls  # export() must never be called


def test_export_default_is_fp32(monkeypatch, tmp_path):
    out = tmp_path / "m.onnx"
    out.write_bytes(b"x" * 32)
    calls = _install_fake_ultralytics(monkeypatch, str(out))

    result = ModelExporter("yolov8n.pt").export(format="onnx")

    assert result.success
    assert calls["export_kwargs"]["int8"] is False
    assert calls["export_kwargs"]["half"] is False
    assert result.precision == "fp32"


def test_export_fp16_precision_label(monkeypatch, tmp_path):
    out = tmp_path / "m.onnx"
    out.write_bytes(b"x" * 32)
    _install_fake_ultralytics(monkeypatch, str(out))

    result = ModelExporter("yolov8n.pt").export(format="onnx", half=True)

    assert result.precision == "fp16"
