"""Test configuration - stub unavailable optional dependencies."""

import sys
import types

# Stub supervision (used by zone_counter) if not installed
if "supervision" not in sys.modules:
    sv_mod = types.ModuleType("supervision")
    # Add commonly used attributes
    sv_mod.Detections = type("Detections", (), {"__init__": lambda self, **kw: None})
    sv_mod.PolygonZone = type("PolygonZone", (), {})
    sv_mod.LineZone = type("LineZone", (), {})
    sv_mod.PolygonZoneAnnotator = type("PolygonZoneAnnotator", (), {})
    sys.modules["supervision"] = sv_mod

# Stub transformers (used by open_vocab_detector) if not installed
if "transformers" not in sys.modules:
    trans_mod = types.ModuleType("transformers")
    trans_mod.Owlv2Processor = type("Owlv2Processor", (), {"from_pretrained": classmethod(lambda cls, *a, **kw: None)})
    trans_mod.Owlv2ForObjectDetection = type("Owlv2ForObjectDetection", (), {"from_pretrained": classmethod(lambda cls, *a, **kw: None)})
    sys.modules["transformers"] = trans_mod

# Stub streamlit if not installed
for mod_name in ["streamlit", "streamlit_webrtc"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Stub ultralytics (YOLO / RT-DETR loaders) if not installed, so modules that
# lazily `from ultralytics import YOLO/RTDETR` import cleanly. Tests that assert
# loader/export behavior override this with their own fake via monkeypatch.
if "ultralytics" not in sys.modules:
    ultra = types.ModuleType("ultralytics")

    class _StubModel:
        def __init__(self, *args, **kwargs):
            self.names = {}
            self.task = "detect"

        def predict(self, *args, **kwargs):
            return []

        def track(self, *args, **kwargs):
            return []

        def export(self, *args, **kwargs):
            return ""

    ultra.YOLO = _StubModel
    ultra.RTDETR = _StubModel
    sys.modules["ultralytics"] = ultra
