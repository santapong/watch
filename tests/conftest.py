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
