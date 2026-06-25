"""Detector registry / factory.

Maps a model ``backend`` family string to a :class:`BaseDetector` wrapper class so
scripts can build a detector from config without hardcoding a specific wrapper.

YOLOv8 / v10 / v11 / YOLO26 all load through Ultralytics' single ``YOLO`` class, so
they share :class:`YOLODetector`; RT-DETR needs its own loader and wrapper.

Example:
    from src.models.registry import build_detector
    detector = build_detector({"name": "yolo11n.pt", "confidence": 0.3})
"""

from __future__ import annotations

from .base import BaseDetector
from .rtdetr_wrapper import RTDETRDetector
from .tiled_detector import TiledDetector
from .yolo_wrapper import YOLODetector

# family name -> wrapper class
_REGISTRY: dict[str, type[BaseDetector]] = {}


def register(family: str, wrapper: type[BaseDetector]) -> None:
    """Register a detector wrapper class under a family name (case-insensitive)."""
    _REGISTRY[family.lower()] = wrapper


def available_families() -> list[str]:
    """Return the sorted list of registered family names."""
    return sorted(_REGISTRY)


# --- Built-in families -------------------------------------------------------
# All YOLO generations alias to the same wrapper (Ultralytics infers the version
# from the weights). RT-DETR is the one family that needs a different loader.
register("yolo", YOLODetector)
for _alias in ("yolov8", "yolov9", "yolov10", "yolo11", "yolov11", "yolo12", "yolo26"):
    register(_alias, YOLODetector)
register("rtdetr", RTDETRDetector)


def _infer_family(name: str) -> str:
    """Infer the detector family from a model weights name.

    Args:
        name: Model file/name, e.g. ``"yolov8n.pt"`` or ``"rtdetr-l.pt"``.

    Returns:
        A registered family string. Defaults to ``"yolo"`` for anything that
        isn't recognizably RT-DETR.
    """
    stem = str(name).lower()
    if "rtdetr" in stem:
        return "rtdetr"
    return "yolo"


def build_detector(cfg: dict) -> BaseDetector:
    """Build a detector from a ``model`` config block.

    Args:
        cfg: A model config dict, e.g.
            ``{"backend": "", "name": "yolov8n.pt", "confidence": 0.25,
            "iou_threshold": 0.45, "classes": None, "device": ""}``.
            ``backend`` is optional: empty or missing means infer from ``name``.

    Returns:
        A :class:`BaseDetector` instance (defaults to YOLO + ``yolov8n.pt``).

    Raises:
        ValueError: if ``backend`` names a family that isn't registered.
    """
    cfg = dict(cfg or {})
    name = cfg.get("name") or "yolov8n.pt"
    family = (cfg.get("backend") or "").strip().lower() or _infer_family(name)
    if family not in _REGISTRY:
        raise ValueError(
            f"Unknown detector backend '{family}'. "
            f"Available families: {', '.join(available_families())}"
        )

    wrapper = _REGISTRY[family]
    kwargs: dict = {"model_name": name}
    for key in ("confidence", "iou_threshold", "classes", "device"):
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    detector: BaseDetector = wrapper(**kwargs)

    # Optional SAHI-style tiled inference (off by default to preserve full-frame FPS).
    if cfg.get("tiled"):
        detector = TiledDetector(
            detector,
            tile_size=cfg.get("tile_size", 640),
            overlap=cfg.get("tile_overlap", 0.2),
        )
    return detector


def build_detector_from_config(
    config: dict,
    *,
    model_name: str | None = None,
    confidence: float | None = None,
) -> BaseDetector:
    """Build a detector from a full app config, applying CLI overrides.

    Centralizes the ``config["model"]`` merge the run-scripts otherwise
    copy-paste. CLI overrides (``model_name`` / ``confidence``) win over YAML.

    Args:
        config: The full loaded config dict (expects a ``"model"`` block).
        model_name: Optional CLI override for the model name.
        confidence: Optional CLI override for the confidence threshold.

    Returns:
        A :class:`BaseDetector` instance.
    """
    model_cfg = dict(config.get("model", {})) if config else {}
    if model_name is not None:
        model_cfg["name"] = model_name
    if confidence is not None:
        model_cfg["confidence"] = confidence
    return build_detector(model_cfg)
