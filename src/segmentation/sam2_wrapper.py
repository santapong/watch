"""SAM 2 promptable segmentation wrapper.

Drives Segment Anything 2 with the bounding boxes of existing detections and writes
a pixel mask back onto each ``Detection.mask``. The heavy model is imported lazily
(via Ultralytics' ``SAM``), so importing this module never requires it.

Note: SAM 2 weights are not bundled and cannot be exercised in the slim CI; see
requirements-phase2.txt. The wiring/structure here is unit-tested with the model
mocked.
"""

from __future__ import annotations

import numpy as np

from src.models.base import Detection
from src.segmentation.base import BaseSegmenter


class SAM2Segmenter(BaseSegmenter):
    """Box-prompted SAM 2 segmenter (Ultralytics backend, lazy-loaded)."""

    def __init__(self, model_name: str = "sam2_b.pt", device: str = ""):
        from ultralytics import SAM  # lazy heavy import

        self._model = SAM(model_name)
        self._model_name = model_name
        self._device = device

    def segment(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        boxes = [list(d.bbox) for d in detections]
        if not boxes:
            return detections
        results = self._model(frame, bboxes=boxes, device=self._device, verbose=False)
        masks = self._extract_masks(results)
        for det, mask in zip(detections, masks):
            det.mask = mask
        return detections

    @staticmethod
    def _extract_masks(results) -> list[np.ndarray | None]:
        """Pull per-instance masks out of an Ultralytics SAM result (order-aligned)."""
        out: list[np.ndarray | None] = []
        for result in results:
            if getattr(result, "masks", None) is None:
                continue
            for m in result.masks.data:
                out.append(np.asarray(m.cpu().numpy() if hasattr(m, "cpu") else m))
        return out

    @property
    def model_name(self) -> str:
        return self._model_name


def build_segmenter(cfg: dict) -> BaseSegmenter:
    """Build a segmenter from a config dict.

    Args:
        cfg: ``{"backend": "sam2", "model": "sam2_b.pt", "device": ""}``.

    Raises:
        ValueError: on an unknown backend.
    """
    cfg = dict(cfg or {})
    backend = (cfg.get("backend") or "sam2").strip().lower()
    if backend in ("sam2", "sam"):
        return SAM2Segmenter(model_name=cfg.get("model", "sam2_b.pt"), device=cfg.get("device", ""))
    raise ValueError(f"Unknown segmenter backend '{backend}'. Use 'sam2'.")
