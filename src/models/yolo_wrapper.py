"""YOLO detector wrapper using Ultralytics."""

import numpy as np

from .base import BaseDetector, Detection


class YOLODetector(BaseDetector):
    """Wraps Ultralytics YOLO for detection and tracking."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        device: str = "",
    ):
        """Initialize YOLO detector.

        Args:
            model_name: Model file or name (e.g., 'yolov8n.pt', 'yolo11n.pt').
            confidence: Minimum confidence threshold.
            iou_threshold: NMS IoU threshold.
            classes: Filter to specific class IDs (None = all classes).
            device: Device string ('' = auto, 'cpu', 'cuda:0', 'mps').
        """
        self._model = self._load_model(model_name)
        self._model_name = model_name
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._classes = classes
        self._device = device

    def _load_model(self, model_name: str):
        """Load the underlying Ultralytics model.

        Imported lazily so this module stays importable without ultralytics
        installed (keeps unit tests and CI light), and so subclasses can load a
        different model class by overriding just this hook (e.g. RT-DETR).
        """
        from ultralytics import YOLO

        return YOLO(model_name)

    def _results_to_detections(self, results) -> list[Detection]:
        """Convert Ultralytics results to Detection objects."""
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names = result.names

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                track_id = int(boxes.id[i].cpu()) if boxes.id is not None else None

                mask = None
                if result.masks is not None:
                    mask = result.masks.data[i].cpu().numpy()

                detections.append(
                    Detection(
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=names.get(cls_id, str(cls_id)),
                        mask=mask,
                        track_id=track_id,
                    )
                )
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self._confidence,
            iou=self._iou_threshold,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )
        return self._results_to_detections(results)

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame,
            conf=self._confidence,
            iou=self._iou_threshold,
            classes=self._classes,
            device=self._device,
            persist=True,
            verbose=False,
        )
        return self._results_to_detections(results)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def class_names(self) -> dict[int, str]:
        return self._model.names
