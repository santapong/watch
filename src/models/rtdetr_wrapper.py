"""RT-DETR detector wrapper using Ultralytics.

RT-DETR is a real-time transformer detector. In Ultralytics it loads through a
dedicated ``RTDETR`` class rather than ``YOLO``; everything else (inference,
tracking, result parsing) is identical to :class:`YOLODetector`, so this wrapper
overrides only the model-loading hook.
"""

from .yolo_wrapper import YOLODetector


class RTDETRDetector(YOLODetector):
    """Wraps Ultralytics RT-DETR for detection and tracking.

    Inherits ``detect``/``detect_and_track``/``_results_to_detections`` from
    :class:`YOLODetector`; RT-DETR results expose the same
    ``boxes.xyxy/conf/cls/id`` interface.

    Example:
        detector = RTDETRDetector(model_name="rtdetr-l.pt")
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        model_name: str = "rtdetr-l.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        device: str = "",
    ):
        """Initialize RT-DETR detector.

        Args:
            model_name: Model file or name (e.g., 'rtdetr-l.pt', 'rtdetr-x.pt').
            confidence: Minimum confidence threshold.
            iou_threshold: NMS IoU threshold.
            classes: Filter to specific class IDs (None = all classes).
            device: Device string ('' = auto, 'cpu', 'cuda:0', 'mps').
        """
        super().__init__(
            model_name=model_name,
            confidence=confidence,
            iou_threshold=iou_threshold,
            classes=classes,
            device=device,
        )

    def _load_model(self, model_name: str):
        """Load the underlying Ultralytics RT-DETR model (imported lazily)."""
        from ultralytics import RTDETR

        return RTDETR(model_name)
