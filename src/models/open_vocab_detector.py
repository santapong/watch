"""Open-vocabulary object detection using HuggingFace OWLv2.

Detects objects from free-text descriptions without training on specific classes.
Example: "red fire extinguisher", "person wearing hard hat"
"""

import numpy as np
import torch
from PIL import Image

from .base import BaseDetector, Detection


class OpenVocabDetector(BaseDetector):
    """Zero-shot object detection using OWLv2 from HuggingFace.

    Accepts arbitrary text prompts and detects matching objects in frames.

    Example:
        detector = OpenVocabDetector(text_queries=["person", "red car", "dog"])
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        text_queries: list[str] | None = None,
        confidence: float = 0.1,
        device: str = "",
        model_name: str = "google/owlv2-base-patch16-ensemble",
    ):
        """Initialize OWLv2 detector.

        Args:
            text_queries: List of text descriptions to detect.
            confidence: Minimum confidence threshold.
            device: Device string ('' = auto, 'cpu', 'cuda:0').
            model_name: HuggingFace model name for OWLv2.
        """
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        self._model_name_str = model_name
        self._confidence = confidence
        self._text_queries = text_queries or ["object"]

        # Determine device
        if device:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._processor = Owlv2Processor.from_pretrained(model_name)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_name).to(
            self._device
        )
        self._model.eval()

    def set_queries(self, text_queries: list[str]) -> None:
        """Update the text queries for detection.

        Args:
            text_queries: New list of text descriptions to detect.
        """
        self._text_queries = text_queries

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run open-vocabulary detection on a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects for matched text queries.
        """
        # Convert BGR to RGB PIL Image
        rgb_frame = frame[:, :, ::-1]
        image = Image.fromarray(rgb_frame)

        # Process inputs
        inputs = self._processor(
            text=[self._text_queries], images=image, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([frame.shape[:2]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self._confidence,
        )[0]

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        for box, score, label_idx in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            class_name = self._text_queries[label_idx] if label_idx < len(self._text_queries) else f"class_{label_idx}"

            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(score),
                    class_id=int(label_idx),
                    class_name=class_name,
                )
            )

        return detections

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        """Open-vocab detector does not support native tracking.

        Returns detections without track IDs.
        """
        return self.detect(frame)

    @property
    def model_name(self) -> str:
        return f"OWLv2 ({self._model_name_str.split('/')[-1]})"

    @property
    def class_names(self) -> dict[int, str]:
        return {i: name for i, name in enumerate(self._text_queries)}
