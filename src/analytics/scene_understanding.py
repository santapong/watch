"""Scene understanding and context analysis.

Goes beyond "what objects are here" to "what is happening" by analyzing
spatial relationships between objects and classifying scene types.
"""

from dataclasses import dataclass

import numpy as np

from src.models.base import Detection


@dataclass
class ObjectRelation:
    """A spatial relationship between two detected objects."""

    object_a: str
    object_b: str
    relation: str  # "near", "above", "below", "left_of", "right_of", "inside", "contains"
    distance: float  # Pixel distance between centers
    confidence: float


@dataclass
class SceneDescription:
    """Complete scene analysis result."""

    scene_type: str  # "indoor", "outdoor", "traffic", "workspace", etc.
    scene_confidence: float
    object_summary: dict[str, int]  # class_name -> count
    relations: list[ObjectRelation]
    description: str  # Natural language description


# Scene type definitions based on object composition
SCENE_SIGNATURES = {
    "traffic": {"car", "truck", "bus", "traffic light", "stop sign", "motorcycle"},
    "indoor_living": {"couch", "tv", "remote", "chair", "dining table", "vase"},
    "office": {"laptop", "keyboard", "mouse", "monitor", "cell phone", "book"},
    "kitchen": {"oven", "microwave", "refrigerator", "sink", "cup", "bowl", "knife", "fork", "spoon"},
    "outdoor_urban": {"person", "car", "bicycle", "bench", "parking meter", "fire hydrant"},
    "sports": {"sports ball", "tennis racket", "baseball bat", "skateboard", "surfboard", "frisbee"},
    "dining": {"dining table", "cup", "wine glass", "fork", "knife", "spoon", "bowl"},
    "nature": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"},
    "workspace": {"person", "laptop", "cell phone", "book", "chair", "backpack"},
}

# Distance thresholds (relative to frame diagonal)
NEAR_THRESHOLD = 0.15
FAR_THRESHOLD = 0.4


class SceneAnalyzer:
    """Analyzes scenes by examining detected objects and their relationships.

    Example:
        analyzer = SceneAnalyzer()
        description = analyzer.analyze(detections, frame_shape=(720, 1280))
        print(description.scene_type)
        print(description.description)
    """

    def __init__(self, relation_distance_threshold: float = 0.2):
        """Initialize scene analyzer.

        Args:
            relation_distance_threshold: Max relative distance for "near" relation.
        """
        self._near_threshold = relation_distance_threshold

    def analyze(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int] = (720, 1280),
    ) -> SceneDescription:
        """Perform full scene analysis.

        Args:
            detections: List of detected objects.
            frame_shape: (height, width) of the frame.

        Returns:
            SceneDescription with type, relations, and natural language description.
        """
        # Count objects by class
        object_summary = {}
        for det in detections:
            object_summary[det.class_name] = object_summary.get(det.class_name, 0) + 1

        # Classify scene type
        scene_type, scene_conf = self._classify_scene(object_summary)

        # Find spatial relations
        relations = self._find_relations(detections, frame_shape)

        # Generate description
        description = self._generate_description(
            object_summary, scene_type, relations
        )

        return SceneDescription(
            scene_type=scene_type,
            scene_confidence=scene_conf,
            object_summary=object_summary,
            relations=relations,
            description=description,
        )

    def _classify_scene(
        self, object_summary: dict[str, int]
    ) -> tuple[str, float]:
        """Classify the scene type based on detected objects.

        Returns:
            Tuple of (scene_type, confidence).
        """
        if not object_summary:
            return "empty", 1.0

        detected_classes = set(object_summary.keys())
        best_scene = "general"
        best_score = 0.0

        for scene_type, signature_classes in SCENE_SIGNATURES.items():
            overlap = detected_classes & signature_classes
            if not overlap:
                continue

            # Score based on overlap ratio
            score = len(overlap) / len(signature_classes)
            # Boost score if more instances of matching classes
            instance_boost = sum(
                object_summary.get(cls, 0) for cls in overlap
            ) / (sum(object_summary.values()) + 1e-6)
            combined_score = 0.6 * score + 0.4 * instance_boost

            if combined_score > best_score:
                best_score = combined_score
                best_scene = scene_type

        return best_scene, min(best_score, 1.0)

    def _find_relations(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[ObjectRelation]:
        """Find spatial relationships between detected objects.

        Args:
            detections: List of detections.
            frame_shape: (height, width) for distance normalization.

        Returns:
            List of ObjectRelation objects.
        """
        if len(detections) < 2:
            return []

        h, w = frame_shape
        diagonal = (h**2 + w**2) ** 0.5
        relations = []

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                det_a = detections[i]
                det_b = detections[j]

                # Calculate center distance
                cx_a, cy_a = det_a.center
                cx_b, cy_b = det_b.center
                distance = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
                rel_distance = distance / diagonal

                if rel_distance > FAR_THRESHOLD:
                    continue  # Too far apart for meaningful relation

                # Determine spatial relation
                relation = self._determine_relation(det_a, det_b, rel_distance)
                confidence = max(0.0, 1.0 - rel_distance / FAR_THRESHOLD)

                relations.append(
                    ObjectRelation(
                        object_a=det_a.class_name,
                        object_b=det_b.class_name,
                        relation=relation,
                        distance=distance,
                        confidence=confidence,
                    )
                )

        # Sort by confidence
        relations.sort(key=lambda r: r.confidence, reverse=True)

        # Limit to top 10 most significant relations
        return relations[:10]

    def _determine_relation(
        self,
        det_a: Detection,
        det_b: Detection,
        rel_distance: float,
    ) -> str:
        """Determine the spatial relation between two objects."""
        cx_a, cy_a = det_a.center
        cx_b, cy_b = det_b.center

        dx = cx_b - cx_a
        dy = cy_b - cy_a

        # Check containment
        a_inside_b = (
            det_a.bbox[0] >= det_b.bbox[0]
            and det_a.bbox[1] >= det_b.bbox[1]
            and det_a.bbox[2] <= det_b.bbox[2]
            and det_a.bbox[3] <= det_b.bbox[3]
        )
        b_inside_a = (
            det_b.bbox[0] >= det_a.bbox[0]
            and det_b.bbox[1] >= det_a.bbox[1]
            and det_b.bbox[2] <= det_a.bbox[2]
            and det_b.bbox[3] <= det_a.bbox[3]
        )

        if a_inside_b:
            return "inside"
        if b_inside_a:
            return "contains"

        if rel_distance < NEAR_THRESHOLD:
            return "near"

        # Determine directional relation
        if abs(dy) > abs(dx):
            return "above" if dy > 0 else "below"
        else:
            return "right_of" if dx > 0 else "left_of"

    def _generate_description(
        self,
        object_summary: dict[str, int],
        scene_type: str,
        relations: list[ObjectRelation],
    ) -> str:
        """Generate a natural language description of the scene."""
        if not object_summary:
            return "Empty scene with no detected objects."

        parts = []

        # Scene type
        scene_labels = {
            "traffic": "a traffic scene",
            "indoor_living": "an indoor living space",
            "office": "an office environment",
            "kitchen": "a kitchen area",
            "outdoor_urban": "an urban outdoor scene",
            "sports": "a sports scene",
            "dining": "a dining scene",
            "nature": "a nature scene",
            "workspace": "a workspace",
            "general": "a general scene",
            "empty": "an empty scene",
        }
        parts.append(f"This appears to be {scene_labels.get(scene_type, 'a scene')}.")

        # Object listing
        total = sum(object_summary.values())
        obj_descriptions = []
        for cls_name, count in sorted(
            object_summary.items(), key=lambda x: x[1], reverse=True
        ):
            if count == 1:
                obj_descriptions.append(f"a {cls_name}")
            else:
                obj_descriptions.append(f"{count} {cls_name}s")

        if obj_descriptions:
            if len(obj_descriptions) <= 3:
                obj_text = ", ".join(obj_descriptions)
            else:
                obj_text = (
                    ", ".join(obj_descriptions[:3])
                    + f", and {len(obj_descriptions) - 3} other types"
                )
            parts.append(f"Detected {total} objects: {obj_text}.")

        # Key relations
        if relations:
            key_relations = relations[:3]
            rel_texts = []
            for rel in key_relations:
                rel_texts.append(
                    f"{rel.object_a} is {rel.relation} {rel.object_b}"
                )
            parts.append("Spatial relations: " + "; ".join(rel_texts) + ".")

        return " ".join(parts)
