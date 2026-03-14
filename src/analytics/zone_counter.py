"""Smart object counting with polygon zones and line-crossing counters.

Uses the supervision library for zone and line detection.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

from src.models.base import Detection


@dataclass
class ZoneConfig:
    """Configuration for a polygon zone."""

    name: str
    polygon: np.ndarray  # Shape (N, 2) of polygon vertices
    color: tuple[int, int, int] = (0, 255, 0)
    triggering_anchors: list[str] = field(
        default_factory=lambda: ["BOTTOM_CENTER"]
    )


@dataclass
class LineConfig:
    """Configuration for a line-crossing counter."""

    name: str
    start: tuple[int, int]
    end: tuple[int, int]
    color: tuple[int, int, int] = (0, 255, 255)


class ZoneCounter:
    """Count objects within polygon zones using supervision library.

    Example:
        zone = ZoneCounter(frame_resolution=(1280, 720))
        zone.add_zone("entrance", np.array([[100, 100], [400, 100], [400, 500], [100, 500]]))
        counts = zone.count(detections)
    """

    def __init__(self, frame_resolution: tuple[int, int]):
        """Initialize zone counter.

        Args:
            frame_resolution: (width, height) of the video frame.
        """
        self._resolution = frame_resolution
        self._zones: dict[str, ZoneConfig] = {}
        self._sv_zones: dict[str, sv.PolygonZone] = {}
        self._zone_annotators: dict[str, sv.PolygonZoneAnnotator] = {}

    def add_zone(
        self,
        name: str,
        polygon: np.ndarray | list,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Add a polygon zone for counting.

        Args:
            name: Zone identifier.
            polygon: Array of (x, y) vertices defining the polygon.
            color: BGR color for visualization.
        """
        polygon = np.array(polygon, dtype=np.int32)
        config = ZoneConfig(name=name, polygon=polygon, color=color)
        self._zones[name] = config

        sv_zone = sv.PolygonZone(polygon=polygon)
        self._sv_zones[name] = sv_zone
        self._zone_annotators[name] = sv.PolygonZoneAnnotator(
            color=sv.Color(*color),
            thickness=2,
        )

    def _detections_to_sv(self, detections: list[Detection]) -> sv.Detections:
        """Convert our Detection objects to supervision Detections."""
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)
        tracker_id = np.array(
            [d.track_id if d.track_id is not None else -1 for d in detections],
            dtype=int,
        )

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id if any(t >= 0 for t in tracker_id) else None,
        )

    def count(self, detections: list[Detection]) -> dict[str, int]:
        """Count objects in each zone.

        Args:
            detections: List of Detection objects from a detector.

        Returns:
            Dict mapping zone name to count of objects inside.
        """
        sv_dets = self._detections_to_sv(detections)
        counts = {}
        for name, zone in self._sv_zones.items():
            mask = zone.trigger(detections=sv_dets)
            counts[name] = int(mask.sum())
        return counts

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw zone overlays on the frame.

        Args:
            frame: BGR image to annotate.

        Returns:
            Annotated frame.
        """
        for name, zone in self._sv_zones.items():
            annotator = self._zone_annotators[name]
            frame = annotator.annotate(scene=frame, polygon_zone=zone)
        return frame

    @property
    def zone_names(self) -> list[str]:
        return list(self._zones.keys())


class LineCrossCounter:
    """Count objects crossing a line boundary.

    Tracks objects crossing a defined line and counts in/out directions.

    Example:
        counter = LineCrossCounter()
        counter.add_line("door", (100, 300), (500, 300))
        result = counter.update(detections)
    """

    def __init__(self):
        self._lines: dict[str, LineConfig] = {}
        self._sv_lines: dict[str, sv.LineZone] = {}
        self._line_annotators: dict[str, sv.LineZoneAnnotator] = {}

    def add_line(
        self,
        name: str,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int] = (0, 255, 255),
    ) -> None:
        """Add a line for crossing detection.

        Args:
            name: Line identifier.
            start: (x, y) start point.
            end: (x, y) end point.
            color: BGR color for visualization.
        """
        config = LineConfig(name=name, start=start, end=end, color=color)
        self._lines[name] = config

        sv_line = sv.LineZone(
            start=sv.Point(*start),
            end=sv.Point(*end),
        )
        self._sv_lines[name] = sv_line
        self._line_annotators[name] = sv.LineZoneAnnotator(
            thickness=2,
            color=sv.Color(*color),
        )

    def update(
        self, detections: list[Detection]
    ) -> dict[str, dict[str, int]]:
        """Update line counters with new detections.

        Args:
            detections: List of Detection objects (must have track_id for crossing detection).

        Returns:
            Dict mapping line name to {"in": count, "out": count}.
        """
        if not detections:
            return {name: {"in": 0, "out": 0} for name in self._lines}

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)
        tracker_id = np.array(
            [d.track_id if d.track_id is not None else -1 for d in detections],
            dtype=int,
        )

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )

        results = {}
        for name, line in self._sv_lines.items():
            line.trigger(detections=sv_dets)
            results[name] = {
                "in": line.in_count,
                "out": line.out_count,
            }
        return results

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw line overlays and counts on the frame."""
        for name, line in self._sv_lines.items():
            annotator = self._line_annotators[name]
            frame = annotator.annotate(frame=frame, line_counter=line)
        return frame

    def reset(self) -> None:
        """Reset all crossing counters to zero."""
        for line in self._sv_lines.values():
            line.in_count = 0
            line.out_count = 0

    @property
    def line_names(self) -> list[str]:
        return list(self._lines.keys())
