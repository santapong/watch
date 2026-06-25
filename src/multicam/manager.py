"""Multi-camera management and fusion.

Manages multiple VideoStream instances, displays them in a grid,
and supports cross-camera detection matching.
"""

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.models.base import BaseDetector, Detection
from src.stream import VideoStream
from src.utils.fps import FPSCounter


@dataclass
class CameraFeed:
    """A single camera feed with its stream and metadata."""

    name: str
    source: int | str
    stream: VideoStream | None = None
    fps_counter: FPSCounter = field(default_factory=FPSCounter)
    last_frame: np.ndarray | None = None
    last_detections: list[Detection] = field(default_factory=list)
    is_active: bool = True


class MultiCameraManager:
    """Manages multiple camera feeds with grid display and cross-camera detection.

    Example:
        manager = MultiCameraManager()
        manager.add_camera("webcam", 0)
        manager.add_camera("phone", "http://192.168.1.105:8080/video")
        manager.start()

        for grid_frame, all_detections in manager.process(detector):
            cv2.imshow("Multi-Camera", grid_frame)
    """

    def __init__(
        self,
        grid_cols: int | None = None,
        cell_size: tuple[int, int] = (640, 480),
    ):
        """Initialize multi-camera manager.

        Args:
            grid_cols: Number of columns in grid display (None = auto).
            cell_size: (width, height) of each camera cell in the grid.
        """
        self._cameras: dict[str, CameraFeed] = {}
        self._grid_cols = grid_cols
        self._cell_size = cell_size
        self._homographies: dict[str, np.ndarray] = {}

    def add_camera(
        self,
        name: str,
        source: int | str,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        """Add a camera source.

        Args:
            name: Unique camera identifier.
            source: Camera index or URL.
            resolution: Optional resolution override.
        """
        self._cameras[name] = CameraFeed(name=name, source=source)

    def start(self) -> None:
        """Start all camera streams."""
        for name, cam in self._cameras.items():
            try:
                cam.stream = VideoStream(source=cam.source)
                cam.is_active = True
            except RuntimeError as e:
                print(f"Warning: Cannot open camera '{name}' ({cam.source}): {e}")
                cam.is_active = False

    def stop(self) -> None:
        """Stop all camera streams."""
        for cam in self._cameras.values():
            if cam.stream is not None:
                cam.stream.release()
                cam.stream = None
            cam.is_active = False

    def read_all(self) -> dict[str, np.ndarray | None]:
        """Read the latest frame from each camera.

        Returns:
            Dict mapping camera name to frame (or None if unavailable).
        """
        frames = {}
        for name, cam in self._cameras.items():
            if cam.stream is not None and cam.is_active:
                frame = cam.stream.read()
                if frame is not None:
                    cam.last_frame = frame
                frames[name] = cam.last_frame
            else:
                frames[name] = None
        return frames

    def detect_all(
        self,
        detector: BaseDetector,
        tracking: bool = False,
    ) -> dict[str, list[Detection]]:
        """Run detection on all camera feeds.

        Args:
            detector: Object detector to use.
            tracking: Whether to use tracking mode.

        Returns:
            Dict mapping camera name to list of detections.
        """
        results = {}
        frames = self.read_all()

        for name, frame in frames.items():
            if frame is None:
                results[name] = []
                continue

            cam = self._cameras[name]
            cam.fps_counter.tick()

            if tracking:
                dets = detector.detect_and_track(frame)
            else:
                dets = detector.detect(frame)

            cam.last_detections = dets
            results[name] = dets

        return results

    def create_grid(
        self,
        frames: dict[str, np.ndarray | None],
        detections: dict[str, list[Detection]] | None = None,
    ) -> np.ndarray:
        """Create a grid view of all cameras.

        Args:
            frames: Dict of camera name to frame.
            detections: Optional detections to overlay info.

        Returns:
            Single combined grid frame.
        """
        active_cameras = [
            name for name, frame in frames.items() if frame is not None
        ]

        if not active_cameras:
            return np.zeros((*self._cell_size[::-1], 3), dtype=np.uint8)

        n = len(active_cameras)
        cols = self._grid_cols or math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        cell_w, cell_h = self._cell_size
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for idx, name in enumerate(active_cameras):
            row = idx // cols
            col = idx % cols
            frame = frames[name]

            if frame is None:
                continue

            # Resize to cell size
            cell = cv2.resize(frame, self._cell_size)

            # Draw camera name
            cv2.putText(
                cell,
                name,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Draw detection count
            if detections and name in detections:
                det_count = len(detections[name])
                cv2.putText(
                    cell,
                    f"Objects: {det_count}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw FPS
            cam = self._cameras[name]
            cv2.putText(
                cell,
                f"FPS: {cam.fps_counter.fps:.1f}",
                (cell_w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            grid[y1:y2, x1:x2] = cell

        return grid

    def set_homography(self, name: str, image_points, ground_points) -> bool:
        """Register a camera's image->ground homography from >=4 correspondences.

        Returns True if a valid homography was stored.
        """
        from src.multicam.geometry import build_homography

        H = build_homography(image_points, ground_points)
        if H is None:
            return False
        self._homographies[name] = H
        return True

    @property
    def has_homography(self) -> bool:
        """True when at least two cameras have a registered homography."""
        return len(self._homographies) >= 2

    def find_cross_camera_matches(
        self,
        detections: dict[str, list[Detection]],
        iou_threshold: float = 0.3,
        max_distance: float = 2.0,
    ) -> list[dict]:
        """Find the same physical object seen by multiple cameras.

        When >=2 cameras have a registered homography, detections are projected to
        a shared ground plane (BEV) and matched by Hungarian assignment on BEV
        distance (same class, within ``max_distance``). Otherwise this falls back
        to the legacy class+confidence heuristic, which cannot establish identity.

        Returns:
            List of match dicts. Geometric matches include camera/index pairs and
            ``bev_distance``.
        """
        geo_cams = [c for c in detections if c in self._homographies]
        if len(geo_cams) >= 2:
            return self._geometric_matches(detections, geo_cams, max_distance)
        return self._legacy_matches(detections)

    def _geometric_matches(self, detections, cams, max_distance):
        from src.multicam.geometry import match_bev, project_detections

        bev = {c: project_detections(self._homographies[c], detections[c]) for c in cams}
        matches = []
        for a in range(len(cams)):
            for b in range(a + 1, len(cams)):
                ca, cb = cams[a], cams[b]
                da, db = detections[ca], detections[cb]
                for i, j, dist in match_bev(
                    bev[ca], bev[cb],
                    [d.class_id for d in da], [d.class_id for d in db],
                    max_distance,
                ):
                    matches.append({
                        "camera_1": ca, "camera_2": cb,
                        "index_1": i, "index_2": j,
                        "class": da[i].class_name,
                        "bev_distance": dist,
                    })
        return matches

    @staticmethod
    def _legacy_matches(detections):
        matches = []
        camera_names = list(detections.keys())
        for i in range(len(camera_names)):
            for j in range(i + 1, len(camera_names)):
                cam1, cam2 = camera_names[i], camera_names[j]
                for d1 in detections[cam1]:
                    for d2 in detections[cam2]:
                        if d1.class_id == d2.class_id and abs(d1.confidence - d2.confidence) < 0.3:
                            matches.append({
                                "camera_1": cam1, "camera_2": cam2,
                                "class": d1.class_name,
                                "confidence_1": d1.confidence,
                                "confidence_2": d2.confidence,
                            })
        return matches

    def assign_global_ids(
        self,
        detections: dict[str, list[Detection]],
        max_distance: float = 2.0,
    ) -> dict[str, list[int]]:
        """Assign a shared global ID to each detection via cross-camera matches.

        Detections matched across cameras share an ID (union-find over the
        geometric matches). Returns per-camera ID lists aligned with ``detections``.
        """
        matches = self.find_cross_camera_matches(detections, max_distance=max_distance)
        parent: dict = {}

        def find(x):
            parent.setdefault(x, x)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for cam, dets in detections.items():
            for i in range(len(dets)):
                find((cam, i))
        for m in matches:
            if "index_1" in m:
                union((m["camera_1"], m["index_1"]), (m["camera_2"], m["index_2"]))

        root_to_id: dict = {}
        result: dict[str, list[int]] = {}
        for cam, dets in detections.items():
            ids = []
            for i in range(len(dets)):
                r = find((cam, i))
                if r not in root_to_id:
                    root_to_id[r] = len(root_to_id)
                ids.append(root_to_id[r])
            result[cam] = ids
        return result

    @property
    def camera_names(self) -> list[str]:
        return list(self._cameras.keys())

    @property
    def active_count(self) -> int:
        return sum(1 for cam in self._cameras.values() if cam.is_active)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
