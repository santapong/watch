"""Video source abstraction for webcam, IP camera, RTSP, and video files."""

import threading
import time

import cv2
import numpy as np


class VideoStream:
    """Thread-safe video stream that always provides the latest frame.

    Supports:
        - Webcam: VideoStream(0) or VideoStream(1)
        - IP camera (MJPEG): VideoStream("http://192.168.1.10:8080/video")
        - RTSP: VideoStream("rtsp://user:pass@ip:554/stream")
        - Video file: VideoStream("path/to/video.mp4")

    Uses a background thread to continuously grab frames, so the consumer
    always gets the latest frame without buffer lag (critical for IP cameras).
    """

    def __init__(self, source: int | str = 0, resolution: tuple[int, int] | None = None):
        """Initialize video stream.

        Args:
            source: Camera index (int) or URL/path (str).
            resolution: Optional (width, height) to set camera resolution.
        """
        self._source = source
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        if resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stopped = False

        # Read first frame
        ret, frame = self._cap.read()
        if ret:
            self._frame = frame

        # Start background reader thread
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        """Continuously grab frames in background to avoid buffer lag."""
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                # For video files, this means end of file
                if isinstance(self._source, str) and not self._source.startswith(("http", "rtsp")):
                    self._stopped = True
                    break
                # For streams, retry after short delay
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        """Get the latest frame.

        Returns:
            BGR frame as numpy array, or None if no frame available.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_opened(self) -> bool:
        return not self._stopped and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return (width, height) of the video source."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def release(self):
        """Stop the stream and release resources."""
        self._stopped = True
        self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __iter__(self):
        """Iterate over frames. Yields latest frame on each call."""
        while self.is_opened:
            frame = self.read()
            if frame is not None:
                yield frame
            else:
                time.sleep(0.01)

    def __del__(self):
        if hasattr(self, "_stopped") and not self._stopped:
            self.release()
