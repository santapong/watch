#!/usr/bin/env python3
"""Run real-time object detection on webcam or video source.

Usage:
    python scripts/run_webcam.py                          # Default webcam
    python scripts/run_webcam.py --config configs/default.yaml
    python scripts/run_webcam.py --model yolov8s.pt       # Override model
    python scripts/run_webcam.py --source 1               # Camera index 1
    python scripts/run_webcam.py --source "http://..."    # IP camera URL
    python scripts/run_webcam.py --track                  # Enable tracking

Controls:
    q     - Quit
    s     - Save screenshot
    t     - Toggle tracking on/off
    c     - Toggle confidence display
    space - Pause/resume
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import build_detector_from_config
from src.stream import VideoStream
from src.tracking.tracker import EnhancedTracker
from src.utils.drawing import draw_detections, draw_fps, draw_info, draw_tracks
from src.utils.fps import FPSCounter


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_source(config: dict, cli_source: str | None):
    """Determine video source from config and CLI args."""
    if cli_source is not None:
        # Try to parse as int (webcam index)
        try:
            return int(cli_source)
        except ValueError:
            return cli_source

    src = config.get("source", {})
    src_type = src.get("type", "webcam")

    if src_type == "webcam":
        return src.get("webcam_index", 0)
    elif src_type == "url":
        return src.get("url", "")
    elif src_type == "file":
        return src.get("file_path", "")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Real-time object detection")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--source", default=None, help="Video source (camera index or URL)")
    parser.add_argument("--track", action="store_true", help="Enable tracking")
    parser.add_argument("--confidence", type=float, default=None, help="Confidence threshold")
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)

    model_cfg = config.get("model", {})
    display_cfg = config.get("display", {})

    # Initialize detector
    model_name = args.model or model_cfg.get("name", "yolov8n.pt")
    confidence = args.confidence or model_cfg.get("confidence", 0.25)

    print(f"Loading model: {model_name}")
    detector = build_detector_from_config(
        config, model_name=model_name, confidence=confidence
    )

    # Enhanced tracker (appearance re-ID + trajectory trails). Built unconditionally
    # so the runtime 't' toggle works; only used while tracking is enabled.
    trk_cfg = config.get("tracking", {})
    tracker = EnhancedTracker(
        max_history=trk_cfg.get("max_history", 50),
        reid_threshold=trk_cfg.get("reid_threshold", 0.7),
        lost_timeout=trk_cfg.get("lost_timeout", 5.0),
        reid_backend=trk_cfg.get("reid_backend", "auto"),
    )

    # Initialize video source
    source = get_source(config, args.source)
    resolution = config.get("source", {}).get("resolution")
    if resolution:
        resolution = tuple(resolution)

    print(f"Opening video source: {source}")
    stream = VideoStream(source=source, resolution=resolution)
    fps_counter = FPSCounter()

    # State
    tracking = args.track or config.get("tracking", {}).get("enabled", False)
    show_confidence = display_cfg.get("show_confidence", True)
    paused = False
    window_name = display_cfg.get("window_name", "Object Detection")

    # Screenshot directory
    screenshot_dir = Path("screenshots")

    print(f"Detection started. Tracking: {'ON' if tracking else 'OFF'}")
    print("Controls: q=quit, s=screenshot, t=toggle tracking, c=toggle confidence, space=pause")

    try:
        last_frame = None
        while stream.is_opened:
            if not paused:
                frame = stream.read()
                if frame is None:
                    continue
                last_frame = frame
            else:
                frame = last_frame
                if frame is None:
                    continue

            # Run detection
            if tracking:
                detections = detector.detect_and_track(frame)
                detections = tracker.update(detections, frame)
            else:
                detections = detector.detect(frame)

            fps_counter.tick()

            # Draw annotations
            draw_detections(
                frame, detections,
                show_confidence=show_confidence,
                show_track_id=tracking,
                thickness=display_cfg.get("bbox_thickness", 2),
                font_scale=display_cfg.get("font_scale", 0.6),
            )

            if tracking:
                draw_tracks(frame, tracker.get_all_trajectories())

            if display_cfg.get("show_fps", True):
                draw_fps(frame, fps_counter.fps)

            if display_cfg.get("show_model_info", True):
                draw_info(frame, model_name, len(detections))

            if paused:
                cv2.putText(
                    frame, "PAUSED", (frame.shape[1] // 2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
                )

            # Display
            cv2.imshow(window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                screenshot_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = screenshot_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Screenshot saved: {path}")
            elif key == ord("t"):
                tracking = not tracking
                print(f"Tracking: {'ON' if tracking else 'OFF'}")
            elif key == ord("c"):
                show_confidence = not show_confidence
                print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")
            elif key == ord(" "):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")

    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")


if __name__ == "__main__":
    main()
