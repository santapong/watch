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

from src.models.yolo_wrapper import YOLODetector
from src.stream import VideoStream
from src.utils.drawing import draw_detections, draw_fps, draw_info
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
    detector = YOLODetector(
        model_name=model_name,
        confidence=confidence,
        iou_threshold=model_cfg.get("iou_threshold", 0.45),
        classes=model_cfg.get("classes"),
        device=model_cfg.get("device", ""),
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
