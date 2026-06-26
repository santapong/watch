#!/usr/bin/env python3
"""Run object detection with zone counting and line crossing.

Usage:
    python scripts/run_zone_counter.py
    python scripts/run_zone_counter.py --source "http://PHONE_IP:8080/video"
    python scripts/run_zone_counter.py --model yolov8s.pt --track

Controls:
    q     - Quit
    s     - Save screenshot
    r     - Reset counters
    space - Pause/resume
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.zone_counter import ZoneCounter, LineCrossCounter
from src.models.registry import build_detector_from_config
from src.stream import VideoStream
from src.utils.drawing import draw_detections, draw_fps, draw_info
from src.utils.fps import FPSCounter


def main():
    parser = argparse.ArgumentParser(description="Zone counting detection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--source", default=None)
    parser.add_argument("--track", action="store_true", default=True)
    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})

    # Initialize detector (tracking required for line crossing)
    detector = build_detector_from_config(config, model_name=args.model)

    # Initialize video
    source = args.source
    if source is None:
        source = config.get("source", {}).get("webcam_index", 0)
    else:
        try:
            source = int(source)
        except ValueError:
            pass

    stream = VideoStream(source=source)
    fps_counter = FPSCounter()

    w, h = stream.frame_size

    # Setup zone counter with example zones
    zone_counter = ZoneCounter(frame_resolution=(w, h))
    zone_cfg = config.get("zones", {})

    if zone_cfg.get("polygons"):
        for zone_def in zone_cfg["polygons"]:
            zone_counter.add_zone(
                zone_def["name"],
                np.array(zone_def["points"]),
                tuple(zone_def.get("color", [0, 255, 0])),
            )
    else:
        # Default: split frame into left/right zones
        zone_counter.add_zone(
            "Left Zone",
            np.array([[0, 0], [w // 2, 0], [w // 2, h], [0, h]]),
            (0, 255, 0),
        )
        zone_counter.add_zone(
            "Right Zone",
            np.array([[w // 2, 0], [w, 0], [w, h], [w // 2, h]]),
            (255, 0, 0),
        )

    # Setup line counter
    line_counter = LineCrossCounter()
    line_counter.add_line("Center Line", (w // 2, 0), (w // 2, h), (0, 255, 255))

    print("Zone counting started. Press 'q' to quit, 'r' to reset counters.")

    paused = False
    last_frame = None

    try:
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

            # Detect with tracking (required for line crossing)
            detections = detector.detect_and_track(frame)
            fps_counter.tick()

            # Count objects in zones
            counts = zone_counter.count(detections)

            # Update line crossings
            line_results = line_counter.update(detections)

            # Draw annotations
            draw_detections(frame, detections, show_confidence=True, show_track_id=True)
            zone_counter.annotate(frame)
            line_counter.annotate(frame)
            draw_fps(frame, fps_counter.fps)
            draw_info(frame, args.model, len(detections))

            # Draw zone counts
            y = 60
            for name, count in counts.items():
                cv2.putText(
                    frame, f"{name}: {count} objects", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
                )
                y += 25

            # Draw line crossing counts
            for name, result in line_results.items():
                cv2.putText(
                    frame, f"{name} - In: {result['in']} Out: {result['out']}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
                )
                y += 25

            cv2.imshow("Zone Counter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                line_counter.reset()
                print("Counters reset.")
            elif key == ord("s"):
                Path("screenshots").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshots/zone_{ts}.jpg", frame)
                print(f"Screenshot saved.")
            elif key == ord(" "):
                paused = not paused

    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
