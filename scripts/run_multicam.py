#!/usr/bin/env python3
"""Run multi-camera detection with grid display.

Usage:
    python scripts/run_multicam.py --sources 0 1
    python scripts/run_multicam.py --sources 0 "http://PHONE_IP:8080/video"
    python scripts/run_multicam.py --sources 0 1 "rtsp://admin:pass@ip:554/stream"

Controls:
    q     - Quit
    s     - Save screenshot
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import build_detector_from_config
from src.multicam.manager import MultiCameraManager
from src.utils.drawing import draw_detections


def main():
    parser = argparse.ArgumentParser(description="Multi-camera detection")
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="Video sources (camera indices or URLs)",
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--cell-size", nargs=2, type=int, default=[640, 480])
    args = parser.parse_args()

    detector = build_detector_from_config({}, model_name=args.model)

    manager = MultiCameraManager(cell_size=tuple(args.cell_size))

    for i, source in enumerate(args.sources):
        try:
            source = int(source)
        except ValueError:
            pass
        name = f"Camera {i}" if isinstance(source, int) else f"Stream {i}"
        manager.add_camera(name, source)

    print(f"Starting {len(args.sources)} cameras...")

    with manager:
        print(f"Active cameras: {manager.active_count}/{len(args.sources)}")
        print("Multi-camera view started. Press 'q' to quit.")

        while True:
            # Read and detect
            frames = manager.read_all()
            all_detections = manager.detect_all(detector, tracking=args.track)

            # Draw detections on each frame
            for name, frame in frames.items():
                if frame is not None and name in all_detections:
                    draw_detections(frame, all_detections[name])

            # Create grid view
            grid = manager.create_grid(frames, all_detections)

            cv2.imshow("Multi-Camera View", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                Path("screenshots").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshots/multicam_{ts}.jpg", grid)
                print("Screenshot saved.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
