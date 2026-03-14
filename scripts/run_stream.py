#!/usr/bin/env python3
"""Run detection on a network stream (IP camera, RTSP, etc.).

This is a convenience wrapper around run_webcam.py for network sources.

Usage:
    python scripts/run_stream.py --url "http://192.168.1.10:8080/video"
    python scripts/run_stream.py --url "rtsp://user:pass@ip:554/stream"

For Android phones, install "IP Webcam" app and use:
    python scripts/run_stream.py --url "http://PHONE_IP:8080/video"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_webcam import main as webcam_main


def main():
    parser = argparse.ArgumentParser(description="Run detection on network stream")
    parser.add_argument("--url", required=True, help="Stream URL (HTTP MJPEG or RTSP)")
    parser.add_argument("--model", default="yolov8n.pt", help="Model name")
    parser.add_argument("--track", action="store_true", help="Enable tracking")
    args = parser.parse_args()

    # Rewrite sys.argv to pass to run_webcam
    sys.argv = [
        "run_stream.py",
        "--source", args.url,
        "--model", args.model,
    ]
    if args.track:
        sys.argv.append("--track")

    webcam_main()


if __name__ == "__main__":
    main()
