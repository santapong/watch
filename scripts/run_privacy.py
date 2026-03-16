"""Run object detection with privacy mode enabled.

Automatically blurs or pixelates detected persons/faces for
GDPR/CCPA compliance while preserving analytics capability.

Usage:
    python scripts/run_privacy.py
    python scripts/run_privacy.py --mode pixelate --target person
    python scripts/run_privacy.py --mode blur --blur-strength 71

Controls:
    q - Quit
    m - Cycle through modes (blur/pixelate/blackout)
    t - Toggle target (person/face/all)
    s - Screenshot
"""

import argparse

import cv2
import yaml

from src.stream import VideoStream
from src.models.yolo_wrapper import YOLODetector
from src.privacy import PrivacyFilter
from src.utils.drawing import draw_fps, draw_info
from src.utils.fps import FPSCounter


def parse_args():
    parser = argparse.ArgumentParser(description="Detection with privacy mode")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--source", default=None, help="Video source")
    parser.add_argument("--model", default=None, help="YOLO model name")
    parser.add_argument("--mode", default="blur", choices=["blur", "pixelate", "blackout"])
    parser.add_argument("--target", default="person", choices=["person", "face", "all"])
    parser.add_argument("--blur-strength", type=int, default=51, help="Blur kernel size")
    parser.add_argument("--pixel-size", type=int, default=15, help="Pixelation block size")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = args.model or config["model"]["name"]
    source = args.source or config["source"]["webcam_index"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    detector = YOLODetector(
        model_name=model_name,
        confidence=config["model"]["confidence"],
        device=config["model"]["device"],
    )

    stream = VideoStream(source)
    fps_counter = FPSCounter()

    privacy = PrivacyFilter(
        mode=args.mode,
        target=args.target,
        blur_strength=args.blur_strength,
        pixel_size=args.pixel_size,
    )

    modes = ["blur", "pixelate", "blackout"]
    targets = ["person", "face", "all"]

    print(f"Privacy mode: {privacy.mode} | Target: {privacy.target}")
    print("Controls: q=quit, m=cycle mode, t=cycle target, s=screenshot")

    while True:
        frame = stream.read()
        if frame is None:
            break

        detections = detector.detect(frame)
        fps_counter.tick()

        # Apply privacy filter
        frame = privacy.apply(frame, detections)

        draw_fps(frame, fps_counter.fps)
        draw_info(frame, detector.model_name, len(detections))

        # Show current privacy settings
        cv2.putText(
            frame, f"Privacy: {privacy.mode} | {privacy.target}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA,
        )

        cv2.imshow("Privacy Mode", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            idx = (modes.index(privacy.mode) + 1) % len(modes)
            privacy.mode = modes[idx]
            print(f"Mode: {privacy.mode}")
        elif key == ord("t"):
            idx = (targets.index(privacy.target) + 1) % len(targets)
            privacy.target = targets[idx]
            print(f"Target: {privacy.target}")
        elif key == ord("s"):
            path = f"output/privacy_screenshot.png"
            cv2.imwrite(path, frame)
            print(f"Saved: {path}")

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
