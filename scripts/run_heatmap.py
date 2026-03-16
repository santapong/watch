"""Run object detection with heatmap overlay.

Generates density heatmaps showing where objects concentrate over time.
The heatmap accumulates gradually and fades with configurable decay.

Usage:
    python scripts/run_heatmap.py
    python scripts/run_heatmap.py --source 0 --classes person
    python scripts/run_heatmap.py --decay 0.99 --save-interval 300

Controls:
    q     - Quit
    s     - Save heatmap snapshot
    r     - Reset heatmap
    space - Toggle heatmap overlay
"""

import argparse
import time

import cv2
import yaml

from src.stream import VideoStream
from src.models.yolo_wrapper import YOLODetector
from src.analytics.heatmap import HeatmapGenerator
from src.utils.drawing import draw_detections, draw_fps
from src.utils.fps import FPSCounter


def parse_args():
    parser = argparse.ArgumentParser(description="Detection with heatmap overlay")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--source", default=None, help="Video source (webcam index or URL)")
    parser.add_argument("--model", default=None, help="YOLO model name")
    parser.add_argument("--decay", type=float, default=0.995, help="Heatmap decay factor")
    parser.add_argument("--radius", type=int, default=40, help="Gaussian blob radius")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay alpha")
    parser.add_argument("--classes", nargs="*", help="Filter to these class names")
    parser.add_argument("--save-interval", type=int, default=0, help="Auto-save interval in frames (0=disabled)")
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
    fps = FPSCounter()

    # Get frame size for heatmap
    frame = stream.read()
    if frame is None:
        print("Error: Could not read from video source.")
        return

    h, w = frame.shape[:2]
    heatmap = HeatmapGenerator(
        frame_shape=(h, w),
        decay=args.decay,
        radius=args.radius,
        class_filter=args.classes,
    )

    show_overlay = True
    print(f"Heatmap running | decay={args.decay} | radius={args.radius}")
    print("Controls: q=quit, s=save, r=reset, space=toggle overlay")

    while True:
        frame = stream.read()
        if frame is None:
            break

        detections = detector.detect(frame)
        heatmap.update(detections)
        fps.tick()

        draw_detections(frame, detections)
        draw_fps(frame, fps.fps)

        if show_overlay:
            frame = heatmap.render(frame, alpha=args.alpha)

        # Frame count display
        cv2.putText(
            frame, f"Heatmap frames: {heatmap.frame_count}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )

        cv2.imshow("Detection + Heatmap", frame)

        # Auto-save
        if args.save_interval > 0 and heatmap.frame_count % args.save_interval == 0:
            path = f"output/heatmap_{heatmap.frame_count}.png"
            heatmap.save_snapshot(path)
            print(f"Auto-saved heatmap: {path}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"output/heatmap_{int(time.time())}.png"
            heatmap.save_snapshot(path)
            print(f"Saved: {path}")
        elif key == ord("r"):
            heatmap.reset()
            print("Heatmap reset")
        elif key == ord(" "):
            show_overlay = not show_overlay
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
