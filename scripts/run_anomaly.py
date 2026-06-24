#!/usr/bin/env python3
"""Run anomaly detection on video stream.

First learns "normal" patterns, then flags deviations.

Usage:
    python scripts/run_anomaly.py
    python scripts/run_anomaly.py --learning-frames 300 --source 0
    python scripts/run_anomaly.py --load-model anomaly_model.pkl

Controls:
    q     - Quit
    s     - Save screenshot
    f     - Force-fit model (skip remaining learning)
    space - Pause/resume
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.anomaly_detector import AnomalyDetector
from src.models.registry import build_detector_from_config
from src.stream import VideoStream
from src.utils.drawing import draw_anomaly_alert, draw_detections, draw_fps, draw_info
from src.utils.fps import FPSCounter


def main():
    parser = argparse.ArgumentParser(description="Anomaly detection from scene patterns")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--learning-frames", type=int, default=500)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--load-model", default=None, help="Load pre-trained anomaly model")
    parser.add_argument("--save-model", default="anomaly_model.pkl")
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    detector = build_detector_from_config({}, model_name=args.model)
    stream = VideoStream(source=source)
    fps_counter = FPSCounter()

    anomaly = AnomalyDetector(
        learning_frames=args.learning_frames,
        contamination=args.contamination,
    )

    if args.load_model and Path(args.load_model).exists():
        anomaly.load(args.load_model)
        print(f"Loaded anomaly model from {args.load_model}")
    else:
        print(f"Learning phase: collecting {args.learning_frames} frames of normal behavior...")

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

            detections = detector.detect(frame)
            fps_counter.tick()

            h, w = frame.shape[:2]
            score, is_anomalous = anomaly.check(detections, (h, w))

            # Draw
            draw_detections(frame, detections, show_confidence=True)
            draw_fps(frame, fps_counter.fps)
            draw_info(frame, args.model, len(detections))

            if anomaly.is_learning:
                progress = anomaly.learning_progress
                cv2.putText(
                    frame, f"LEARNING: {progress:.0%}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA,
                )
                # Progress bar
                bar_w = int(w * 0.6)
                cv2.rectangle(frame, (10, h - 50), (10 + bar_w, h - 35), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, h - 50), (10 + int(bar_w * progress), h - 35), (0, 255, 0), -1)
            else:
                draw_anomaly_alert(frame, score, is_anomalous)

            cv2.imshow("Anomaly Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                if anomaly.is_learning:
                    try:
                        anomaly.fit()
                        print("Model fitted (forced).")
                        if args.save_model:
                            anomaly.save(args.save_model)
                            print(f"Saved model to {args.save_model}")
                    except ValueError as e:
                        print(f"Cannot fit: {e}")
            elif key == ord("s"):
                Path("screenshots").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshots/anomaly_{ts}.jpg", frame)
            elif key == ord(" "):
                paused = not paused

    finally:
        if anomaly.is_fitted and args.save_model:
            anomaly.save(args.save_model)
            print(f"Saved anomaly model to {args.save_model}")
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
