#!/usr/bin/env python3
"""Run active learning to collect uncertain samples for labeling.

Identifies and exports low-confidence detections for human review.

Usage:
    python scripts/run_active_learning.py
    python scripts/run_active_learning.py --session my_session --max-samples 200
    python scripts/run_active_learning.py --source video.mp4 --export

Controls:
    q     - Quit and export
    s     - Save screenshot
    e     - Export current queue
    space - Pause/resume
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import build_detector_from_config
from src.stream import VideoStream
from src.training.active_learner import ActiveLearner
from src.utils.drawing import draw_detections, draw_fps, draw_info
from src.utils.fps import FPSCounter


def main():
    parser = argparse.ArgumentParser(description="Active learning sample collection")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--session", default="session_001")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--interval", type=int, default=30, help="Min frames between samples")
    parser.add_argument("--export", action="store_true", help="Export on exit")
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    detector = build_detector_from_config({}, model_name=args.model)
    stream = VideoStream(source=source)
    fps_counter = FPSCounter()

    learner = ActiveLearner(
        output_dir="active_learning",
        max_samples=args.max_samples,
        sampling_interval=args.interval,
    )
    learner.start_session(args.session)

    print(f"Active learning session: {args.session}")
    print("Collecting uncertain samples. Press 'q' to quit, 'e' to export.")

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

            # Evaluate for active learning
            sample = learner.evaluate(detections, frame)

            # Draw
            draw_detections(frame, detections, show_confidence=True)
            draw_fps(frame, fps_counter.fps)
            draw_info(frame, args.model, len(detections))

            # Show active learning status
            h = frame.shape[0]
            stats = learner.stats
            status_text = f"AL Queue: {learner.queue_size}/{args.max_samples} | Frames: {stats.get('frames_processed', 0)}"
            cv2.putText(
                frame, status_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
            )

            if sample:
                cv2.putText(
                    frame, f"SAMPLED (score: {sample.uncertainty_score:.2f})", (10, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
                )

            cv2.imshow("Active Learning", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                export_path = learner.export_for_labeling()
                print(f"Exported {learner.queue_size} samples to {export_path}")
            elif key == ord("s"):
                Path("screenshots").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshots/al_{ts}.jpg", frame)
            elif key == ord(" "):
                paused = not paused

    finally:
        if args.export or learner.queue_size > 0:
            export_path = learner.export_for_labeling()
            print(f"Exported {learner.queue_size} samples to {export_path}")
        stream.release()
        cv2.destroyAllWindows()
        print(f"Session stats: {learner.stats}")


if __name__ == "__main__":
    main()
