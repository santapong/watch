#!/usr/bin/env python3
"""Run open-vocabulary (zero-shot) object detection.

Detects objects based on free-text descriptions using OWLv2.

Usage:
    python scripts/run_open_vocab.py --queries "person" "dog" "red car"
    python scripts/run_open_vocab.py --queries "fire extinguisher" --source 0
    python scripts/run_open_vocab.py --queries "person wearing hat" --source "http://PHONE_IP:8080/video"

Controls:
    q     - Quit
    s     - Save screenshot
    space - Pause/resume
    n     - Enter new queries (via terminal)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.open_vocab_detector import OpenVocabDetector
from src.stream import VideoStream
from src.utils.drawing import draw_detections, draw_fps, draw_info
from src.utils.fps import FPSCounter


def main():
    parser = argparse.ArgumentParser(description="Open-vocabulary detection with OWLv2")
    parser.add_argument(
        "--queries", nargs="+", default=["person", "car", "dog"],
        help="Text queries for detection",
    )
    parser.add_argument("--source", default="0", help="Video source")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    print(f"Loading OWLv2 model: {args.model}")
    print(f"Text queries: {args.queries}")

    detector = OpenVocabDetector(
        text_queries=args.queries,
        confidence=args.confidence,
        model_name=args.model,
    )

    stream = VideoStream(source=source)
    fps_counter = FPSCounter()

    print("Open-vocabulary detection started.")
    print("Controls: q=quit, s=screenshot, n=new queries, space=pause")

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

            draw_detections(frame, detections, show_confidence=True)
            draw_fps(frame, fps_counter.fps)
            draw_info(frame, detector.model_name, len(detections))

            # Show current queries
            queries_text = f"Queries: {', '.join(args.queries)}"
            cv2.putText(
                frame, queries_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow("Open-Vocab Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                Path("screenshots").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshots/openvocab_{ts}.jpg", frame)
                print("Screenshot saved.")
            elif key == ord("n"):
                new_queries = input("Enter new queries (comma-separated): ")
                args.queries = [q.strip() for q in new_queries.split(",") if q.strip()]
                detector.set_queries(args.queries)
                print(f"Updated queries: {args.queries}")
            elif key == ord(" "):
                paused = not paused

    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
