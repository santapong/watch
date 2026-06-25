#!/usr/bin/env python3
"""Run pose estimation and action recognition.

Detects human poses and classifies actions (standing, sitting, walking, etc.).

Usage:
    python scripts/run_action.py
    python scripts/run_action.py --source "http://PHONE_IP:8080/video"
    python scripts/run_action.py --model yolov8s-pose.pt

Controls:
    q     - Quit
    s     - Save screenshot
    t     - Toggle tracking
    space - Pause/resume
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts import AlertManager, AlertRule, LogNotifier
from src.models.pose_detector import PoseDetector, ActionClassifier
from src.stream import VideoStream
from src.utils.drawing import draw_action_label, draw_fps, draw_skeleton
from src.utils.fps import FPSCounter


def main():
    parser = argparse.ArgumentParser(description="Pose estimation + action recognition")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config file (action/alert settings)")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    action_cfg = config.get("action", {})
    fall_cfg = action_cfg.get("fall", {})

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    print(f"Loading pose model: {args.model}")
    pose_detector = PoseDetector(
        model_name=args.model,
        confidence=args.confidence,
    )
    action_classifier = ActionClassifier(
        sequence_length=action_cfg.get("sequence_length", 15),
        fall_angle_deg=fall_cfg.get("angle_deg", 45.0),
        fall_velocity=fall_cfg.get("velocity", 8.0),
        fall_debounce=fall_cfg.get("debounce", 3),
    )

    # Fall alerts: fire a critical, cooldown-limited alert when anyone falls.
    alert_manager = AlertManager()
    alert_manager.add_rule(AlertRule(
        name="fall_detection",
        condition=lambda ctx: ctx.get("fall_detected", False),
        alert_type="fall",
        message="Fall detected",
        severity="critical",
        cooldown=fall_cfg.get("alert_cooldown", 10.0),
    ))
    alert_manager.add_notifier(LogNotifier(config.get("alerts", {}).get("log_path", "alerts.json")))

    stream = VideoStream(source=source)
    fps_counter = FPSCounter()
    tracking = args.track

    print("Action recognition started. Press 'q' to quit.")

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

            # Detect poses
            if tracking:
                poses = pose_detector.detect_poses_and_track(frame)
            else:
                poses = pose_detector.detect_poses(frame)

            # Classify actions
            poses = action_classifier.classify_batch(poses)

            # Fall alerting
            fallers = [p.track_id for p in poses if p.action == "falling"]
            for alert in alert_manager.evaluate(
                {"fall_detected": bool(fallers), "track_ids": fallers}
            ):
                print(f"[ALERT] {alert.severity.upper()}: {alert.message} (tracks={fallers})")

            fps_counter.tick()

            # Draw
            for pose in poses:
                # Draw skeleton
                draw_skeleton(frame, pose.keypoints)

                # Draw bounding box
                x1, y1, x2, y2 = [int(v) for v in pose.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw track ID
                if pose.track_id is not None:
                    cv2.putText(
                        frame, f"#{pose.track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                    )

                # Draw action label
                if pose.action:
                    draw_action_label(frame, pose.bbox, pose.action, pose.action_confidence)

            draw_fps(frame, fps_counter.fps)

            # Show person count and actions
            action_summary = {}
            for pose in poses:
                if pose.action:
                    action_summary[pose.action] = action_summary.get(pose.action, 0) + 1

            y = 60
            cv2.putText(
                frame, f"Persons: {len(poses)}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
            for action, count in action_summary.items():
                y += 22
                cv2.putText(
                    frame, f"  {action}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
                )

            cv2.imshow("Action Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                tracking = not tracking
                print(f"Tracking: {'ON' if tracking else 'OFF'}")
            elif key == ord("s"):
                Path("screenshots").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshots/action_{ts}.jpg", frame)
            elif key == ord(" "):
                paused = not paused

    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
