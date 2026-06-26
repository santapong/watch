#!/usr/bin/env python3
"""Run monocular depth estimation alongside detection.

Overlays a relative distance per detection and raises a proximity alert when an
object gets too close. The depth model is an ONNX file (Depth Anything V2 / MiDaS);
see docs/webcam_depth_research.md and requirements-phase2.txt (onnxruntime + weights
are NOT installed by default).

Usage:
    python scripts/run_depth.py --depth-model models/depth_anything_v2_vits.onnx
    python scripts/run_depth.py --depth-model m.onnx --source 0 --model yolo11n.pt
    python scripts/run_depth.py --depth-model m.onnx --depth-backend midas

Controls:
    q - Quit
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts import AlertManager, AlertRule, LogNotifier
from src.depth import (
    annotate_depth,
    build_depth_estimator,
    is_too_close,
    prepare_depth_map,
)
from src.models.registry import build_detector_from_config
from src.stream import VideoStream
from src.utils.drawing import draw_detections, draw_fps
from src.utils.fps import FPSCounter


def main():
    parser = argparse.ArgumentParser(description="Depth-aware detection")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--depth-model", default=None, help="Path to the ONNX depth model")
    parser.add_argument("--depth-backend", default=None,
                        help="depth_anything | depth_anything_metric | midas")
    parser.add_argument("--depth-input-size", nargs=2, type=int, default=None,
                        metavar=("W", "H"),
                        help="override depth model input size; smaller = faster on edge "
                             "(e.g. 384 384 or 256 256)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    import yaml
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    depth_cfg = dict(config.get("depth", {}))
    if args.depth_model:
        depth_cfg["model_path"] = args.depth_model
    if args.depth_backend:
        depth_cfg["backend"] = args.depth_backend
    if args.depth_input_size:
        depth_cfg["input_size"] = args.depth_input_size
    # Relative backends use a normalized [0,1] threshold (larger = nearer); the metric
    # backend uses meters (smaller = nearer). The estimator declares which via .units.
    proximity_rel = depth_cfg.get("proximity_threshold", 0.8)
    proximity_m = depth_cfg.get("proximity_threshold_m", 2.0)

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    detector = build_detector_from_config(config, model_name=args.model)
    depth_estimator = build_depth_estimator(depth_cfg)  # raises if model_path missing
    units = getattr(depth_estimator, "units", "relative")
    threshold = proximity_m if units == "metric" else proximity_rel
    print(f"Depth model: {depth_estimator.model_name} (units={units}, threshold={threshold})")

    alert_manager = AlertManager()
    alert_manager.add_rule(AlertRule(
        name="proximity",
        condition=lambda ctx: ctx.get("too_close", False),
        alert_type="proximity",
        message="Object too close",
        severity="warning",
        cooldown=depth_cfg.get("alert_cooldown", 10.0),
    ))
    alert_manager.add_notifier(LogNotifier(config.get("alerts", {}).get("log_path", "alerts.json")))

    stream = VideoStream(source=source)
    fps = FPSCounter()
    print("Depth detection started. Press 'q' to quit.")

    try:
        while stream.is_opened:
            frame = stream.read()
            if frame is None:
                continue

            detections = detector.detect(frame)
            raw_depth = depth_estimator.estimate(frame)
            # Metric depth is already in meters; only relative/inverse maps get normalized.
            depth_map = prepare_depth_map(raw_depth, units)
            annotate_depth(detections, depth_map, units=units)
            fps.tick()

            draw_detections(frame, detections)
            suffix = "m" if units == "metric" else ""
            too_close = []
            for det in detections:
                if det.depth is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in det.bbox)
                cv2.putText(
                    frame, f"d={det.depth:.2f}{suffix}", (x1, max(12, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
                )
                if is_too_close(det.depth, threshold, units):
                    too_close.append(det)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            for alert in alert_manager.evaluate({"too_close": bool(too_close)}):
                print(f"[ALERT] {alert.severity.upper()}: {alert.message} ({len(too_close)} object(s))")

            draw_fps(frame, fps.fps)
            cv2.imshow("Depth", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
