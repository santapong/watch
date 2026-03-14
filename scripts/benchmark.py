#!/usr/bin/env python3
"""Benchmark model inference performance.

Usage:
    python scripts/benchmark.py --models yolov8n.pt yolov8s.pt
    python scripts/benchmark.py --models yolov8n.pt yolov8n.onnx --device cpu
    python scripts/benchmark.py --models yolov8n.pt --iterations 200 --device cuda:0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deployment.exporter import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model files to benchmark",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    runner = BenchmarkRunner()

    model_paths = {}
    for model_path in args.models:
        suffix = Path(model_path).suffix
        format_map = {
            ".pt": "pytorch",
            ".onnx": "onnx",
            ".engine": "tensorrt",
            ".xml": "openvino",
            ".mlmodel": "coreml",
            ".tflite": "tflite",
        }
        fmt = format_map.get(suffix, "unknown")
        name = Path(model_path).stem
        model_paths[f"{name} ({fmt})"] = model_path

    print(f"Device: {args.device}")
    print(f"Image size: {args.imgsz}")
    print(f"Iterations: {args.iterations}")
    print()

    results = runner.compare(
        model_paths=model_paths,
        device=args.device,
        image_size=tuple(args.imgsz),
        num_iterations=args.iterations,
    )

    print()
    print(BenchmarkRunner.generate_report(results))


if __name__ == "__main__":
    main()
