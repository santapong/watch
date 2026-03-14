#!/usr/bin/env python3
"""Export YOLO model to optimized formats for deployment.

Usage:
    python scripts/export_model.py                              # Export to ONNX
    python scripts/export_model.py --format onnx torchscript    # Multiple formats
    python scripts/export_model.py --model yolov8s.pt --half    # FP16 quantization
    python scripts/export_model.py --benchmark                  # Export + benchmark
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deployment.exporter import ModelExporter, BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Export and benchmark YOLO models")
    parser.add_argument("--model", default="yolov8n.pt", help="Model to export")
    parser.add_argument(
        "--format", nargs="+", default=["onnx"],
        help="Export formats: onnx, torchscript, openvino, engine, coreml, tflite, ncnn",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", help="FP16 quantization")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark after export")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--device", default="cpu", help="Benchmark device")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Formats: {args.format}")
    print(f"Image size: {args.imgsz}")
    print(f"FP16: {args.half}")
    print()

    # Export
    exporter = ModelExporter(args.model)
    results = exporter.export_multiple(
        formats=args.format,
        imgsz=args.imgsz,
        half=args.half,
    )

    print()
    print("Export Summary:")
    print("-" * 60)
    for r in results:
        status = "OK" if r.success else "FAILED"
        size = f"{r.file_size_mb:.1f} MB" if r.success else r.error_message
        print(f"  {r.format:<15} {status:<8} {size}")

    # Benchmark if requested
    if args.benchmark:
        print()
        print("Running benchmarks...")
        runner = BenchmarkRunner()

        model_paths = {}
        # Always benchmark original PyTorch model
        model_paths["pytorch"] = args.model

        for r in results:
            if r.success:
                model_paths[r.format] = r.output_path

        bench_results = runner.compare(
            model_paths=model_paths,
            device=args.device,
            num_iterations=args.iterations,
        )

        print()
        print(BenchmarkRunner.generate_report(bench_results))


if __name__ == "__main__":
    main()
