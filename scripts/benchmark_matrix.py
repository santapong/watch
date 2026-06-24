#!/usr/bin/env python3
"""Benchmark a matrix of {model} x {precision} x {runtime} for deployment selection.

Exports each needed combination, measures latency / FPS (and optional mAP), and
writes a comparison table (markdown) plus machine-readable JSON. Runtimes that
aren't installed on the host are recorded as "skipped" rather than failing, so the
same command yields a useful (partial) matrix on a CPU laptop or a GPU box — and the
numbers, not a research claim, pick the production default.

Usage:
    python scripts/benchmark_matrix.py --models yolov8n.pt yolo11n.pt
    python scripts/benchmark_matrix.py --models yolov8n.pt --runtimes onnx openvino \\
        --precisions fp32 int8 --data coco128.yaml --device cpu
    python scripts/benchmark_matrix.py --models yolov8n.pt --device cuda:0 --val
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deployment.exporter import BenchmarkRunner, ModelExporter
from src.models.registry import _infer_family

# runtime -> how to reach it and which precisions this harness benchmarks for it.
# (YOLO PyTorch fp16 inference is a runtime flag, not an export, so pytorch stays
# fp32-only here; quantized precisions go through the export-based runtimes.)
RUNTIME_SPECS = {
    "pytorch": {"export_format": None, "bench_format": "pytorch", "precisions": ("fp32",)},
    "onnx": {"export_format": "onnx", "bench_format": "onnx", "precisions": ("fp32", "int8")},
    "openvino": {"export_format": "openvino", "bench_format": "openvino", "precisions": ("fp32", "int8")},
    "tensorrt": {"export_format": "engine", "bench_format": "tensorrt", "precisions": ("fp16", "int8")},
}
ALL_RUNTIMES = tuple(RUNTIME_SPECS)
ALL_PRECISIONS = ("fp32", "fp16", "int8")


def _probe(device: str) -> dict:
    """Detect which runtimes are importable on this host (never raises)."""
    avail: dict = {}
    try:
        import torch

        try:
            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False
        avail["pytorch"] = (True, f"torch {torch.__version__} (cuda={cuda})")
        avail["_cuda"] = (cuda, "")
    except Exception as e:  # noqa: BLE001
        avail["pytorch"] = (False, f"torch not importable: {e}")
        avail["_cuda"] = (False, "")

    for name, mod in (("onnx", "onnxruntime"), ("openvino", "openvino"), ("tensorrt", "tensorrt")):
        try:
            m = __import__(mod)
            avail[name] = (True, f"{mod} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            avail[name] = (False, f"{mod} not installed: {e}")
    return avail


def _row(model, runtime, precision, status, reason="", latency=None, fps=None,
         size=None, mAP=None) -> dict:
    """Build a single matrix-cell record."""
    return {
        "model": model,
        "family": _infer_family(model),
        "runtime": runtime,
        "precision": precision,
        "status": status,
        "reason": reason,
        "latency_ms": latency,
        "fps": fps,
        "model_size_mb": size,
        "map50_95": mAP,
    }


def _run_cell(model, runtime, precision, args, exporter_cache) -> dict:
    """Export (if needed), benchmark, and optionally validate one cell."""
    spec = RUNTIME_SPECS[runtime]

    # 1) Get a model artifact to benchmark.
    if spec["export_format"] is None:
        model_path = model
        try:
            size_mb = round(Path(model).stat().st_size / (1024 * 1024), 2)
        except OSError:
            size_mb = None
    else:
        exporter = exporter_cache.get(model)
        if exporter is None:
            try:
                exporter = ModelExporter(model)
            except Exception as e:  # noqa: BLE001
                return _row(model, runtime, precision, "export_failed",
                            reason=f"could not load model: {e}")
            exporter_cache[model] = exporter
        res = exporter.export(
            format=spec["export_format"],
            imgsz=args.imgsz,
            half=(precision == "fp16"),
            int8=(precision == "int8"),
            data=args.data,
        )
        if not res.success:
            return _row(model, runtime, precision, "export_failed",
                        reason=f"export failed: {res.error_message}")
        model_path = res.output_path
        size_mb = round(res.file_size_mb, 2)

    # 2) Benchmark latency / FPS.
    try:
        b = BenchmarkRunner().benchmark(
            model_path=model_path,
            format=spec["bench_format"],
            device=args.device,
            image_size=(args.imgsz, args.imgsz),
            num_iterations=args.iterations,
            warmup=args.warmup,
        )
    except Exception as e:  # noqa: BLE001
        return _row(model, runtime, precision, "benchmark_failed",
                    reason=f"benchmark failed: {e}", size=size_mb)

    # 3) Optional accuracy (downloads the val split — best effort, only with --val).
    mAP = None
    if args.val and args.data:
        try:
            from ultralytics import YOLO

            metrics = YOLO(model_path).val(data=args.data, imgsz=args.imgsz, verbose=False)
            mAP = round(float(metrics.box.map), 4)
        except Exception:  # noqa: BLE001
            mAP = None

    latency = {
        "p50": round(b.p50_latency_ms, 3),
        "p95": round(b.p95_latency_ms, 3),
        "avg": round(b.avg_latency_ms, 3),
        "min": round(b.min_latency_ms, 3),
        "max": round(b.max_latency_ms, 3),
    }
    return _row(model, runtime, precision, "ok", latency=latency,
                fps=round(b.fps, 2), size=size_mb, mAP=mAP)


def _render_markdown(out: dict) -> str:
    h = out["host"]
    lines = [
        "# Benchmark matrix",
        "",
        f"_Generated {out['generatedDate']}_",
        "",
        f"- Device: `{h['device']}` (CUDA available: {h['cuda_available']})",
        f"- Image size: {h['imgsz']}, iterations: {h['iterations']}",
        "- Runtimes detected:",
    ]
    for rt, detail in h["runtimes"].items():
        lines.append(f"  - **{rt}**: {detail}")
    lines.append("")

    by_model: dict = {}
    for r in out["rows"]:
        by_model.setdefault(r["model"], []).append(r)

    def sort_key(r):
        lat = r["latency_ms"]["p50"] if r["latency_ms"] else float("inf")
        return (0 if r["status"] == "ok" else 1, lat)

    for model, rows in by_model.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| Runtime | Precision | p50 ms | p95 ms | FPS | Size MB | mAP50-95 | Status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=sort_key):
            lat = r["latency_ms"]
            p50 = f"{lat['p50']:.2f}" if lat else "—"
            p95 = f"{lat['p95']:.2f}" if lat else "—"
            fps = f"{r['fps']:.1f}" if r["fps"] is not None else "—"
            size = f"{r['model_size_mb']:.1f}" if r["model_size_mb"] is not None else "—"
            mp = f"{r['map50_95']:.4f}" if r["map50_95"] is not None else "—"
            status = r["status"]
            if r["status"] != "ok" and r["reason"]:
                status += f" ({r['reason']})"
            lines.append(
                f"| {r['runtime']} | {r['precision']} | {p50} | {p95} | {fps} | {size} | {mp} | {status} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Model x precision x runtime benchmark matrix")
    parser.add_argument("--models", nargs="+", required=True, help="Model weights, e.g. yolov8n.pt yolo11n.pt rtdetr-l.pt")
    parser.add_argument("--precisions", nargs="+", default=list(ALL_PRECISIONS), choices=ALL_PRECISIONS)
    parser.add_argument("--runtimes", nargs="+", default=list(ALL_RUNTIMES), choices=ALL_RUNTIMES)
    parser.add_argument("--device", default="cpu", help='Benchmark device ("cpu", "cuda:0")')
    parser.add_argument("--data", default=None, help="Calibration/val dataset YAML (e.g. coco128.yaml)")
    parser.add_argument("--val", action="store_true", help="Also measure mAP50-95 (downloads val split)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", default="docs/research/data/benchmark_matrix.json")
    args = parser.parse_args()

    probe = _probe(args.device)
    cuda_ok = probe["_cuda"][0]
    device_is_cuda = str(args.device).startswith("cuda")

    rows = []
    exporter_cache: dict = {}
    for model in args.models:
        for runtime in args.runtimes:
            spec = RUNTIME_SPECS[runtime]
            available, detail = probe.get(runtime, (False, "unknown runtime"))
            for precision in args.precisions:
                if precision not in spec["precisions"]:
                    rows.append(_row(model, runtime, precision, "skipped",
                                     reason=f"{runtime} not benchmarked at {precision} here "
                                            f"(supported: {', '.join(spec['precisions'])})"))
                    continue
                if not available:
                    rows.append(_row(model, runtime, precision, "skipped", reason=detail))
                    continue
                if (runtime == "tensorrt" or precision == "fp16") and not (cuda_ok and device_is_cuda):
                    rows.append(_row(model, runtime, precision, "skipped",
                                     reason="requires a CUDA device"))
                    continue
                print(f"[bench] {model} | {runtime} | {precision} ...", flush=True)
                rows.append(_run_cell(model, runtime, precision, args, exporter_cache))

    out = {
        "generatedDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": {
            "device": args.device,
            "cuda_available": cuda_ok,
            "imgsz": args.imgsz,
            "iterations": args.iterations,
            "runtimes": {rt: probe.get(rt, (False, ""))[1] for rt in ALL_RUNTIMES},
        },
        "models": args.models,
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    md = _render_markdown(out)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(md)

    print()
    print(md)
    print(f"\nWrote {out_path} and {md_path}")


if __name__ == "__main__":
    main()
