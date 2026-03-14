"""Model export and benchmarking for edge deployment.

Supports exporting YOLO models to ONNX, TensorRT, and other formats,
and benchmarking inference latency across different configurations.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ExportResult:
    """Result of a model export operation."""

    format: str
    output_path: str
    file_size_mb: float
    success: bool
    error_message: str = ""


@dataclass
class BenchmarkResult:
    """Result of a model benchmark."""

    model_path: str
    format: str
    device: str
    image_size: tuple[int, int]
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    fps: float
    num_iterations: int
    warmup_iterations: int


class ModelExporter:
    """Export YOLO models to optimized formats for deployment.

    Supported formats:
    - ONNX: Cross-platform, CPU/GPU inference
    - TensorRT: NVIDIA GPU optimized (requires tensorrt)
    - OpenVINO: Intel hardware optimized
    - CoreML: Apple devices
    - TFLite: Mobile/embedded devices
    - NCNN: Mobile-optimized (Tencent)

    Example:
        exporter = ModelExporter("yolov8n.pt")
        result = exporter.export("onnx")
        print(f"Exported to {result.output_path} ({result.file_size_mb:.1f} MB)")
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize model exporter.

        Args:
            model_path: Path to YOLO model file.
        """
        from ultralytics import YOLO

        self._model_path = model_path
        self._model = YOLO(model_path)

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        output_dir: str | None = None,
    ) -> ExportResult:
        """Export model to specified format.

        Args:
            format: Target format ("onnx", "torchscript", "openvino",
                    "engine" for TensorRT, "coreml", "tflite", "ncnn").
            imgsz: Input image size.
            half: Use FP16 quantization.
            dynamic: Enable dynamic input shapes (ONNX).
            simplify: Simplify ONNX graph.
            output_dir: Custom output directory.

        Returns:
            ExportResult with export details.
        """
        try:
            export_path = self._model.export(
                format=format,
                imgsz=imgsz,
                half=half,
                dynamic=dynamic,
                simplify=simplify,
            )

            export_path = str(export_path)
            file_size = Path(export_path).stat().st_size / (1024 * 1024)

            return ExportResult(
                format=format,
                output_path=export_path,
                file_size_mb=file_size,
                success=True,
            )

        except Exception as e:
            return ExportResult(
                format=format,
                output_path="",
                file_size_mb=0,
                success=False,
                error_message=str(e),
            )

    def export_multiple(
        self,
        formats: list[str] | None = None,
        imgsz: int = 640,
        half: bool = False,
    ) -> list[ExportResult]:
        """Export model to multiple formats.

        Args:
            formats: List of formats. Defaults to common formats.
            imgsz: Input image size.
            half: Use FP16 quantization.

        Returns:
            List of ExportResult objects.
        """
        if formats is None:
            formats = ["onnx", "torchscript"]

        results = []
        for fmt in formats:
            print(f"Exporting to {fmt}...")
            result = self.export(format=fmt, imgsz=imgsz, half=half)
            results.append(result)
            if result.success:
                print(f"  -> {result.output_path} ({result.file_size_mb:.1f} MB)")
            else:
                print(f"  -> Failed: {result.error_message}")

        return results

    @property
    def model_info(self) -> dict:
        """Get model information."""
        return {
            "model_path": self._model_path,
            "task": str(self._model.task),
            "names": self._model.names,
        }


class BenchmarkRunner:
    """Benchmark model inference performance across formats and devices.

    Example:
        runner = BenchmarkRunner()
        result = runner.benchmark("yolov8n.onnx", format="onnx", device="cpu")
        print(f"Average: {result.avg_latency_ms:.1f}ms, FPS: {result.fps:.1f}")
    """

    def benchmark(
        self,
        model_path: str,
        format: str = "pytorch",
        device: str = "cpu",
        image_size: tuple[int, int] = (640, 640),
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> BenchmarkResult:
        """Benchmark a model's inference performance.

        Args:
            model_path: Path to model file.
            format: Model format for loading.
            device: Device to benchmark on.
            image_size: Input image size (width, height).
            num_iterations: Number of inference iterations.
            warmup: Number of warmup iterations.

        Returns:
            BenchmarkResult with timing statistics.
        """
        from ultralytics import YOLO

        model = YOLO(model_path)

        # Create dummy input
        dummy_input = np.random.randint(
            0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8
        )

        # Warmup
        for _ in range(warmup):
            model.predict(dummy_input, device=device, verbose=False)

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            model.predict(dummy_input, device=device, verbose=False)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0

        return BenchmarkResult(
            model_path=model_path,
            format=format,
            device=device,
            image_size=image_size,
            avg_latency_ms=float(avg_latency),
            min_latency_ms=float(min_latency),
            max_latency_ms=float(max_latency),
            fps=float(fps),
            num_iterations=num_iterations,
            warmup_iterations=warmup,
        )

    def compare(
        self,
        model_paths: dict[str, str],
        device: str = "cpu",
        image_size: tuple[int, int] = (640, 640),
        num_iterations: int = 50,
    ) -> list[BenchmarkResult]:
        """Compare performance of multiple models.

        Args:
            model_paths: Dict of format_name -> model_path.
            device: Device to benchmark on.
            image_size: Input image size.
            num_iterations: Iterations per model.

        Returns:
            List of BenchmarkResult objects.
        """
        results = []
        for fmt, path in model_paths.items():
            print(f"Benchmarking {fmt}: {path}")
            result = self.benchmark(
                model_path=path,
                format=fmt,
                device=device,
                image_size=image_size,
                num_iterations=num_iterations,
            )
            results.append(result)
            print(
                f"  Avg: {result.avg_latency_ms:.1f}ms, "
                f"FPS: {result.fps:.1f}, "
                f"Min: {result.min_latency_ms:.1f}ms, "
                f"Max: {result.max_latency_ms:.1f}ms"
            )

        return results

    @staticmethod
    def generate_report(results: list[BenchmarkResult]) -> str:
        """Generate a formatted benchmark comparison report.

        Args:
            results: List of benchmark results to compare.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 80,
            "MODEL BENCHMARK REPORT",
            "=" * 80,
            "",
            f"{'Format':<15} {'Device':<10} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'FPS':<10}",
            "-" * 80,
        ]

        for r in sorted(results, key=lambda x: x.avg_latency_ms):
            lines.append(
                f"{r.format:<15} {r.device:<10} {r.avg_latency_ms:<12.1f} "
                f"{r.min_latency_ms:<12.1f} {r.max_latency_ms:<12.1f} {r.fps:<10.1f}"
            )

        lines.extend(["", "=" * 80])

        # Find fastest
        if results:
            fastest = min(results, key=lambda x: x.avg_latency_ms)
            lines.append(f"Fastest: {fastest.format} at {fastest.fps:.1f} FPS")

        return "\n".join(lines)
