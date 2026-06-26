"""Monocular depth estimation subsystem."""

from src.depth.base import (
    BaseDepthEstimator,
    annotate_depth,
    is_too_close,
    percentile_normalize,
    sample_depth,
)
from src.depth.onnx_estimator import (
    DepthAnythingV2,
    DepthAnythingV2Metric,
    MidasONNX,
    build_depth_estimator,
)

__all__ = [
    "BaseDepthEstimator",
    "annotate_depth",
    "is_too_close",
    "percentile_normalize",
    "sample_depth",
    "DepthAnythingV2",
    "DepthAnythingV2Metric",
    "MidasONNX",
    "build_depth_estimator",
]
