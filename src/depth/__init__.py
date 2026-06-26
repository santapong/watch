"""Monocular depth estimation subsystem."""

from src.depth.base import (
    BaseDepthEstimator,
    annotate_depth,
    is_too_close,
    percentile_normalize,
    prepare_depth_map,
    sample_depth,
)
from src.depth.calibration import DepthScaleCalibrator
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
    "prepare_depth_map",
    "sample_depth",
    "DepthScaleCalibrator",
    "DepthAnythingV2",
    "DepthAnythingV2Metric",
    "MidasONNX",
    "build_depth_estimator",
]
