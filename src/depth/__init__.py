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
from src.depth.ground_plane import (
    GroundPlaneHomographyRanger,
    PinholeGroundRanger,
    annotate_ground_range,
    build_ground_ranger,
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
    "prepare_depth_map",
    "sample_depth",
    "DepthScaleCalibrator",
    "PinholeGroundRanger",
    "GroundPlaneHomographyRanger",
    "annotate_ground_range",
    "build_ground_ranger",
    "DepthAnythingV2",
    "DepthAnythingV2Metric",
    "MidasONNX",
    "build_depth_estimator",
]
