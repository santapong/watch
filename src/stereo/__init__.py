"""Real-time stereo depth subsystem (lazy scaffold)."""

from src.stereo.base import (
    BaseStereoMatcher,
    StereoRig,
    disparity_to_depth,
    disparity_to_depth_map,
    rectify_stereo_pair,
)
from src.stereo.onnx_matcher import (
    ESMStereo,
    OnnxStereoMatcher,
    build_stereo_matcher,
)

__all__ = [
    "BaseStereoMatcher",
    "StereoRig",
    "disparity_to_depth",
    "disparity_to_depth_map",
    "rectify_stereo_pair",
    "ESMStereo",
    "OnnxStereoMatcher",
    "build_stereo_matcher",
]
