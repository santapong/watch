"""Skeleton-based action recognition (CTR-GCN)."""

from src.pose.ctrgcn import (
    CTRGCNActionClassifier,
    SkeletonSequenceBuffer,
    normalize_skeleton,
)

__all__ = ["CTRGCNActionClassifier", "SkeletonSequenceBuffer", "normalize_skeleton"]
