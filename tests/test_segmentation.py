"""Tests for SAM2 segmentation wiring and mask-aware privacy (model mocked)."""

import sys
import types

import numpy as np
import pytest

from src.models.base import Detection
from src.privacy import PrivacyFilter


def _det(x1=10, y1=10, x2=90, y2=90, name="person", mask=None):
    return Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                     confidence=0.9, class_id=0, class_name=name, mask=mask)


class TestMaskAwarePrivacy:
    def test_mask_limits_filter_to_masked_pixels(self):
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:50] = 1  # left half of the bbox only
        out = PrivacyFilter(mode="blackout", target="person").apply(frame, [_det(mask=mask)])
        assert out[10:90, 10:50].sum() == 0  # masked region blacked out
        assert np.array_equal(out[10:90, 50:90], frame[10:90, 50:90])  # rest untouched

    def test_no_mask_filters_whole_bbox(self):
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        out = PrivacyFilter(mode="blackout", target="person").apply(frame, [_det()])
        assert out[10:90, 10:90].sum() == 0

    def test_roi_mask_crops_full_frame(self):
        full = np.zeros((100, 100), dtype=np.uint8)
        full[10:90, 10:50] = 1
        m = PrivacyFilter._roi_mask(full, 10, 10, 90, 90, (100, 100))
        assert m.shape == (80, 80)
        assert m[:, :40].all() and not m[:, 40:].any()

    def test_roi_mask_resizes_bbox_sized(self):
        m = PrivacyFilter._roi_mask(np.ones((40, 40), dtype=np.uint8), 10, 10, 90, 90, (100, 100))
        assert m.shape == (80, 80) and m.all()

    def test_roi_mask_none(self):
        assert PrivacyFilter._roi_mask(None, 0, 0, 10, 10, (100, 100)) is None


def _install_fake_sam(monkeypatch, masks_per_call=None):
    class _Masks:
        def __init__(self, data):
            self.data = data

    class _Result:
        def __init__(self, masks):
            self.masks = masks

    class _SAM:
        def __init__(self, name):
            self.name = name

        def __call__(self, frame, bboxes=None, **kwargs):
            data = masks_per_call if masks_per_call is not None else [
                np.ones((10, 10), dtype=np.float32) for _ in (bboxes or [])
            ]
            return [_Result(_Masks(data))]

    fake = types.ModuleType("ultralytics")
    fake.SAM = _SAM
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


class TestSegmenter:
    def test_build_unknown_raises(self):
        from src.segmentation import build_segmenter
        with pytest.raises(ValueError):
            build_segmenter({"backend": "bogus"})

    def test_build_sam2(self, monkeypatch):
        _install_fake_sam(monkeypatch)
        from src.segmentation import SAM2Segmenter, build_segmenter
        seg = build_segmenter({"backend": "sam2", "model": "sam2_b.pt"})
        assert isinstance(seg, SAM2Segmenter)
        assert seg.model_name == "sam2_b.pt"

    def test_segment_sets_masks(self, monkeypatch):
        _install_fake_sam(monkeypatch)
        from src.segmentation import SAM2Segmenter
        seg = SAM2Segmenter("sam2_b.pt")
        dets = [_det(0, 0, 10, 10)]
        out = seg.segment(np.zeros((20, 20, 3), dtype=np.uint8), dets)
        assert out[0].mask is not None
        assert out[0].mask.shape == (10, 10)

    def test_segment_empty_is_noop(self, monkeypatch):
        _install_fake_sam(monkeypatch)
        from src.segmentation import SAM2Segmenter
        assert SAM2Segmenter("sam2_b.pt").segment(np.zeros((20, 20, 3), dtype=np.uint8), []) == []
