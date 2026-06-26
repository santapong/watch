"""Tests for CTR-GCN skeleton action recognition (pure helpers; model mocked)."""

import numpy as np

from src.pose.ctrgcn import CTRGCNActionClassifier, SkeletonSequenceBuffer, normalize_skeleton


def _kpts(dy=0.0):
    k = np.zeros((17, 3), dtype=np.float32)
    k[5] = [40, 100, 0.9]          # left shoulder
    k[6] = [60, 100, 0.9]          # right shoulder
    k[11] = [45, 200 + dy, 0.9]    # left hip
    k[12] = [55, 200 + dy, 0.9]    # right hip
    return k


def test_normalize_centers_on_hip():
    n = normalize_skeleton(_kpts())
    hip_mid = (n[11] + n[12]) / 2.0
    assert np.allclose(hip_mid, [0.0, 0.0], atol=1e-5)


def test_buffer_ready_and_sequence_shape():
    b = SkeletonSequenceBuffer(length=3)
    assert not b.is_ready(1)
    for _ in range(3):
        b.update(1, _kpts())
    assert b.is_ready(1)
    assert b.sequence(1).shape == (3, 17, 2)


def test_classify_unknown_without_model():
    clf = CTRGCNActionClassifier(sequence_length=2)  # no model -> fallback
    action, conf = clf.classify(1, _kpts())
    assert action == "unknown" and conf == 0.0


def test_classify_dispatches_to_model_when_ready(monkeypatch):
    clf = CTRGCNActionClassifier(sequence_length=2, model=object(), labels=["walk"])
    monkeypatch.setattr(clf, "_infer", lambda seq: ("walk", 0.9))
    assert clf.classify(1, _kpts())[0] == "unknown"   # buffer not full yet (1/2)
    assert clf.classify(1, _kpts()) == ("walk", 0.9)  # full -> _infer
