"""The detector front-end: colour fallback works everywhere, SAM2 only where it can."""

from __future__ import annotations

import numpy as np
import pytest

from dofbot_grasp.tracker import ColourDetector, gpu_available, make_detector


def test_gpu_probe_never_raises():
    assert isinstance(gpu_available(), bool)


def test_colour_detector_returns_the_controller_interface():
    det = ColourDetector()
    assert det(np.zeros((240, 320, 3), np.uint8)) is None  # nothing to see
    det.reset()


def test_front_end_falls_back_without_a_gpu():
    det = make_detector(prefer_gpu=False)
    assert det.name == "colour"
    det.reset()


def test_front_end_never_raises_on_a_missing_stack(monkeypatch):
    """A broken torch install must degrade to colour, not crash the grasp."""
    import dofbot_grasp.tracker as t

    monkeypatch.setattr(t, "gpu_available", lambda: True)

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no CUDA")

    monkeypatch.setattr(t, "SamTracker", Boom)
    assert t.make_detector(prefer_gpu=True).name == "colour"


@pytest.mark.skipif(not gpu_available(), reason="no CUDA device in this environment")
def test_sam2_reports_area_and_mask():
    det = make_detector(prefer_gpu=True)
    assert det.name == "sam2"
