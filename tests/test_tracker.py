"""The detector front end: a GPU segmenter, and a clear failure when there is no GPU stack."""

from __future__ import annotations

import numpy as np
import pytest

from dofbot_grasp.tracker import gpu_available, make_detector


def test_gpu_probe_never_raises():
    assert isinstance(gpu_available(), bool)


def test_the_detector_is_the_segmenter():
    """There is no colour fallback: a silent worse answer was worse than an error."""
    pytest.importorskip("ultralytics", reason="the segmenter needs the GPU stack")
    det = make_detector()
    assert type(det).__name__ == "SamCubeDetector"


def test_a_blank_frame_has_no_cube():
    """A frame with nothing in it must not invent a cube."""
    pytest.importorskip("ultralytics", reason="the segmenter needs the GPU stack")
    if not gpu_available():
        pytest.skip("no CUDA device on this machine")
    assert make_detector()(np.zeros((240, 320, 3), np.uint8)) is None
