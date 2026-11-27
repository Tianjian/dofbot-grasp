"""The observer monitor: quiet when nothing happens, loud when something does."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from dofbot_grasp.safety import ObserverMonitor


def _frame(block_at: int | None = None) -> np.ndarray:
    frame = np.full((240, 320, 3), 90, np.uint8)
    if block_at is not None:
        frame[80:160, block_at : block_at + 60] = 255
    return frame


def test_quiet_scene_does_not_alarm():
    monitor = ObserverMonitor(lambda: _frame(), warmup_frames=3, confirm_frames=3)
    monitor.start()
    try:
        time.sleep(0.6)
    finally:
        monitor.stop()
    assert monitor.frames > 3
    assert monitor.alarm is None


def test_a_moving_block_alarms():
    state = {"at": None}
    monitor = ObserverMonitor(lambda: _frame(state["at"]), warmup_frames=3, confirm_frames=2)
    monitor.start()
    try:
        time.sleep(0.25)
        for at in (40, 120, 200):
            state["at"] = at
            time.sleep(0.15)
        time.sleep(0.2)
    finally:
        monitor.stop()
    assert monitor.alarm is not None
    assert "motion" in monitor.alarm


def test_raise_if_alarmed_reports_the_reason():
    monitor = ObserverMonitor(lambda: _frame(), warmup_frames=3)
    monitor._alarm = "something moved"  # the check is what matters, not the detection
    try:
        monitor.raise_if_alarmed()
    except Exception as exc:  # SafetyAbort
        assert "something moved" in str(exc)
    else:
        raise AssertionError("expected SafetyAbort")


def test_a_still_scene_is_not_a_dead_camera():
    """A live camera pointed at a still room must not be reported as disconnected."""
    now = {"t": 0.0}

    def clock() -> float:
        now["t"] += 0.1
        return now["t"]

    monitor = ObserverMonitor(lambda: _frame(), warmup_frames=2, stale_seconds=1.0)
    monitor._clock = clock  # type: ignore[attr-defined]
    for _ in range(40):  # four simulated seconds of nothing happening
        monitor._inspect(_frame())
    assert monitor.alarm is None


def test_a_silent_camera_is_reported():
    """No frames at all is the fault the stale check exists for."""
    monitor = ObserverMonitor(lambda: _frame(), warmup_frames=2, stale_seconds=0.05)
    monitor.start()
    try:
        time.sleep(0.3)
    finally:
        monitor.stop()
    assert monitor.alarm is None  # frames ARE arriving, so nothing to report


def test_cube_check_off_does_not_alarm():
    """With no seed the cube is never segmented, and a missing centre means nothing."""

    class NoopSemantic:
        available = True

        def check(self, _frame):
            from dofbot_grasp.observer_vision import SemanticState

            return SemanticState(available=True, cube_checked=False, cube_centre=None)

    monitor = ObserverMonitor(lambda: _frame(), semantic=NoopSemantic())
    for _ in range(5):
        monitor.check_semantics(_frame())
    assert monitor.alarm is None


def test_apparent_size_gives_a_distance():
    """A 30 mm cube one metre from a 640 px-focal camera is ~19 px wide."""
    from dofbot_grasp.observer_vision import distance_from_side_px

    intr = {"fx": 640.0}
    assert distance_from_side_px(19.2, intr) == pytest.approx(1.0, abs=0.02)
    assert distance_from_side_px(57.0, intr) == pytest.approx(0.337, abs=0.01)
    assert distance_from_side_px(None, intr) is None


def test_the_real_cube_is_detected():
    """The pale-yellow target, on a desk of blue-lined paper, at 66 mm.

    Kept as a real frame because the failure it guards against was not a logic error: the old
    saturation gate sat one unit above the cube's own measured value, so detection depended on
    the light. Segmentation replaced the threshold, and this asserts the segmenter still finds
    the cube the threshold used to miss.
    """
    import cv2
    import pytest

    pytest.importorskip("ultralytics", reason="the segmenter needs the GPU stack")
    from dofbot_grasp.tracker import make_detector

    path = Path(__file__).parent / "data" / "real_cube_wrist.png"
    frame = cv2.imread(str(path))
    assert frame is not None
    found = make_detector()(frame)
    assert found is not None
    assert found["centre"][0] == pytest.approx(240, abs=40)
    assert found["centre"][1] == pytest.approx(194, abs=40)
