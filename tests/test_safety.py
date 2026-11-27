"""The observer monitor: quiet when nothing happens, loud when something does."""

from __future__ import annotations

import time

import numpy as np

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
