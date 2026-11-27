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
