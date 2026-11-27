"""Watching the room while the arm moves.

The observer camera has calibrated intrinsics but no extrinsics, so this cannot police a
3-D safety envelope. What it can do is notice that the scene changed when it should not
have: someone reaching in, an object appearing, the arm leaving the area it is supposed to
be working in. That covers the failure the wrist camera cannot -- the wrist camera only sees
its own narrow cone, and nothing at all while it is blacked out against the table.

Two signals, both in image space:

* **motion** -- the fraction of pixels that differ from a slowly updating baseline. A hand
  entering the frame moves far more than the arm's own slow motion, and a threshold well
  above the arm's own rate separates them.
* **freshness** -- a frame that has not changed at all for a while means the camera died or
  is unplugged, which is indistinguishable from "nothing happened" unless it is counted.

On alarm the monitor keeps writing frames next to the log, because the most useful thing
after an incident is what the room looked like just before it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


class ObserverMonitor:
    """A background thread that flags unexplained motion in the observer view."""

    def __init__(
        self,
        capture: Callable[[], np.ndarray | None],
        *,
        motion_fraction: float = 0.04,
        confirm_frames: int = 3,
        warmup_frames: int = 20,
        stale_seconds: float = 3.0,
        evidence_dir: Path | None = None,
        interval_s: float = 0.1,
    ) -> None:
        self._capture = capture
        self._motion_fraction = motion_fraction
        self._confirm = confirm_frames
        self._warmup = warmup_frames
        self._strikes = 0
        self._stale_seconds = stale_seconds
        self._interval = interval_s
        self._evidence_dir = evidence_dir
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._baseline: np.ndarray | None = None
        self._last_change = time.time()
        self._lock = threading.Lock()
        self._alarm: str | None = None
        self.frames = 0

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="observer-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> ObserverMonitor:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- state ----------------------------------------------------------
    @property
    def alarm(self) -> str | None:
        with self._lock:
            return self._alarm

    def raise_if_alarmed(self) -> None:
        """Called between stages: a set alarm ends the run."""
        reason = self.alarm
        if reason is not None:
            raise SafetyAbort(reason)

    # -- worker ---------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._capture()
            if frame is not None:
                self._inspect(frame)
            time.sleep(self._interval)

    def _inspect(self, frame: np.ndarray) -> None:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey = cv2.resize(grey, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        self.frames += 1
        with self._lock:
            if self._baseline is None:
                self._baseline = grey
                self._last_change = time.time()
                return
            if self.frames < self._warmup:
                # auto-exposure and auto-white-balance settle over the first frames; alarming
                # on that would make the monitor useless on a real webcam
                self._baseline = grey
                self._last_change = time.time()
                return
            changed = float(np.mean(np.abs(grey - self._baseline) > 25.0))
            if changed > self._motion_fraction:
                self._last_change = time.time()
                # one frame above the line is noise; several in a row is something moving
                self._strikes += 1
                if self._strikes >= self._confirm and self._alarm is None:
                    self._alarm = f"unexplained motion in the observer view ({changed:.1%})"
                    self._write_evidence(frame, changed)
            else:
                self._strikes = 0
            # the baseline follows slowly, so the arm's own motion is not an alarm
            self._baseline = 0.95 * self._baseline + 0.05 * grey
            stale = time.time() - self._last_change > self._stale_seconds
            if stale and self.frames > 5 and self._alarm is None:
                self._alarm = "observer camera has not changed -- is it still connected?"

    def _write_evidence(self, frame: np.ndarray, changed: float) -> None:
        if self._evidence_dir is None:
            return
        try:
            self._evidence_dir.mkdir(parents=True, exist_ok=True)
            name = time.strftime("observer_%Y%m%d_%H%M%S.png")
            cv2.imwrite(str(self._evidence_dir / name), frame)
        except Exception:
            pass


class SafetyAbort(RuntimeError):
    """Raised between stages when the observer monitor flags something."""
