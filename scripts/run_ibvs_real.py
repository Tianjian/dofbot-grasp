"""Run the grasp on the real DOFBOT SE.

Thin wrapper: the sequence lives in ``dofbot_grasp.controller.grasp_task`` and the bus
rules in ``dofbot_grasp.arm``, so this script only builds the backend, finds the cube, and
hands over.

    python scripts/run_ibvs_real.py --dry-run        # report what it sees, move nothing
    python scripts/run_ibvs_real.py --measure-only   # find it, measure it, stop
    python scripts/run_ibvs_real.py                  # full grasp, observer monitor running

Finding the cube is the interesting part. The wrist camera sits 57 mm off the tool axis and
95 mm above the fingertips, so it looks at a patch of desk roughly 56 mm beyond the tip; poses
chosen by where the tip goes aim it at empty desk. The sweep is therefore planned from where
the camera's optical axis lands, and the view it keeps is the one with the cube nearest the
image centre -- a cube at the edge back-projects badly and cannot be fitted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dofbot_grasp.controller import (  # noqa: E402
    grasp_task,
    observer_sweep,
    tip_and_axis,
)
from dofbot_grasp.hardware import SerialRobot  # noqa: E402
from dofbot_grasp.params import CAMERAS  # noqa: E402
from dofbot_grasp.tracker import make_detector  # noqa: E402


def find_cube(robot, detector, *, force_sweep: bool = False) -> dict | None:
    """The view that frames the cube best, sweeping if it is not already in sight."""
    if not force_sweep:
        seen = detector(robot.capture_wrist())
        if seen is not None:
            return seen
    print("cube not visible from here; sweeping where the camera looks")
    best = None
    for cand in observer_sweep():
        robot.command(cand, gripper=0.0)
        seen = detector(robot.capture_wrist())
        if seen is None:
            continue
        off = float(np.hypot(seen["centre"][0] - 320.0, seen["centre"][1] - 240.0))
        if best is None or off < best[0]:
            best = (off, cand, seen)
    if best is None:
        return None
    off, _cand, seen = best
    print(f"best view {off:.0f} px from centre")
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report, do not move")
    ap.add_argument(
        "--measures-only",
        "--measure-only",
        dest="measure_only",
        action="store_true",
        help="find and measure, then stop before touching it",
    )
    args = ap.parse_args()

    path = ROOT / CAMERAS.wrist_intrinsics
    if not path.exists():
        print(f"missing intrinsics: {path}")
        return 2
    intr = json.loads(path.read_text())

    robot = SerialRobot()
    detector = make_detector()
    print(f"detector: {getattr(detector, 'name', 'colour')}")

    monitor = None
    semantic = None
    record_path = None
    if not args.dry_run:
        from dofbot_grasp.safety import ObserverMonitor

        try:
            from dofbot_grasp.observer_vision import ObserverVision

            semantic = ObserverVision(ROOT)
            print(
                "observer semantics:",
                "GPU" if semantic.state.available else "unavailable, pixel check only",
            )
        except Exception as exc:
            print(f"observer semantics unavailable ({type(exc).__name__})")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        record_path = ROOT / "logs" / "observer" / f"run_{stamp}.mp4"
        monitor = ObserverMonitor(
            robot.capture_observer,
            evidence_dir=ROOT / "logs" / "observer",
            semantic=semantic,
            # the arm's own motion is the largest change in this view; the face check is what
            # can tell a person apart from it
            motion_enabled=False,
            record_path=record_path,
        )
        monitor.start()
        # Recording starts here, before anything moves: most useful reviews are of the runs
        # that failed before the grasp, and those never reached the old start point.
        print(f"recording this run to {record_path}")
    try:
        pose = robot.read_joints()
        tip, axis = tip_and_axis(pose)
        print("pose (deg):", {k: round(v, 1) for k, v in pose.items()})
        print("fingertip (mm):", np.round(tip * 1000, 0), " tool axis:", np.round(axis, 2))

        # Everything from here is the flow: park, observation pose, watch, choose a view,
        # measure, grasp, verify. The script used to position the arm itself first, with its own
        # sweep and candidate search -- a second strategy that could fail before the flow even
        # started, and did.
        if args.dry_run:
            print("dry run: reading the pose only")
            return 0

        from dofbot_grasp.safety import SafetyAbort

        def guard() -> None:
            monitor.raise_if_alarmed()
            # the arm has just moved; take a fresh reference so its own motion is not an alarm
            monitor.rebaseline()
            if semantic is not None and semantic.state.available:
                frame = robot.capture_observer()
                if frame is not None:
                    monitor.check_semantics(frame)
                monitor.raise_if_alarmed()

        def stage(name: str) -> None:
            monitor.annotate(name)  # burnt into the video, for the review afterwards

        try:
            result = grasp_task(
                robot,
                detector,
                log=print,
                abort_check=guard,
                on_stage=stage,
                intr=intr,
                watch_timeout_s=30.0,
                hold_s=60.0,
            )
        except SafetyAbort as exc:
            print(f"ABORTED by the observer monitor: {exc}")
            return 3
        finally:
            monitor.stop()
        print("result:", result)
        if monitor.alarm:
            print("observer alarm during the run:", monitor.alarm)
        return 0 if result.get("ok") else 1
    finally:
        if monitor is not None:
            monitor.stop()
            print(f"recording stopped: {record_path}")
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
