"""Run the grasp on the real DOFBOT SE.

Thin wrapper: the sequence lives in ``dofbot_grasp.controller.grasp_measured`` and the bus
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dofbot_grasp.controller import (  # noqa: E402
    camera_pose,
    grasp_measured,
    measure_cube,
    observer_sweep,
    search_poses,
    tip_and_axis,
)
from dofbot_grasp.hardware import SerialRobot  # noqa: E402
from dofbot_grasp.params import CAMERAS, GRASP, JOINTS  # noqa: E402
from dofbot_grasp.tracker import make_detector  # noqa: E402


def find_cube(robot, detector) -> dict | None:
    """The view that frames the cube best, sweeping if it is not already in sight."""
    seen = detector(robot.capture_wrist())
    if seen is not None:
        return seen
    print("cube not visible from here; sweeping where the camera looks")
    best = None
    for cand in observer_sweep():
        robot.command(cand, gripper=JOINTS.gripper_open_deg)
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
    try:
        pose = robot.read_joints()
        tip, axis = tip_and_axis(pose)
        print("pose (deg):", {k: round(v, 1) for k, v in pose.items()})
        print("fingertip (mm):", np.round(tip * 1000, 0), " tool axis:", np.round(axis, 2))

        det = find_cube(robot, detector)
        if det is None:
            print("no pose saw the cube -- put it where the arm can reach it")
            return 1
        pose = robot.read_joints()
        rough = robot.unproject_to_table(det["centre"][0], det["centre"][1], GRASP.table_z_m)
        cam_c, _ = camera_pose(pose)
        print(
            f"cube pixel {tuple(round(v) for v in det['centre'])}, "
            f"back-projected ({rough[0] * 1000:.0f}, {rough[1] * 1000:.0f}) mm"
        )
        print(
            f"camera at ({cam_c[0] * 1000:.0f}, {cam_c[1] * 1000:.0f}, "
            f"{cam_c[2] * 1000:.0f}) mm, fx {intr['fy']:.0f}"
        )

        # Seeing the cube is not measuring it: from where it was first noticed the silhouette
        # may be half out of frame or grazing. Try the poses that should frame it properly and
        # can also reach it, and keep the first whose fit is good enough to act on.
        measured = None
        for cand in search_poses(intr, (float(rough[0]), float(rough[1])), limit=8):
            robot.command(cand, gripper=JOINTS.gripper_open_deg)
            measured = measure_cube(robot, detector, log=print)
            if measured is not None:
                print(f"measured from { {k: round(v) for k, v in cand.items()} }")
                break
        if measured is None:
            print("no pose gave a usable fit; refusing to move on a guess")
            return 1

        # Independent check: the observer sees the same cube, and although its pose is not
        # calibrated, its apparent size gives a distance. If that disagrees badly with the
        # wrist camera's position, one of them is wrong and the run should not continue.
        from dofbot_grasp.observer_vision import cube_side_px, distance_from_side_px

        obs_frame = robot.capture_observer()
        side = cube_side_px(obs_frame) if obs_frame is not None else None
        obs_intr = json.loads((ROOT / CAMERAS.observer_intrinsics).read_text())
        obs_dist = distance_from_side_px(side, obs_intr) if side else None
        if obs_dist is not None:
            here = np.array([measured["x"], measured["y"], 0.015])
            observer_at = np.array(obs_intr.get("position_m", [0.41, 0.35, 0.025]))
            expect = float(np.linalg.norm(here - observer_at))
            # The apparent size gives a distance from the observer with no extrinsics at all,
            # which is why it is worth printing. Turning it into a verdict needs the observer's
            # own position, and that mount is nominal -- so this stays information until
            # somebody calibrates it. (Calibrating it from a chessboard is scripted:
            # scripts/calibrate_observer_extrinsics.py.)
            print(
                f"observer check: cube {side:.0f} px wide -> {obs_dist * 1000:.0f} mm from the "
                f"observer; the wrist position sits {expect * 1000:.0f} mm from the nominal mount"
            )
        else:
            print("observer check: cube not measurable in the observer view")

        if args.dry_run:
            print("dry run: nothing moved")
            return 0
        if args.measure_only:
            print(
                f"measured ({measured['x'] * 1000:.0f}, {measured['y'] * 1000:.0f}) mm, "
                f"iou {measured['iou']:.2f} -- stopping before the grasp"
            )
            return 0

        monitor = None
        from dofbot_grasp.safety import ObserverMonitor, SafetyAbort

        semantic = None
        try:
            from dofbot_grasp.observer_vision import ObserverVision

            semantic = ObserverVision(ROOT)
            print(
                "observer semantics:",
                "GPU" if semantic.state.available else "unavailable, pixel check only",
            )
        except Exception as exc:
            print(f"observer semantics unavailable ({type(exc).__name__})")
        monitor = ObserverMonitor(
            robot.capture_observer, evidence_dir=ROOT / "logs" / "observer", semantic=semantic
        )
        monitor.start()

        def guard() -> None:
            monitor.raise_if_alarmed()
            if semantic is not None and semantic.state.available:
                frame = robot.capture_observer()
                if frame is not None:
                    monitor.check_semantics(frame)
                monitor.raise_if_alarmed()

        try:
            result = grasp_measured(robot, detector, log=print, abort_check=guard)
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
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
