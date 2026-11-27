"""Run the grasp on the real DOFBOT SE.

Thin wrapper: the sequence itself lives in ``dofbot_grasp.controller.grasp_measured`` and the
bus rules in ``dofbot_grasp.arm``, so this script only has to build the backend, report what
it sees, and hand over.

    python scripts/run_ibvs_real.py --dry-run     # measure and print, do not move
    python scripts/run_ibvs_real.py               # measure, then grasp

Safety comes from the bus wrapper: it never reads inside the write guard, it polls until the
joints stop moving rather than sleeping a fixed time, and it raises when a servo lags its
command by more than ``SAFETY.max_servo_lag_deg`` -- which is what contact looks like.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dofbot_grasp.controller import camera_pose, grasp_measured, tip_and_axis  # noqa: E402
from dofbot_grasp.hardware import SerialRobot  # noqa: E402
from dofbot_grasp.params import CAMERAS, GRASP, JOINTS  # noqa: E402
from dofbot_grasp.tracker import make_detector  # noqa: E402


def sweep_poses() -> list[dict[int, float]]:
    """A handful of poses covering the reachable table, all with the fingers well clear."""
    out: list[dict[int, float]] = []
    for j1 in (60.0, 90.0, 120.0):
        for j2, j3 in ((90.0, 60.0), (110.0, 20.0), (70.0, 100.0)):
            j4 = j2 + j3 - 90.0
            if 0.0 <= j4 <= 180.0:
                out.append({1: j1, 2: j2, 3: j3, 4: j4, 5: 135.0})
    return out


def intr_path() -> Path:
    return ROOT / CAMERAS.wrist_intrinsics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="measure and print, do not move")
    ap.add_argument("--observer", default=None, help="observer camera device to record")
    args = ap.parse_args()

    path = intr_path()
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

        frame = robot.capture_wrist()
        if frame is None:
            print("no wrist camera frame")
            return 2
        det = detector(frame)
        if det is None and args.dry_run:
            print("cube not visible from the parked pose; a real run would sweep:")
            for cand in sweep_poses():
                print("   ", {k: round(v) for k, v in cand.items()})
            return 1
        if det is None:
            # A model-driven search cannot help here: it would need the cube's position,
            # which is exactly what we are looking for. Sweep instead -- a few poses spread
            # over the workspace, stop at the first that sees it.
            print("cube not visible from here; sweeping the workspace")
            for cand in sweep_poses():
                robot.command(cand, gripper=JOINTS.gripper_open_deg)
                det = detector(robot.capture_wrist())
                if det is not None:
                    print("found it from", {k: round(v, 1) for k, v in cand.items()})
                    break
            if det is None:
                print("the sweep did not see the cube -- put it where the arm can reach")
                return 1
            pose = robot.read_joints()
        est = robot.unproject_to_table(det["centre"][0], det["centre"][1], GRASP.table_z_m)
        cam_c, _ = camera_pose(pose)
        print(
            f"cube pixel {tuple(round(v) for v in det['centre'])}, "
            f"back-projected ({est[0] * 1000:.0f}, {est[1] * 1000:.0f}) mm"
        )
        print(
            f"camera at ({cam_c[0] * 1000:.0f}, {cam_c[1] * 1000:.0f}, {cam_c[2] * 1000:.0f}) mm, "
            f"intrinsics fx {intr['fy']:.0f}"
        )

        if args.dry_run:
            print("dry run: nothing moved")
            return 0

        result = grasp_measured(robot, detector, log=print)
        print("result:", result)
        return 0 if result.get("ok") else 1
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
