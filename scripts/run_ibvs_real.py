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
from dofbot_grasp.params import CAMERAS, GRASP  # noqa: E402
from dofbot_grasp.tracker import make_detector  # noqa: E402


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
        if det is None:
            print("cube not visible from here -- move the arm until it is")
            return 1
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
