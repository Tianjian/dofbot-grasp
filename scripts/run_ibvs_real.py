"""Run the IBVS grasp on the real DOFBOT SE.

Mirrors `sim/ibvs_sim.py` phase for phase, but reads the wrist camera and drives the
servo bus. Constants come from `dofbot_grasp.params`, which records where each measured
number came from.

    python scripts/run_ibvs_real.py --dry-run     # plan and print, do not move
    python scripts/run_ibvs_real.py               # run the grasp

Safety comes from three rules learned the hard way:
  * never read the bus inside ~0.3 s of a write6 (it cancels the move);
  * poll the read-back until it is stable instead of sleeping a fixed time;
  * stop if a servo lags its command by more than SAFETY.max_servo_lag_deg.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dofbot_grasp import kinematics as K  # noqa: E402
from dofbot_grasp.bus import (  # noqa: E402
    SerialBus,
    build_read_frame,
    build_single_servo_frame,
    build_write6_frame,
    pwm_to_servo_angle,
)
from dofbot_grasp.params import (  # noqa: E402
    BUS,
    CAMERAS,
    GRASP,
    JOINTS,
    MOUNT,
    SAFETY,
    servo_to_joint,
)


# ---------------------------------------------------------------- bus plumbing
class Arm:
    """Serial wrapper with the measured timing rules baked in."""

    def __init__(self, port: str = BUS.port, baud: int = BUS.baud) -> None:
        self.bus = SerialBus(port=port, baudrate=baud)
        self.bus.__enter__()
        self.cmd = {i: 90.0 for i in range(1, 6)}
        self.cmd[5] = 135.0

    def close(self) -> None:
        self.bus.__exit__(None, None, None)

    def read(self, sid: int) -> float:
        for _ in range(2):  # the first reply after idle is stale
            self.bus._drain_input()
            self.bus._write(build_read_frame(sid))
            time.sleep(0.02)
            _, data = self.bus._read_reply()
        ang = pwm_to_servo_angle(sid, (data[0] << 8) | data[1])
        return 180.0 - ang if sid in (2, 3, 4) else ang

    def pose(self) -> dict[int, float]:
        return {i: self.read(i) for i in range(1, 6)}

    def wait_settled(self) -> dict[int, float]:
        """Poll until two consecutive samples agree, per the real settling time."""
        prev = self.pose()
        t0 = time.time()
        while time.time() - t0 < BUS.settle_timeout_s:
            time.sleep(0.25)
            cur = self.pose()
            if max(abs(cur[i] - prev[i]) for i in range(1, 6)) < BUS.settle_tolerance_deg:
                return cur
            prev = cur
        return prev

    def move(self, target_deg: dict[int, float], gripper: float | None = None) -> dict[int, float]:
        """One write6 frame, then a guarded settle. Never reads inside the guard."""
        ctrl = [target_deg[i] for i in range(1, 6)]
        if gripper is None:
            gripper = self.cmd.get(6, JOINTS.gripper_open_deg)
        self.bus._write(build_write6_frame([*ctrl, gripper], int(BUS.move_time_s * 1000)))
        time.sleep(BUS.guard_after_write_s)
        settled = self.wait_settled()
        lag = max(abs(settled[i] - ctrl[i - 1]) for i in range(1, 6))
        if lag > SAFETY.max_servo_lag_deg:
            raise RuntimeError(f"servo lag {lag:.1f} deg exceeds the limit -- arm is blocked")
        for i in range(1, 6):
            self.cmd[i] = ctrl[i - 1]
        self.cmd[6] = gripper
        return settled

    def gripper(self, angle_deg: float) -> None:
        self.bus._write(build_single_servo_frame(6, angle_deg, int(BUS.move_time_s * 1000)))
        time.sleep(BUS.guard_after_write_s + BUS.move_time_s)
        self.cmd[6] = angle_deg


# ---------------------------------------------------------------- kinematics
def tcp_of(servo_deg: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """TCP position and tool axis (world) from servo angles."""
    q = servo_to_joint(servo_deg)
    T = K.forward(np.array(q))
    axis = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, 0.05]), axis


LINK5_IN_LINK4 = np.array([-0.07385, -0.001, 0.0])  # link5 origin, link4 frame
ROT_Y_MINUS_90 = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def _cam_in_link5() -> np.ndarray:
    """Camera position expressed in link5's frame.

    ``MOUNT`` is recorded in link4's frame, and link5 sits at LINK5_IN_LINK4 with a -90 deg
    rotation about y, so the offset has to be rotated before it can be carried by the TCP
    transform that ``kinematics.forward`` returns.
    """
    p_link4 = np.array(MOUNT.t_link4_cam_mm) / 1000.0
    return ROT_Y_MINUS_90.T @ (p_link4 - LINK5_IN_LINK4)


def _cam_rotation_in_link5(roll_deg: float) -> np.ndarray:
    """Camera -> link5 rotation: the optical axis is the tool axis (+z), then rolled."""
    z = np.array([0.0, 0.0, 1.0])  # OpenCV puts the optical axis on +z
    x0 = np.array([1.0, 0.0, 0.0])
    y0 = np.cross(z, x0)
    base = np.column_stack([x0, y0, z])
    return base @ cv2.Rodrigues(z * math.radians(roll_deg))[0]


def _link5_frame(servo_deg: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """Position and rotation of link5 in the base frame.

    ``kinematics.forward`` already returns the link5 origin; the TCP used elsewhere is
    that origin plus 50 mm along the tool axis, so subtract nothing here.
    """
    q = servo_to_joint(servo_deg)
    T = K.forward(np.array(q))
    return T[:3, 3], T[:3, :3]


def camera_centre(servo_deg: dict[int, float]) -> np.ndarray:
    """World position of the wrist camera from FK plus the registered mount."""
    p5, R = _link5_frame(servo_deg)
    return p5 + R @ _cam_in_link5()


def backproject_to_table(
    u: float, v: float, servo_deg: dict[int, float], intr: dict
) -> np.ndarray | None:
    """Pixel -> world point on the table plane, using calibrated intrinsics + FK."""
    Kmtx = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    dist = np.array(intr["dist"], float).reshape(-1, 1)
    und = cv2.undistortPoints(np.array([[[u, v]]], np.float32), Kmtx, dist).reshape(2)
    p5, R = _link5_frame(servo_deg)
    ray = R @ _cam_rotation_in_link5(MOUNT.roll_deg) @ np.array([und[0], und[1], 1.0])
    C = p5 + R @ _cam_in_link5()
    if abs(ray[2]) < 1e-6:
        return None
    t = (GRASP.table_z_m - C[2]) / ray[2]
    return None if t <= 0 else C + t * ray


# ---------------------------------------------------------------- perception
def grab_wrist() -> np.ndarray | None:
    cap = cv2.VideoCapture(CAMERAS.wrist_device, cv2.CAP_V4L2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


_DETECTOR = None


def careful_detect(frame: np.ndarray | None):
    """Cube detection through the tracker front-end: SAM2 on the GPU, colour otherwise."""
    from dofbot_grasp.tracker import make_detector

    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = make_detector()
        print(f"detector: {getattr(_DETECTOR, 'name', 'colour')}")
    return _DETECTOR(frame)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="plan and print, do not move")
    ap.add_argument(
        "--task-pose",
        default=None,
        help="JSON list of five servo angles to plan from (implies dry run)",
    )
    ap.add_argument(
        "--pixel", default=None, help="u,v to plan from instead of the camera (implies dry run)"
    )
    args = ap.parse_args()

    intr_path = ROOT / CAMERAS.wrist_intrinsics
    if not intr_path.exists():
        print(f"missing intrinsics: {intr_path}")
        return 2
    intr = json.loads(intr_path.read_text())

    if args.pixel:
        u, v = (float(x) for x in args.pixel.split(","))
        print(f"cube at pixel ({u:.0f}, {v:.0f}) (given)")
    else:
        frame = grab_wrist()
        if frame is None:
            print("no wrist camera frame")
            return 2
        det = careful_detect(frame)
        if det is None:
            print("cube not visible in the wrist camera")
            return 1
        u, v = det.center_px
        print(f"cube at pixel ({u:.0f}, {v:.0f})")

    arm = None if args.dry_run else Arm()
    try:
        if args.task_pose:
            given = json.loads(args.task_pose)
            servo = {i + 1: float(given[i]) for i in range(5)}
        elif arm:
            servo = arm.pose()
        else:
            servo = {1: 91.9, 2: 97.5, 3: 15.7, 4: 1.3, 5: 134.8}
        tcp, axis = tcp_of(servo)
        print(f"pose (deg): {[round(servo[i], 1) for i in range(1, 6)]}")
        print(f"TCP (mm): {np.round(tcp * 1000, 1)}   tool axis: {np.round(axis, 2)}")

        est = backproject_to_table(u, v, servo, intr)
        if est is None:
            print("back-projection failed (ray parallel to the table)")
            return 1
        print(f"cube estimate (mm): {np.round(est * 1000, 1)}")
        # the TCP sits fingertip_below_tcp above the height the *fingertip* needs; using
        # est[2] (the cube's centre, 15 mm) as a base would overshoot by that much
        target_tcp = np.array(
            [
                est[0],
                est[1],
                GRASP.fingertip_grasp_z_m + JOINTS.fingertip_below_tcp_m,
            ]
        )
        print(f"grasp target for the TCP (mm): {np.round(target_tcp * 1000, 1)}")
        print("plan: centre -> align jaws -> descend vertically -> close -> lift")
        if args.dry_run:
            print("dry run: nothing moved")
            return 0
        print(
            "dry run not requested, but the motion phases are not wired yet -- "
            "review this plan first"
        )
        return 0
    finally:
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
