#!/usr/bin/env python3
"""Hand-eye calibration: wrist camera -> robot base.

The board lies still on the desk; the arm moves and looks at it from several
poses. For each accepted pose we solve PnP (board -> camera) and read the joint
angles (-> forward kinematics), then solve AX = XB.

Usage
-----
Collect (moves the arm; each accepted pose is printed as it is stored):

    python scripts/handeye_calibrate.py collect \
        --intrinsics calib/intrinsics_wrist.json \
        --cols 6 --rows 4 --square-mm 11 \
        --out calib/handeye_views.json

Solve:

    python scripts/handeye_calibrate.py solve \
        --views calib/handeye_views.json \
        --out calib/handeye.json

The collect step only keeps poses where the whole board is detected, so more
poses than requested may be skipped. At least 4 poses are needed, 8+ preferred.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dofbot_grasp import kinematics as K  # noqa: E402
from dofbot_grasp.bus import (  # noqa: E402  # noqa: E402
    SERVO_RANGE_DEG,
    SerialBus,
    build_read_frame,
    build_single_servo_frame,
    pwm_to_servo_angle,
)

# Poses are described in servo degrees. The arm sweeps yaw and pitch so the board
# is seen from different directions; the exact values matter less than the spread.
POSES: list[dict[int, float]] = [
    # Orbit the board instead of translating away from it: keep the reach roughly
    # fixed and vary yaw and wrist pitch. CalibrateHandEye needs rotational
    # diversity on non-parallel axes; a set of near-identical pitch nudges makes
    # Tsai degenerate (it returns the identity).
    # Values are servo degrees; u4 = 90 + q4, and q4 saturates at -90 (u4 = 0).
    {1: 70.0, 2: 97.0, 3: 20.0, 4: 3.0},
    {1: 70.0, 2: 97.0, 3: 20.0, 4: 30.0},
    {1: 90.0, 2: 97.0, 3: 20.0, 4: 3.0},
    {1: 90.0, 2: 97.0, 3: 20.0, 4: 30.0},
    {1: 90.0, 2: 97.0, 3: 20.0, 4: 55.0},
    {1: 110.0, 2: 97.0, 3: 20.0, 4: 3.0},
    {1: 110.0, 2: 97.0, 3: 20.0, 4: 30.0},
    {1: 80.0, 2: 105.0, 3: 20.0, 4: 15.0},
    {1: 100.0, 2: 89.0, 3: 20.0, 4: 15.0},
    {1: 90.0, 2: 103.0, 3: 30.0, 4: 40.0},
]


def read_servo(bus: SerialBus, sid: int) -> float:
    for _ in range(2):
        bus._drain_input()
        bus._write(build_read_frame(sid))
        time.sleep(0.02)
        _t, data = bus._read_reply()
    angle = pwm_to_servo_angle(sid, (data[0] << 8) | data[1])
    return 180.0 - angle if sid in (2, 3, 4) else angle


def read_all(bus: SerialBus) -> list[float]:
    return [round(read_servo(bus, i), 1) for i in range(1, 7)]


def goto(bus: SerialBus, sid: int, target: float, tol: float = 2.0, step: float = 12.0) -> float:
    hi = SERVO_RANGE_DEG[sid - 1]
    target = float(np.clip(target, 1.0, hi - 1.0))
    for _ in range(14):
        cur = read_servo(bus, sid)
        err = target - cur
        if abs(err) <= tol:
            return cur
        nxt = float(np.clip(cur + max(-step, min(step, err)), 1.0, hi - 1.0))
        bus._write(build_single_servo_frame(sid, nxt, 1100))
        time.sleep(1.4)
    return read_servo(bus, sid)


def grab(device: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def pose_matrix(servo_deg: list[float]) -> np.ndarray:
    """T_base_gripper from the joint angles (TCP frame)."""
    q = np.radians(np.array(servo_deg[:4]) - 90.0)
    q5 = np.radians(servo_deg[4] - 135.0)
    T = K.forward(np.array([*q, q5]))
    out = np.eye(4)
    out[:3, :3] = T[:3, :3]
    out[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, 0.05])
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    intr = json.loads(Path(args.intrinsics).read_text())
    Kmtx = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    dist = np.array(intr["dist"], dtype=float).reshape(-1, 1)
    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * (
        args.square_mm / 1000.0
    )
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    views: list[dict] = []
    with SerialBus(port=args.port) as bus:
        for idx, target in enumerate(POSES):
            for sid, val in target.items():
                goto(bus, sid, val)
            time.sleep(0.4)
            frame = grab(args.camera)
            if frame is None:
                print(f"pose {idx}: no frame")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern, flags)
            if not found:
                print(f"pose {idx}: board not found")
                continue
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
            )
            ok, rvec, tvec = cv2.solvePnP(objp, corners, Kmtx, dist)
            if not ok:
                print(f"pose {idx}: solvePnP failed")
                continue
            angles = read_all(bus)
            views.append(
                {
                    "pose_index": idx,
                    "servo_deg": angles,
                    "rvec": rvec.reshape(-1).tolist(),
                    "tvec": tvec.reshape(-1).tolist(),
                    "corner_spread_px": float(corners[..., 0].std()),
                }
            )
            print(
                f"pose {idx}: accepted ({len(views)} total) spread={views[-1]['corner_spread_px']:.0f}px"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"views": views, "pattern": list(pattern), "square_mm": args.square_mm},
            indent=2,
        )
    )
    print(f"wrote {out} with {len(views)} views")
    return 0 if len(views) >= 4 else 1


def cmd_solve(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.views).read_text())
    views = data["views"]
    if len(views) < 3:
        print(f"error: need at least 3 views, got {len(views)}", file=sys.stderr)
        return 1

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for v in views:
        T = pose_matrix(v["servo_deg"])
        R_g2b.append(T[:3, :3])
        t_g2b.append(T[:3, 3].reshape(3, 1))
        rvec = np.array(v["rvec"], dtype=float).reshape(3, 1)
        R_t2c.append(cv2.Rodrigues(rvec)[0])
        t_t2c.append(np.array(v["tvec"], dtype=float).reshape(3, 1))

    method = cv2.CALIB_HAND_EYE_TSAI
    R_c2b, t_c2b = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)

    # residual: the board pose in base frame must be the same for every view
    positions = []
    for Rg, tg, Rc, tc in zip(R_g2b, t_g2b, R_t2c, t_t2c, strict=True):
        R_bc = Rg @ R_c2b
        t_bc = (Rg @ t_c2b.reshape(3, 1)) + tg
        positions.append((R_bc @ Rc, (R_bc @ tc) + t_bc))
    board_t = np.array([p[1].reshape(3) for p in positions])
    spread = board_t.std(axis=0)
    print(f"board origin in base frame per view (mm): {np.round(board_t * 1000, 1).tolist()}")
    print(f"spread (mm): {np.round(spread * 1000, 2).tolist()}")

    T = np.eye(4)
    T[:3, :3] = R_c2b
    T[:3, 3] = t_c2b.reshape(3)
    report = {
        "method": "CALIB_HAND_EYE_TSAI",
        "views": len(views),
        "T_base_camera": T.tolist(),
        "camera_in_base_mm": [round(float(v) * 1000, 2) for v in T[:3, 3]],
        "board_origin_spread_mm": [round(float(v) * 1000, 3) for v in spread],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    worst = float(spread.max()) * 1000
    print(
        f"worst axis spread {worst:.1f} mm -> "
        + ("good" if worst < 5 else "poor: recollect with more spread")
    )
    return 0 if worst < 10 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="move the arm and record board observations")
    c.add_argument("--port", default="/dev/ttyUSB0")
    c.add_argument("--camera", default="/dev/video0")
    c.add_argument("--intrinsics", default="calib/intrinsics_wrist.json")
    c.add_argument("--cols", type=int, default=6)
    c.add_argument("--rows", type=int, default=4)
    c.add_argument("--square-mm", type=float, default=11.0)
    c.add_argument("--out", default="calib/handeye_views.json")
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser("solve", help="solve AX=XB from the collected views")
    s.add_argument("--views", default="calib/handeye_views.json")
    s.add_argument("--out", default="calib/handeye.json")
    s.set_defaults(func=cmd_solve)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
