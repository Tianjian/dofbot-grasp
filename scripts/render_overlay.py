#!/usr/bin/env python3
"""Render the robot model from a candidate camera pose and overlay a real frame.

The URDF/mesh in the three.js viewer and the MJCF in ``sim/`` describe the same
geometry, so rendering the model is a way to *verify* a calibration: if the
silhouette lands on the real arm, the kinematics and the camera extrinsics agree.
It cannot *supply* the camera mount by itself -- that transform is the unknown.

    python scripts/render_overlay.py \
        --image /tmp/gate/observer.png \
        --servo 90,97,20,3,135 \
        --camera-pos -0.35,0.28,0.25 --lookat 0,0.1,0.10 \
        --out /tmp/overlay.png

``--camera-pos``/``--lookat`` are in the robot base frame (metres). Once
``calib/handeye.json`` exists, pass ``--extrinsics calib/handeye.json`` instead to
use the solved camera pose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", default=None, help="real frame to overlay onto")
    ap.add_argument("--servo", default="90,90,90,90,135", help="six servo angles, degrees")
    ap.add_argument("--camera-pos", default=None, help="x,y,z in metres (base frame)")
    ap.add_argument("--lookat", default="0,0.10,0.10", help="x,y,z in metres")
    ap.add_argument("--extrinsics", default=None, help="JSON with T_base_camera (4x4)")
    ap.add_argument("--out", default="/tmp/overlay.png")
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args(argv)

    import mujoco
    import run_grasp_sim as sim

    model = mujoco.MjModel.from_xml_path(str(sim.XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joints = ["joint1", "joint2", "joint3", "joint4", "joint5"]
    servo = [float(v) for v in args.servo.split(",")]
    q = np.radians(np.array(servo[:4]) - 90.0)
    q5 = np.radians(servo[4] - 135.0)
    for name, value in zip(joints, [*q, q5], strict=True):
        data.qpos[sim.qadr(model, name)] = value
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(v) for v in args.lookat.split(",")]
    if args.extrinsics:
        T = np.array(json.loads(Path(args.extrinsics).read_text())["T_base_camera"], dtype=float)
        cam.distance = 0.45
        cam.azimuth = 0.0
        cam.elevation = 0.0
        print(
            f"note: --extrinsics given, but this viewer only draws a free camera "
            f"(camera at {np.round(T[:3, 3], 3)} m)"
        )
    elif args.camera_pos:
        pos = np.array([float(v) for v in args.camera_pos.split(",")])
        lookat = np.array([float(v) for v in args.lookat.split(",")])
        delta = pos - lookat
        cam.distance = float(np.linalg.norm(delta))
        cam.azimuth = float(np.degrees(np.arctan2(delta[1], delta[0])))
        cam.elevation = float(np.degrees(np.arcsin(delta[2] / cam.distance)))
    else:
        cam.distance = 0.45
        cam.azimuth = 135.0
        cam.elevation = -20.0

    renderer = mujoco.Renderer(model, 480, 640)
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()

    if args.image:
        real = cv2.imread(args.image)
        if real is None:
            print(f"error: cannot read {args.image}", file=sys.stderr)
            return 2
        real = cv2.resize(real, (640, 480))
        overlay = cv2.addWeighted(real, 1.0 - args.alpha, rgb[:, :, ::-1], args.alpha, 0)
    else:
        overlay = rgb[:, :, ::-1]

    cv2.imwrite(args.out, overlay)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
