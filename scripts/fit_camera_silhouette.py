#!/usr/bin/env python3
"""Fit the observer camera pose by matching the rendered robot silhouette.

This is the model-based calibration the mesh actually enables: the MJCF is the
robot's real geometry (copied from the URDF), the joint angles are read from the
servos, so the only unknown is where the camera sits. We render the arm from a
candidate camera pose and maximise overlap with the arm's pixels in a real frame.

The real segmentation is restricted to the part of the image where the arm is
against a dark background (ceiling), so bright/dark is a reliable proxy for
arm/background there. Outside that region the cluttered office would dominate.

    python scripts/fit_camera_silhouette.py \
        --image /tmp/fit_pose2.png \
        --servo 90.1,95.7,19.6,3.3,45.1 \
        --roi 120,20,470,230 \
        --out calib/observer_extrinsics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", required=True)
    ap.add_argument("--servo", default="90.1,95.7,19.6,3.3,45.1")
    ap.add_argument("--roi", default="140,40,400,175", help="x0,y0,x1,y1 of the clean region")
    ap.add_argument("--vmin", type=float, default=105.0, help="brightness floor for the arm")
    ap.add_argument("--smax", type=float, default=80.0, help="saturation ceiling for the arm")
    ap.add_argument("--init-eye", default="0.37,0.18,0.50", help="metres, base frame")
    ap.add_argument("--init-lookat", default="0.0,0.03,0.16")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--out", default="calib/observer_extrinsics.json")
    args = ap.parse_args(argv)

    import mujoco
    import run_grasp_sim as sim

    x0, y0, x1, y1 = (int(v) for v in args.roi.split(","))
    real = cv2.imread(args.image)
    if real is None:
        print(f"error: cannot read {args.image}", file=sys.stderr)
        return 2
    # scale to the render size
    sx, sy = args.width / real.shape[1], args.height / real.shape[0]
    small = cv2.resize(real, (args.width, args.height))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    roi_mask = np.zeros((args.height, args.width), bool)
    roi_mask[int(y0 * sy) : int(y1 * sy), int(x0 * sx) : int(x1 * sx)] = True
    # bright = arm, dark = ceiling, inside the ROI only
    # the arm is bright *and* desaturated; the white wall is bright too, so the
    # saturation term plus a tight ROI keeps the wall and the yellow banner out
    real_arm = (hsv[..., 2] > args.vmin) & (hsv[..., 1] < args.smax) & roi_mask
    if real_arm.sum() < 200:
        print(f"error: only {int(real_arm.sum())} arm pixels in the ROI; widen it", file=sys.stderr)
        return 2
    print(f"reference: {int(real_arm.sum())} bright pixels in the ROI")

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

    # Only the visual mesh geoms: the simplified collision primitives are kept for
    # physics (invisible, alpha 0) and would otherwise pollute the silhouette.
    robot_geoms = np.array(
        [
            g
            for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_visual")
        ]
    )
    if robot_geoms.size == 0:  # fall back to any robot body
        robot_geoms = np.array(
            [
                g
                for g in range(model.ngeom)
                if model.body(model.geom_bodyid[g]).name not in ("world", "floor", "table")
            ]
        )
    print(f"silhouette geoms: {robot_geoms.tolist()}")

    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    cache: dict[str, object] = {}

    def arm_mask(eye: np.ndarray, lookat: np.ndarray) -> np.ndarray:
        delta = eye - lookat
        dist = float(np.linalg.norm(delta))
        cam.lookat[:] = lookat
        cam.distance = dist
        cam.azimuth = float(np.degrees(np.arctan2(delta[1], delta[0])))
        # MuJoCo's elevation is the camera's angle *below* the horizon: an eye
        # above the lookat needs a negative elevation (verified empirically).
        cam.elevation = float(-np.degrees(np.arcsin(np.clip(delta[2] / dist, -1, 1))))
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render()[:, :, 0]
        renderer.disable_segmentation_rendering()
        return np.isin(seg, robot_geoms)

    def cost(p: np.ndarray) -> float:
        eye = p[:3]
        lookat = p[3:]
        rendered = arm_mask(eye, lookat)
        inter = np.logical_and(rendered, real_arm).sum()
        union = np.logical_or(rendered, real_arm).sum()
        iou = float(inter) / max(float(union), 1.0)
        cache["iou"] = iou
        cache["p"] = p.copy()
        return -iou

    lookat0 = np.array([float(v) for v in args.init_lookat.split(",")])
    # coarse grid over the camera position with the lookat held at the arm centre:
    # Nelder-Mead alone drops into local minima when the initial guess is far off
    best = (0.0, None)
    for gx in np.arange(0.15, 0.80, 0.10):
        for gy in np.arange(-0.55, 0.60, 0.10):
            for gz in np.arange(0.20, 0.85, 0.10):
                rendered = arm_mask(np.array([gx, gy, gz]), lookat0)
                inter = np.logical_and(rendered, real_arm).sum()
                union = np.logical_or(rendered, real_arm).sum()
                v = float(inter) / max(float(union), 1.0)
                if v > best[0]:
                    best = (v, np.array([gx, gy, gz]))
    print(f"grid best IoU = {best[0]:.3f} at eye {np.round(best[1] * 1000, 0).tolist()} mm")
    p0 = np.array([*best[1], *lookat0])
    print(f"start IoU = {-cost(p0):.3f}")
    res = minimize(
        cost, p0, method="Nelder-Mead", options={"maxiter": 900, "xatol": 1e-3, "fatol": 1e-4}
    )
    eye, lookat = res.x[:3], res.x[3:]
    iou = -float(res.fun)
    print(f"final IoU = {iou:.3f}")
    print(f"camera eye in base (mm): {np.round(eye * 1000, 1).tolist()}")
    print(f"lookat    in base (mm): {np.round(lookat * 1000, 1).tolist()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "image": args.image,
                "servo_deg": servo,
                "iou": round(iou, 4),
                "eye_m": [round(float(v), 5) for v in eye],
                "lookat_m": [round(float(v), 5) for v in lookat],
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    return 0 if iou > 0.45 else 1


if __name__ == "__main__":
    raise SystemExit(main())
