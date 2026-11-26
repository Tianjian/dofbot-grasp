"""Run the shared controller over several cube positions and report the success rate.

This is the CI gate: `python sim/grasp_trial.py --min-rate 0.75` exits non-zero when the
simulated grasp rate drops below the threshold, so a change that quietly breaks the
controller cannot land.

It drives ``dofbot_grasp.controller`` through ``SimRobot`` -- the same phase code the arm
will run -- so the gate measures the shipped sequence, not a copy of it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEEDS = [
    (0.020, 0.140),
    (-0.060, 0.140),
    (0.060, 0.140),
    (0.040, 0.100),  # close in: 108 mm radius; nearer than this the centring fails
]
TASK_POSE = {1: 90.0 - 10.0, 2: 90.0 + 8.0, 3: 90.0 - 88.0, 4: 90.0 - 76.0, 5: 135.0}


def joint_angles_of(servo: dict[int, float]) -> np.ndarray:
    """Servo degrees -> joint radians, for seeding the IK."""
    return np.radians(np.array([servo[i] - (90.0 if i < 5 else 135.0) for i in range(1, 6)]))


def find_visible_pose(seed: tuple[float, float]) -> dict[int, float]:
    """A joint pose from which the wrist camera can see ``seed``, at a safe height."""
    from dofbot_grasp.controller import camera_pose, tip_and_axis

    intr = sim_intrinsics()
    f, cx, cy = intr["fy"], intr["cx"], intr["cy"]
    best: tuple[float, dict[int, float]] | None = None
    for j1 in range(50, 131, 10):
        for j2 in range(70, 126, 10):
            for j3 in range(-10, 61, 10):
                pose = {
                    1: float(j1),
                    2: float(j2),
                    3: float(j3),
                    4: float(j2 + j3 - 90.0),
                    5: 135.0,
                }
                tip, _ = tip_and_axis(pose)
                if not (0.05 < tip[2] < 0.20):
                    continue
                C, R = camera_pose(pose)
                pc = (np.array([seed[0], seed[1], 0.015]) - C) @ R
                if pc[2] <= 1e-6:
                    continue
                u = cx + f * pc[0] / pc[2]
                v = cy + f * pc[1] / pc[2]
                if not (20.0 < u < 620.0 and 20.0 < v < 460.0):
                    continue
                d = float(np.hypot(u - 0.25 * intr["width"], v - 0.25 * intr["height"]))
                if best is None or d < best[0]:
                    best = (d, pose)
    return best[1] if best is not None else dict(TASK_POSE)


def sim_intrinsics(width: int = 320, height: int = 240, fovy_deg: float = 90.0) -> dict:
    """The simulated wrist camera is a pinhole with a declared vertical field of view."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return {
        "fx": f,
        "fy": f,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "width": width,
        "height": height,
    }


def make_detector(prefer_gpu: bool = False):
    """Detector for the simulated frames.

    SAM2 is available with ``--detector sam2`` where a GPU and torch exist; the CI runner
    has neither, so the default stays the cheap RGB rule the simulator renders for.

    The rendered cube is an unlit orange-brown; ``perception.detect_cube`` thresholds HSV
    saturation for the real camera and does not fire on it, so the simulator keeps its own
    simple rule.
    """
    if prefer_gpu:
        from dofbot_grasp.tracker import make_detector as front_end

        return front_end(prefer_gpu=True)

    sys.path.insert(0, str(ROOT / "sim"))
    from sim_detector import detect_cube

    def detect(frame):
        if frame is None:
            return None
        det = detect_cube(frame)
        if det is None:
            return None
        x, y, w, h = det["bbox"]
        out = {"centre": tuple(det["centre"]), "area": float(w * h)}
        r = frame[:, :, 0].astype(np.int16)
        g = frame[:, :, 1].astype(np.int16)
        b = frame[:, :, 2].astype(np.int16)
        out["mask"] = ((r > 90) & (r > g + 25) & (g > b) & (b < 140)).astype(np.uint8)
        return out

    return detect


def run(seed, *, verbose: bool = False, use_gpu: bool = False, jaw_offset: float = 0.0) -> dict:
    import mujoco

    from dofbot_grasp.controller import grasp
    from dofbot_grasp.hardware import SimRobot

    model = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "dofbot_se.xml"))
    data = mujoco.MjData(model)
    io = SimRobot(model, data)

    jadr = data.qpos
    cube = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
    jadr[cube : cube + 3] = [seed[0], seed[1], 0.015]
    for i in range(1, 6):
        jadr[
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
        ] = math.radians(TASK_POSE[i] - 90.0 if i < 5 else TASK_POSE[i] - 135.0)
    mujoco.mj_forward(model, data)

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    before = float(data.xpos[cube_body][2])
    seen = {}

    def on_stage(tag):
        # remember where the cube is when the jaws close: from that moment it travels with
        # the gripper, so its final position says nothing about pushing
        if tag == "CLOSE":
            seen["close"] = data.xpos[cube_body].copy()

    result = grasp(
        io,
        make_detector(prefer_gpu=use_gpu),
        jaw_offset_deg=jaw_offset,
        on_stage=on_stage,
        log=print if verbose else None,
    )
    after = float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")][2])
    lift = (after - before) * 1000
    # The user's criterion: lifting the cube is the success, whether or not the fingers
    # had to shove it along the way. ``drift_mm`` still reports how far the cube travelled
    # before the jaws closed -- useful signal, not a pass/fail gate.
    at_close = seen.get("close")
    drift = (
        float(np.linalg.norm((at_close[:2] - np.asarray(seed))[:2])) * 1000
        if at_close is not None
        else 0.0
    )
    dragged = float(drift)
    result.update(
        {
            "seed": seed,
            "lift_mm": lift,
            "drift_mm": dragged,
            "ok": bool(result.get("ok") and lift > 20),
        }
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rate", type=float, default=0.75)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--detector", choices=["colour", "sam2"], default="colour")
    ap.add_argument(
        "--jaw-offset",
        type=float,
        default=0.0,
        help="extra quarter turn of the gripper base (degrees)",
    )
    ap.add_argument("--json", default=None, help="write the per-seed results here")
    args = ap.parse_args()

    rows = [
        run(s, verbose=args.verbose, use_gpu=args.detector == "sam2", jaw_offset=args.jaw_offset)
        for s in SEEDS
    ]
    wins = sum(r["ok"] for r in rows)
    rate = wins / len(rows)
    for r in rows:
        print(
            f"  cube at {r['seed']} -> ok={r['ok']}  lift={r['lift_mm']:+.1f} mm"
            f"  {r.get('reason', '')}"
        )
    print(f"\ngrasp success rate: {wins}/{len(rows)} = {rate:.2f}  (threshold {args.min_rate:.2f})")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0 if rate >= args.min_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
