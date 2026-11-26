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
    (0.100, 0.140),  # every seed must be somewhere the wrist camera can actually see
]
TASK_POSE = {1: 90.0 - 10.0, 2: 90.0 + 8.0, 3: 90.0 - 88.0, 4: 90.0 - 76.0, 5: 135.0}


def joint_angles_of(servo: dict[int, float]) -> np.ndarray:
    """Servo degrees -> joint radians, for seeding the IK."""
    return np.radians(np.array([servo[i] - (90.0 if i < 5 else 135.0) for i in range(1, 6)]))


def visible_pose_candidates(seed: tuple[float, float], limit: int = 12) -> list[dict[int, float]]:
    """Ranked joint poses that should put ``seed`` in view, best first.

    The analytic projection is only good to a few tens of pixels, so a pose can pass the
    test and still render without the cube. The caller renders and confirms.
    """
    from dofbot_grasp.controller import camera_pose, tip_and_axis

    intr = real_wrist_intrinsics()
    f, cx, cy = intr["fy"], intr["cx"], intr["cy"]
    scored: list[tuple[float, dict[int, float]]] = []
    for j1 in range(40, 141, 10):
        for j2 in range(40, 141, 5):
            for j3 in range(-40, 81, 10):
                pose = {
                    1: float(j1),
                    2: float(j2),
                    3: float(j3),
                    4: float(j2 + j3 - 90.0),
                    5: 135.0,
                }
                tip, _ = tip_and_axis(pose)
                if tip[2] < 0.040:
                    continue
                C, R = camera_pose(pose)
                pc = (np.array([seed[0], seed[1], 0.015]) - C) @ R
                depth = -pc[2]
                if depth <= 1e-6:
                    continue
                u = cx + f * pc[0] / depth
                v = cy - f * pc[1] / depth
                if not (30.0 < u < 610.0 and 30.0 < v < 450.0):
                    continue
                # being able to see the cube is not enough: the grasp pose must also be
                # reachable from here, and poses that see the cube well tend to sit in
                # configurations from which the vertical grasp is 30 mm out of reach
                from dofbot_grasp import kinematics as K
                from dofbot_grasp.params import JOINTS

                goal = np.eye(4)
                goal[:3, :3] = np.diag([1.0, -1.0, -1.0])
                gz = 0.010
                goal[:3, 3] = np.array(
                    [seed[0], seed[1], gz]
                ) - JOINTS.fingertip_below_tcp_m * np.array([0.0, 0.0, -1.0])
                q_goal = K.inverse(goal, seed=joint_angles_of(pose))
                Tf = K.forward(q_goal)
                tip_goal = Tf[:3, 3] + JOINTS.fingertip_below_tcp_m * (
                    Tf[:3, :3] @ np.array([0.0, 0.0, 1.0])
                )
                reach = float(np.linalg.norm(tip_goal - np.array([seed[0], seed[1], gz])))
                if reach > 0.012:
                    continue
                d = float(np.hypot(u - 0.4 * 640.0, v - 0.6 * 480.0))
                scored.append((d, pose))
    scored.sort(key=lambda kv: kv[0])
    return [p for _d, p in scored[:limit]]


def find_visible_pose(seed: tuple[float, float]) -> dict[int, float]:
    """A joint pose from which the wrist camera can see ``seed``, at a safe height."""
    from dofbot_grasp.controller import camera_pose, tip_and_axis

    # the measured wrist intrinsics, matching what the renderer produces -- using the old
    # 320x240 / 90 deg numbers here found poses the real camera cannot actually see
    intr = real_wrist_intrinsics()
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
                # the fingertips hang 40 mm below the palm now, so clearing the cube's
                # 30 mm top is the binding constraint, not a tidy "high" pose
                if tip[2] < 0.040:
                    continue
                C, R = camera_pose(pose)
                pc = (np.array([seed[0], seed[1], 0.015]) - C) @ R
                depth = -pc[2]  # MuJoCo cameras look along -z
                if depth <= 1e-6:
                    continue
                u = cx + f * pc[0] / depth
                v = cy - f * pc[1] / depth
                if not (20.0 < u < 620.0 and 20.0 < v < 460.0):
                    continue
                d = float(np.hypot(u - 0.35 * 640.0, v - 0.55 * 480.0))
                if best is None or d < best[0]:
                    best = (d, pose)
    return best[1] if best is not None else dict(TASK_POSE)


def real_wrist_intrinsics() -> dict:
    """The measured wrist camera, which the recorded frames are rendered through."""
    import json

    return json.loads((ROOT / "calib" / "intrinsics_wrist.json").read_text())


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

    from dofbot_grasp.controller import grasp_measured
    from dofbot_grasp.hardware import SimRobot

    model = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "dofbot_se.xml"))
    data = mujoco.MjData(model)
    sys.path.insert(0, str(ROOT / "sim"))
    from real_camera import RealCamera

    io = SimRobot(
        model,
        data,
        img_w=640,
        img_h=480,
        real_cameras={"wrist_cam": RealCamera(ROOT / "calib" / "intrinsics_wrist.json")},
    )

    jadr = data.qpos
    cube = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
    jadr[cube : cube + 3] = [seed[0], seed[1], 0.015]
    # start where the camera can actually see the cube. The analytic search only ranks
    # candidates -- the projection is good to tens of pixels -- so each one is rendered and
    # the detector decides, which is what a real search would do.
    detector = make_detector(prefer_gpu=use_gpu)
    start_pose = None
    for cand in visible_pose_candidates(seed):
        for i in range(1, 6):
            jadr[
                model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
            ] = math.radians(cand[i] - (90.0 if i < 5 else 135.0))
        mujoco.mj_forward(model, data)
        if detector(io.capture_wrist()) is not None:
            start_pose = cand
            break
    if start_pose is None:
        return {
            "ok": False,
            "reason": "no pose sees the cube",
            "seed": seed,
            "lift_mm": 0.0,
            "drift_mm": 0.0,
        }
    mujoco.mj_forward(model, data)

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    before = float(data.xpos[cube_body][2])
    seen = {}

    def on_stage(tag):
        # remember where the cube is when the jaws close: from that moment it travels with
        # the gripper, so its final position says nothing about pushing
        if tag == "CLOSE":
            seen["close"] = data.xpos[cube_body].copy()

    result = grasp_measured(
        io,
        detector,
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
