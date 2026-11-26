"""Record the shared controller in simulation, frame by frame.

Drives ``dofbot_grasp.controller.grasp`` through ``SimRobot`` -- the same code the arm will
run -- and captures both cameras by wrapping ``mujoco.mj_step``. Recording after each whole
phase gave a handful of frames; stepping capture shows what the arm actually does between
them, which is where the current failures live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "sim"))

import grasp_trial as trial  # noqa: E402

from dofbot_grasp.controller import grasp, tip_and_axis  # noqa: E402
from dofbot_grasp.hardware import SimRobot  # noqa: E402

EVERY = 100  # sim steps per captured frame (dt = 2 ms -> 0.2 s)
FPS = 30.0  # 0.2 s per frame at 30 fps = 6x real time (2x the previous cut)
W, H = 480, 360


def label(img, text, foot=""):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if foot:
        cv2.rectangle(out, (0, out.shape[0] - 22), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            out,
            foot,
            (8, out.shape[0] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 255, 200),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sim" / "controller_sim.mp4"))
    ap.add_argument("--seeds", default="0,1,2,3,", help="indices into grasp_trial.SEEDS")
    ap.add_argument("--jaw-offset", type=float, default=0.0)
    args = ap.parse_args()

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H))
    model = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "dofbot_se.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    state = {"n": 0, "phase": "", "seed": ""}

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    def current_info() -> str:
        """Fingertip, cube and their table-plane separation -- the drift is invisible."""
        try:
            pose = {}
            for i in range(1, 6):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                pose[i] = float(np.degrees(data.qpos[model.jnt_qposadr[jid]])) + (
                    90.0 if i < 5 else 135.0
                )
            tip, _ = tip_and_axis(pose)
            cube = data.xpos[cube_id]
            off = float(np.linalg.norm(tip[:2] - cube[:2])) * 1000
            return (
                f"tip ({tip[0] * 1000:.0f},{tip[1] * 1000:.0f},{tip[2] * 1000:.0f})  "
                f"cube ({cube[0] * 1000:.0f},{cube[1] * 1000:.0f},{cube[2] * 1000:.0f})  "
                f"xy gap {off:.0f} mm"
            )
        except Exception as exc:  # never let a label break a recording
            return f"(label error: {type(exc).__name__})"

    def grab():
        views = []
        for cam in ("wrist_cam", "overview"):
            renderer.update_scene(data, camera=cam)
            views.append(cv2.resize(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR), (W, H)))
        info = current_info()
        a = label(views[0], f"{state['seed']}  wrist  {state['phase']}", info)
        b = label(views[1], f"{state['phase']}  t={data.time:.0f}s", info)
        writer.write(np.hstack([a, b]))

    real_step = mujoco.mj_step

    def step_and_capture(m, d, nstep=1):
        real_step(m, d, nstep)
        state["n"] += 1
        if state["n"] % EVERY == 0:
            grab()

    mujoco.mj_step = step_and_capture
    io = SimRobot(model, data)
    try:
        for tag in [t for t in args.seeds.split(",") if t.strip()]:
            idx = int(tag)
            seed = trial.SEEDS[idx]
            state["seed"] = f"seed{idx} {seed}"
            cube = model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
            ]
            data.qpos[cube : cube + 3] = [seed[0], seed[1], 0.015]
            for i in range(1, 6):
                data.qpos[
                    model.jnt_qposadr[
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                    ]
                ] = np.radians(trial.TASK_POSE[i] - (90.0 if i < 5 else 135.0))
            mujoco.mj_forward(model, data)
            state["phase"] = "start"
            grab()
            res = grasp(
                io,
                trial.make_detector(),
                jaw_offset_deg=args.jaw_offset,
                log=lambda m: state.update(phase=m.strip()[:28]),
            )
            state["phase"] = f"done ok={res.get('ok')} {res.get('reason', '')}"
            # hold the final state for a moment so the clip ends on the result instead of
            # cutting off mid-motion
            for _ in range(int(FPS * 1.2)):
                grab()
            print(f"  {state['seed']}: {res}")
    finally:
        mujoco.mj_step = real_step
        writer.release()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
