#!/usr/bin/env python3
"""Scripted wrist-camera grasp in MuJoCo — the reference motion for the real arm.

Solves the reach with MuJoCo's Jacobian (no hand-tuned angles), drives the
position actuators the way the servos are driven on hardware, and reports whether
the cube actually left the table.

Usage:
    .venv/bin/python dofbot_grasp/sim/run_grasp_sim.py --out /tmp/grasp_frames
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).with_name("dofbot_se.xml")
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]


def jid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def qadr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[jid(model, name)])


def ik_tcp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    zdes: np.ndarray,
    seed: np.ndarray,
    iters: int = 80,
    lam: float = 0.05,
) -> np.ndarray:
    """Local damped-least-squares IK on TCP position + tool z-axis.

    Deliberately local (few iterations, high damping): the arm has several
    kinematic branches and a global solve picks a folded one that self-collides.
    """
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    adr = [qadr(model, n) for n in ARM_JOINTS]
    lim = np.array([model.jnt_range[jid(model, n)] for n in ARM_JOINTS])
    q = seed.copy()
    for _ in range(iters):
        for i, a in enumerate(adr):
            data.qpos[a] = q[i]
        mujoco.mj_forward(model, data)
        err = np.concatenate(
            [
                target - data.site_xpos[site],
                0.02 * np.cross(data.site_xmat[site].reshape(3, 3)[:, 2], zdes),
            ]
        )
        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        jac = np.vstack([jacp[:, adr], 0.02 * jacr[:, adr]])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(jac.shape[0]), err)
        q = np.clip(q + dq, lim[:, 0], lim[:, 1])
    return q


def drive(model, data, q_target, gripper, steps, renderer=None, frames=None, every=40, setup=None):
    """Move the servos to q_target with a linear interpolation, like a timed servo write."""
    start = np.array([data.ctrl[i] for i in range(5)])
    for s in range(steps):
        a = s / max(steps - 1, 1)
        for i in range(5):
            data.ctrl[i] = start[i] + a * (q_target[i] - start[i])
        data.ctrl[5] = gripper
        data.ctrl[6] = -gripper
        mujoco.mj_step(model, data)
        if renderer is not None and frames is not None and s % every == 0:
            if setup:
                setup()
            renderer.update_scene(data, camera="wrist_cam")
            frames.append(renderer.render())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dofbot_grasp_frames")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cube = model.body("cube").id
    cube0 = data.xpos[cube].copy()
    z_down = np.array([0.0, 0.0, -1.0])

    # natural folded-down branch, found by sampling the reachable set
    seed = np.array([-0.058, 0.909, 0.987, 1.091, 1.127])
    # 1) pre-grasp: 8 cm above the cube, tool pointing down
    pre = ik_tcp(model, data, cube0 + np.array([0, 0, 0.08]), z_down, seed)
    # 2) grasp: TCP at cube centre
    grasp = ik_tcp(model, data, cube0 + np.array([0, 0, 0.004]), z_down, pre)
    # 3) lift
    lift = ik_tcp(model, data, cube0 + np.array([0, 0, 0.12]), z_down, grasp)

    frames = []
    renderer = None if args.headless else mujoco.Renderer(model, 480, 640)

    open_g, closed_g = 0.010, -0.002
    drive(model, data, pre, open_g, 1800, renderer, frames)
    drive(model, data, grasp, open_g, 900, renderer, frames)
    drive(model, data, grasp, closed_g, 600, renderer, frames)
    drive(model, data, lift, closed_g, 900, renderer, frames)
    for _ in range(400):
        mujoco.mj_step(model, data)
        if renderer is not None:
            renderer.update_scene(data, camera="wrist_cam")
            frames.append(renderer.render())

    cube_final = data.xpos[cube].copy()
    lifted_mm = (cube_final[2] - cube0[2]) * 1000.0
    contacts = 0
    for c in data.contact[: data.ncon]:
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        names = {model.body(b).name for b in (b1, b2)}
        if "cube" in names and any("finger" in n for n in names):
            contacts += 1

    result = {
        "q_pre": pre.tolist(),
        "q_grasp": grasp.tolist(),
        "q_lift": lift.tolist(),
        "cube_start_m": cube0.tolist(),
        "cube_final_m": cube_final.tolist(),
        "lifted_mm": lifted_mm,
        "finger_cube_contacts": contacts,
        "verdict": "PASS" if lifted_mm > 20.0 and contacts >= 1 else "FAIL",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    if frames:
        from PIL import Image

        for i, f in enumerate(frames[::6]):
            Image.fromarray(f).save(out / f"frame_{i:04d}.png")
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
