"""Record the IBVS run in MuJoCo: wrist view (the feedback) beside an external view.

Produces a side-by-side mp4 so the control behaviour can be reviewed visually:
left = what the controller sees, right = what the arm is actually doing.
"""

from __future__ import annotations

import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, "/home/toby/Desktop/dofbot-grasp/sim")

import ibvs_sim as S  # noqa: E402

OUT = "/home/toby/Desktop/dofbot-grasp/sim/ibvs_sim.mp4"
SEED = (-0.030, -0.250)
W, H = 480, 360


def make_scene(model, data, renderer, cam_id):
    """MuJoCo renders RGB. Keep it RGB for the detector; shoot() converts for video."""
    renderer.update_scene(data, camera=cam_id)
    return renderer.render().copy()


def label(img, text, colour=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return out


def main():
    model = mujoco.MjModel.from_xml_path("/home/toby/Desktop/dofbot-grasp/sim/dofbot_se.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W * 2, H))

    jadr = {
        j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        for j in ("joint1", "joint2", "joint3", "joint4", "joint5")
    }
    q = np.array([0.0, 0.55, 0.9, 1.0, 1.0])
    for j, v in zip(("joint1", "joint2", "joint3", "joint4", "joint5"), q, strict=True):
        data.qpos[jadr[j]] = v
    data.qpos[-7:-4] = [SEED[0], SEED[1], S.CUBE_Z]
    S.servos(model, data, np.array([*q, S.GRIP_OPEN]), max_steps=1500)

    def shoot(tag):
        a = label(make_scene(model, data, renderer, "wrist_cam"), f"wrist: {tag}")
        b = label(make_scene(model, data, renderer, "overview"), f"overview: {tag}")
        frame = np.hstack([a, b])
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    shoot("start")
    det0 = S.detect_cube(make_scene(model, data, renderer, "wrist_cam"))
    if det0 is None:
        print("cube not visible at start")
        return
    tgt_u, tgt_v = S.tool_axis_pixel(model, data)
    J = S.identify_jacobian(model, data, renderer, q, probe_deg=4.0)

    for it in range(30):
        det = S.detect_cube(make_scene(model, data, renderer, "wrist_cam"))
        if det is None:
            shoot("cube lost")
            break
        e = np.array([det["centre"][0] - tgt_u, det["centre"][1] - tgt_v])
        shoot(f"CENTRE it{it} err=({e[0]:+.0f},{e[1]:+.0f})")
        if abs(e[0]) < S.DEADBAND_PX and abs(e[1]) < S.DEADBAND_PX:
            break
        dq = np.clip(-S.LAMBDA * np.linalg.pinv(J) @ e, -S.MAX_STEP_DEG, S.MAX_STEP_DEG)
        q[:3] += np.radians(dq)
        S.servos(model, data, np.array([*q, S.GRIP_OPEN]))

    for step in range(40):
        tip, _ = S._tip_and_axis(model, data)
        shoot(f"DESCEND step{step} tip_z={tip[2]*1000:.0f}mm")
        if tip[2] <= S.TIP_STOP_Z:
            break
        for k in range(3):
            q[k] = float(data.qpos[k])
        S.descend_step(model, data, q, dz=-0.008)
        S.servos(model, data, np.array([*q, S.GRIP_OPEN]))

    shoot("CLOSE")
    S.servos(model, data, np.array([*q, S.GRIP_CLOSE]))
    for k in range(6):
        shoot(f"CLOSE hold {k}")
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    z0 = float(data.xpos[cube_body][2])
    q[2] += np.radians(25.0)
    for k in range(10):
        S.servos(model, data, np.array([*q, S.GRIP_CLOSE]), max_steps=120)
        shoot(f"LIFT {k}")
    z1 = float(data.xpos[cube_body][2])
    writer.release()
    print(f"wrote {OUT}   cube z: {z0*1000:.1f} -> {z1*1000:.1f} mm  (lift {(z1-z0)*1000:+.1f} mm)")


if __name__ == "__main__":
    main()
