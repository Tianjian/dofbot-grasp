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
SEED = (0.020, 0.140)   # the +y region the real arm works in
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
    # start from the parked zero pose so the opening frames are readable
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    for j, v in zip(("joint1", "joint2", "joint3", "joint4", "joint5"), q, strict=True):
        data.qpos[jadr[j]] = v
    data.qpos[-7:-4] = [SEED[0], SEED[1], S.CUBE_Z]
    S.servos(model, data, np.array([*q, S.GRIP_OPEN]), max_steps=1500)

    tgt_px = [None]          # target pixel, updated by the caller each stage

    def shoot(tag):
        raw = make_scene(model, data, renderer, "wrist_cam")
        det = S.detect_cube(raw)
        vis = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        if tgt_px[0] is not None:
            cv2.drawMarker(vis, (int(tgt_px[0][0]), int(tgt_px[0][1])), (0, 255, 0),
                           cv2.MARKER_CROSS, 18, 2)
        if det is not None:
            cv2.circle(vis, (int(det["centre"][0]), int(det["centre"][1])), 7, (0, 0, 255), 2)
        a = label(vis, f"wrist: {tag}  green=target red=cube")
        b = label(cv2.cvtColor(make_scene(model, data, renderer, "overview"),
                               cv2.COLOR_RGB2BGR), f"overview: {tag}")
        writer.write(np.hstack([a, b]))

    shoot("PARK (zero pose)")
    q = np.array([-0.1745, 0.1396, -1.5359, -1.3265, 1.0])
    S.servos(model, data, np.array([*q, S.GRIP_OPEN]))
    shoot("READY (task pose)")
    det0 = S.detect_cube(make_scene(model, data, renderer, "wrist_cam"))
    if det0 is None:
        print("cube not visible at start")
        return
    tgt_u, tgt_v = S.tool_axis_pixel(model, data)
    tgt_px[0] = (tgt_u, tgt_v)
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

    j5 = S.align_jaw_axis(model, data, q)
    S.servos(model, data, np.array([*q, S.GRIP_OPEN]))
    shoot(f"ALIGN j5={__import__('numpy').degrees(j5):.0f}deg")

    cub = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_xyz = data.xpos[cub].copy()
    l5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
    jaw = data.xmat[l5].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
    jaw[2] = 0.0
    jaw /= np.linalg.norm(jaw)
    hover = cube_xyz - jaw * 0.035
    hover[2] = 0.050
    for k in range(40):
        tip, _ = S._tip_and_axis(model, data)
        shoot(f"HOVER k{k} tip=({tip[0]*1000:.0f},{tip[1]*1000:.0f},{tip[2]*1000:.0f})")
        if np.linalg.norm(hover - tip) < 0.007:
            break
        for j in range(3):
            q[j] = float(data.qpos[j])
        S.move_tip(model, data, q, (hover - tip) * 0.5)
        S.servos(model, data, np.array([*q, S.GRIP_OPEN]))
    enter = hover + jaw * 0.035
    enter[2] = 0.012
    for k in range(40):
        tip, _ = S._tip_and_axis(model, data)
        shoot(f"ENTER k{k} tip=({tip[0]*1000:.0f},{tip[1]*1000:.0f},{tip[2]*1000:.0f})")
        if np.linalg.norm(enter - tip) < 0.007:
            break
        for j in range(3):
            q[j] = float(data.qpos[j])
        S.move_tip(model, data, q, (enter - tip) * 0.5)
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
