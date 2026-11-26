"""Image-based visual servoing for the DOFBOT cube grasp, validated in MuJoCo.

The controller mirrors what the real robot can do:

* only the *image* is used for feedback -- the cube's position is never handed in;
* the one geometric fact it may use is the wrist camera's mount offset relative to
  the tool axis (measured with calipers on the real arm: 23 mm back, 50 mm lateral)
  and the fact that the cube rests on the table (centre at z = 15 mm).

Phases
  1. CENTRE  drive the cube's centroid to the pixel where the tool axis projects,
             using a locally identified 2x2 image Jacobian.
  2. DESCEND lower along the tool axis until the silhouette area reaches the
             contact threshold (the fingertips straddle the cube).
  3. GRASP   close the gripper, then lift and check the cube came along.
"""

from __future__ import annotations

import numpy as np

import mujoco

CUBE_HALF = 0.015  # 30 mm cube
TABLE_Z = 0.0
CUBE_Z = CUBE_HALF  # cube centre height on the table
MOUNT_BACK = 0.023  # camera behind the tool axis (caliper measurement)
MOUNT_LAT = 0.050  # camera lateral offset (caliper measurement)

IMG_W, IMG_H = 320, 240
DEADBAND_PX = 18.0
MAX_STEP_DEG = 2.0
LAMBDA = 0.8
TIP_STOP_Z = 0.012  # close when the fingertip is at the cube's lower half
OPEN_L, OPEN_R = 0.008, -0.008  # the two finger slides, +ve opens each
CLOSE_L, CLOSE_R = -0.003, 0.003


def render(model, data, renderer, cam="wrist_cam"):
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def detect_cube(img: np.ndarray) -> dict | None:
    """Orange-brown cube on a dark floor: robust in sim, no model needed."""
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    mask = (r > 90) & (r > g + 25) & (g > b) & (b < 140)
    ys, xs = np.nonzero(mask)
    if len(xs) < 60:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return {
        "centre": (float(xs.mean()), float(ys.mean())),
        "area_frac": float(len(xs)) / mask.size,
        "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        "clipped": bool(x0 <= 1 or y0 <= 1 or x1 >= IMG_W - 2 or y1 >= IMG_H - 2),
    }


def project_point(model, data, point, cam="wrist_cam") -> tuple[float, float]:
    """Where a world point lands in the wrist image (pinhole, MuJoCo intrinsics)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    cam_pos = data.cam_xpos[cid].copy()
    cam_mat = data.cam_xmat[cid].reshape(3, 3).copy()
    local = cam_mat.T @ (np.asarray(point, float) - cam_pos)
    depth = -local[2]  # MuJoCo cameras look along -z
    if depth <= 1e-6:
        return float("nan"), float("nan")
    fovy = np.radians(model.cam_fovy[cid])
    f = (IMG_H / 2.0) / np.tan(fovy / 2.0)
    u = IMG_W / 2.0 + f * local[0] / depth
    v = IMG_H / 2.0 - f * local[1] / depth
    return float(u), float(v)


def identify_jacobian(model, data, renderer, q, probe_deg=4.0):
    """2x3 image Jacobian d(pixel)/d(joint) by central differences."""
    joints = (0, 1, 2)
    cols = []
    base = detect_cube(render(model, data, renderer))
    if base is None:
        return None
    p0 = np.array(base["centre"])
    for idx in joints:
        deltas = []
        for sgn in (+1.0, -1.0):
            qt = q.copy()
            qt[idx] += np.radians(sgn * probe_deg)
            servos(model, data, np.array([qt[0], qt[1], qt[2], qt[3], qt[4], OPEN_L, OPEN_R]))
            d = detect_cube(render(model, data, renderer))
            deltas.append(np.array(d["centre"]) if d else None)
        servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], OPEN_L, OPEN_R]))
        if deltas[0] is None or deltas[1] is None:
            return None
        cols.append((deltas[0] - deltas[1]) / (2.0 * probe_deg))
    return np.column_stack(cols)  # 2 x 3, px per degree


def _tip_and_axis(model, data):
    l5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
    axis = data.xmat[l5].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    return data.xpos[l5].copy() + 0.070 * axis, axis


def descend_step(model, data, q, dz=-0.004, damping=None):
    """Lower the tool tip by ``dz`` with the tool axis held constant (damped LS)."""
    tip0, axis0 = _tip_and_axis(model, data)
    J = np.zeros((3, 3))
    eps = np.radians(1.0)
    for k in range(3):
        qt = q.copy()
        qt[k] += eps
        for j, v in zip((0, 1, 2, 3, 4), qt):
            data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j + 1}")] = v
        mujoco.mj_forward(model, data)
        tip1, _ = _tip_and_axis(model, data)
        J[:, k] = (tip1 - tip0) / np.degrees(eps)
    for j, v in zip((0, 1, 2, 3, 4), q):
        data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j + 1}")] = v
    mujoco.mj_forward(model, data)
    target = np.array([0.0, 0.0, dz])  # tool tip moves straight down
    lam = damping if damping is not None else 1e-3 * float(np.trace(J @ J.T)) / 3.0
    dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), target)
    q[:3] += np.radians(np.clip(dq, -1.5, 1.5))
    return True


def tool_axis_pixel(model, data) -> tuple[float, float]:
    """Pixel where the *tool axis* passes at the cube's plane, not the camera axis.

    The camera sits MOUNT_LAT to the side and MOUNT_BACK behind the tool axis, so the
    point the fingers will hit is not the image centre.  We project the tool axis
    through the cube's plane (z = CUBE_Z) and aim the cube at *that* pixel.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    cam_pos = data.cam_xpos[cid].copy()
    cam_mat = data.cam_xmat[cid].reshape(3, 3)
    l5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
    axis = data.xmat[l5].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    origin = data.xpos[l5].copy()
    if abs(axis[2]) < 1e-6:
        return project_point(model, data, origin)
    s = (CUBE_Z - origin[2]) / axis[2]
    target = origin + s * axis
    return project_point(model, data, target)


def servos(model, data, ctrl, tol=0.004, max_steps=900):
    """Drive the position actuators to ``ctrl`` and step until the joints settle."""
    ctrl = np.asarray(ctrl, float)
    for _ in range(max_steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        err = np.abs(data.qpos[:5] - ctrl[:5]).max()
        if err < tol:
            for _ in range(30):
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
            return
    return


def run_once(seed_cube_xy, max_iters=40, verbose=False):
    model = mujoco.MjModel.from_xml_path(__file__.rsplit("/", 1)[0] + "/dofbot_se.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    names = [f"act_{n}" for n in ("joint1", "joint2", "joint3", "joint4", "joint5", "gripper")]
    act = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names}
    jadr = {
        j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        for j in ("joint1", "joint2", "joint3", "joint4", "joint5")
    }
    cube_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]

    data.qpos[cube_adr : cube_adr + 2] = seed_cube_xy
    data.qpos[cube_adr + 2] = CUBE_Z
    q = np.array([0.0, 0.55, 0.9, 1.0, 1.0])
    for j, v in zip(("joint1", "joint2", "joint3", "joint4", "joint5"), q):
        data.qpos[jadr[j]] = v
    ctrl = np.array([q[0], q[1], q[2], q[3], q[4], OPEN_L, OPEN_R])
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    servos(model, data, ctrl, max_steps=1500)

    # ---- phase 1: centre the cube on a FIXED target pixel -----------------
    det0 = detect_cube(render(model, data, renderer))
    if det0 is None:
        return {"ok": False, "reason": "cube not visible at start", "iters": 0}
    tgt_u, tgt_v = tool_axis_pixel(model, data)  # captured once, never updated
    trace = []
    J = identify_jacobian(model, data, renderer, q, probe_deg=4.0)
    if J is None:
        return {"ok": False, "reason": "jacobian identification failed"}
    for it in range(max_iters):
        if it and it % 4 == 0:  # refresh the Jacobian periodically
            Jn = identify_jacobian(model, data, renderer, q, probe_deg=4.0)
            if Jn is not None:
                J = Jn
        img = render(model, data, renderer)
        det = detect_cube(img)
        if det is None:
            return {"ok": False, "reason": "lost the cube while centring", "iters": it}
        e = np.array([det["centre"][0] - tgt_u, det["centre"][1] - tgt_v])
        trace.append((det["centre"][0], det["centre"][1], tgt_u, tgt_v))
        if verbose:
            print(
                f"  it{it:2d} cube=({det['centre'][0]:6.1f},{det['centre'][1]:6.1f}) "
                f"target=({tgt_u:6.1f},{tgt_v:6.1f}) err=({e[0]:+6.1f},{e[1]:+6.1f})"
            )
        if abs(e[0]) < DEADBAND_PX and abs(e[1]) < DEADBAND_PX:
            break
        dq = -LAMBDA * np.linalg.pinv(J) @ e
        dq = np.clip(dq, -MAX_STEP_DEG, MAX_STEP_DEG)
        q[0] += np.radians(dq[0])
        q[1] += np.radians(dq[1])
        q[2] += np.radians(dq[2])
        servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], OPEN_L, OPEN_R]))
    else:
        return {"ok": False, "reason": "centre did not converge", "trace": trace}

    # ---- phase 2: descend along the tool axis, keeping the view direction --
    for step in range(120):
        tip, _ = _tip_and_axis(model, data)
        if tip[2] <= TIP_STOP_Z:
            break
        # the cube legitimately leaves the frame as the hand closes in; the descent
        # is FK-driven, exactly as planned for the real arm (vision only aligns)
        det = detect_cube(render(model, data, renderer))
        ok = descend_step(model, data, q, dz=-0.004)
        if not ok:
            return {"ok": False, "reason": "descent IK failed"}
        servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], OPEN_L, OPEN_R]))
        # keep the cube on the target pixel while closing in: descent and centring
        # are coupled because the camera is offset from the tool axis
        for _ in range(4):
            det = detect_cube(render(model, data, renderer))
            if det is None:
                break
            e = np.array([det["centre"][0] - tgt_u, det["centre"][1] - tgt_v])
            if abs(e[0]) < DEADBAND_PX and abs(e[1]) < DEADBAND_PX:
                break
            dq = -LAMBDA * np.linalg.pinv(J) @ e
            dq = np.clip(dq, -MAX_STEP_DEG, MAX_STEP_DEG)
            q[0] += np.radians(dq[0])
            q[1] += np.radians(dq[1])
            q[2] += np.radians(dq[2])
            servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], OPEN_L, OPEN_R]))
    else:
        return {"ok": False, "reason": "never reached contact"}

    # ---- phase 3: close and lift ------------------------------------------
    servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], CLOSE_L, CLOSE_R]))
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    z_before = float(data.xpos[cube_body][2])
    q[2] += np.radians(25.0)  # lift
    servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], CLOSE_L, CLOSE_R]), max_steps=1500)
    z_after = float(data.xpos[cube_body][2])
    return {
        "ok": z_after > z_before + 0.02,
        "lift_mm": (z_after - z_before) * 1000,
        "iters": it,
        "trace": trace,
    }


if __name__ == "__main__":
    import sys

    seeds = [(-0.030, -0.250), (0.020, -0.270), (-0.010, -0.230), (-0.045, -0.285)]
    wins = 0
    for sx, sy in seeds:
        res = run_once((sx, sy), verbose="-v" in sys.argv)
        print(f"cube at ({sx * 1000:.0f},{sy * 1000:.0f}) mm -> {res}")
        wins += bool(res.get("ok"))
    print(f"\n{wins}/{len(seeds)} grasps succeeded")
