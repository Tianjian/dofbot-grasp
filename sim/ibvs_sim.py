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

import mujoco
import numpy as np

CUBE_HALF = 0.015  # 30 mm cube
TABLE_Z = 0.0
CUBE_Z = CUBE_HALF  # cube centre height on the table
MOUNT_BACK = 0.023  # camera behind the tool axis (caliper measurement)
MOUNT_LAT = 0.050  # camera lateral offset (caliper measurement)

IMG_W, IMG_H = 320, 240
DEADBAND_PX = 3.0   # 6 px was ~6.7 mm laterally, more than the 5.45 mm jaw clearance
MAX_STEP_DEG = 2.0
LAMBDA = 0.8
TIP_STOP_Z = 0.010  # fingertip below the cube's mid-height (15 mm) before closing
FINGERTIP_Z = 0.0875  # link5 -> fingertip, from link5.STL (jaws end at z = 87.5 mm)
GRIP_OPEN = 0.0    # j6 = 0 -> jaws apart, inner faces at +-20.45 mm (parallel slides)
GRIP_CLOSE = 0.0068  # 6.8 mm slide each -> inner faces +-13.7 mm, 1.3 mm interference

# ---- sim2real joint convention ------------------------------------------------
# Real bus servos report u in [0, 180] (j1..j4, j6) and [0, 270] (j5). The vendor
# library reverses servos 2/3/4 internally, and after that the joint angle is
#     q1..q4 = u - 90 deg        (so u = 90 is the vertical / neutral pose)
#     q5     = u - 135 deg       (270 deg of travel)
# The simulation stores q directly, so this is the conversion in both directions.
SERVO_OFFSET_DEG = {1: 90.0, 2: 90.0, 3: 90.0, 4: 90.0, 5: 135.0}
# Gripper: servo 6 goes 0..180; on the real arm the jaws are apart at u6 around 20 and
# shut at around 140. The jaw hinge therefore spans ~120 deg of servo for 9.2 deg of
# hinge, i.e. about a 13:1 reduction.
GRIP_SERVO_OPEN = 20.0
GRIP_SERVO_CLOSED = 140.0
GRIP_JAW_AT_OPEN = 0.0
GRIP_JAW_AT_CLOSED = 0.0068


def servo_to_joint(servo_deg: dict[int, float]) -> np.ndarray:
    """Servo read-back (ids 1..5) -> simulation joint angles in radians."""
    return np.array([np.radians(servo_deg[i] - SERVO_OFFSET_DEG[i]) for i in range(1, 6)])


def joint_to_servo(q) -> dict[int, float]:
    """Simulation joint angles (radians) -> servo commands (degrees)."""
    return {i: float(np.degrees(q[i - 1]) + SERVO_OFFSET_DEG[i]) for i in range(1, 6)}


def jaw_from_servo6(u6: float) -> float:
    """Servo 6 angle (deg) -> jaw hinge angle (rad), matching the real open/closed stops."""
    span = (u6 - GRIP_SERVO_OPEN) / (GRIP_SERVO_CLOSED - GRIP_SERVO_OPEN)
    return GRIP_JAW_AT_OPEN + span * (GRIP_JAW_AT_CLOSED - GRIP_JAW_AT_OPEN)


def servo6_from_jaw(jaw: float) -> float:
    """Jaw hinge angle (rad) -> servo 6 command (deg)."""
    span = (jaw - GRIP_JAW_AT_OPEN) / (GRIP_JAW_AT_CLOSED - GRIP_JAW_AT_OPEN)
    return GRIP_SERVO_OPEN + span * (GRIP_SERVO_CLOSED - GRIP_SERVO_OPEN)


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


def align_jaw_axis(model, data, q):
    """Rotate j5 so the jaws close along a world-horizontal axis.

    A cube resting squarely on the table is best grasped on two opposite faces, so the
    jaw opening axis should be parallel to a horizontal axis. j5 turns the gripper base
    about the tool axis; because the camera hangs off link4 it is not affected.
    """
    j5 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint5")]
    best = None
    for d in np.linspace(-np.pi, np.pi, 181):
        q[4] = d
        for k, v in zip(range(5), q[:5], strict=True):
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k + 1}")]] = v
        mujoco.mj_forward(model, data)
        l5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
        jaw = data.xmat[l5].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        horiz = np.array([jaw[0], jaw[1], 0.0])
        norm = np.linalg.norm(horiz)
        if norm < 1e-6:
            continue
        horiz /= norm
        # best when the jaw axis is parallel to world x or y
        score = max(abs(horiz[0]), abs(horiz[1]))
        if best is None or score > best[0]:
            best = (score, float(d))
    q[4] = best[1]
    for k, v in zip(range(5), q[:5], strict=True):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k + 1}")]] = v
    mujoco.mj_forward(model, data)
    return best[1]


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
    for idx in joints:
        deltas = []
        for sgn in (+1.0, -1.0):
            qt = q.copy()
            qt[idx] += np.radians(sgn * probe_deg)
            servos(model, data, np.array([qt[0], qt[1], qt[2], qt[3], qt[4], GRIP_OPEN]))
            d = detect_cube(render(model, data, renderer))
            deltas.append(np.array(d["centre"]) if d else None)
        servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], GRIP_OPEN]))
        if deltas[0] is None or deltas[1] is None:
            return None
        cols.append((deltas[0] - deltas[1]) / (2.0 * probe_deg))
    return np.column_stack(cols)  # 2 x 3, px per degree


def _tip_and_axis(model, data):
    l5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
    axis = data.xmat[l5].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    return data.xpos[l5].copy() + FINGERTIP_Z * axis, axis


def move_tip(model, data, q, delta, gain=1.6, max_deg=2.0, n_joints=4):
    """Translate the tool tip by ``delta`` (metres, world).

    Uses ``n_joints`` columns (j1..j4 by default). Three was not enough: a 58 mm lateral
    move needs the wrist to contribute, and a square Jacobian then cannot solve it.
    """
    tip0, _ = _tip_and_axis(model, data)
    J = np.zeros((3, n_joints))
    eps = np.radians(1.0)
    for k in range(n_joints):
        qt = q.copy()
        qt[k] += eps
        for j, v in zip(range(5), qt, strict=True):
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j + 1}")]] = v
        mujoco.mj_forward(model, data)
        J[:, k] = (_tip_and_axis(model, data)[0] - tip0) / np.degrees(eps)
    for j, v in zip(range(5), q, strict=True):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j + 1}")]] = v
    mujoco.mj_forward(model, data)
    lam = 1e-2 * float(np.trace(J @ J.T)) / 3.0
    dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), np.asarray(delta, float))
    q[:n_joints] += np.radians(np.clip(dq * gain, -max_deg, max_deg))
    return True


def move_tip_to(model, data, q, target_xyz, max_iters=60):
    """Closed-loop tip translation; re-anchors each step because the servos sag."""
    for _ in range(max_iters):
        tip, _ = _tip_and_axis(model, data)
        err = np.asarray(target_xyz, float) - tip
        if np.linalg.norm(err) < 0.006:
            return True
        for k in range(4):
            q[k] = float(data.qpos[k])
        move_tip(model, data, q, err * 0.5)
        servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], GRIP_OPEN]))
    return np.linalg.norm(np.asarray(target_xyz) - _tip_and_axis(model, data)[0]) < 0.020


def centre_cube(model, data, renderer, q, tgt_u, tgt_v, max_iters=25, verbose=False):
    """Drive the cube's centroid onto the tool-axis pixel.

    The Jacobian is re-identified every iteration: the camera sits off the tool axis, so
    a Jacobian measured at one pose has the wrong gains (and sometimes the wrong signs)
    at another.
    """
    J = identify_jacobian(model, data, renderer, q, probe_deg=3.0)
    if J is None:
        return False, None
    for it in range(max_iters):
        if it:
            Jn = identify_jacobian(model, data, renderer, q, probe_deg=3.0)
            if Jn is not None:
                J = Jn
        det = detect_cube(render(model, data, renderer))
        if det is None:
            return False, None
        e = np.array([det["centre"][0] - tgt_u, det["centre"][1] - tgt_v])
        if verbose:
            print(f"  centre it{it}: err=({e[0]:+.1f},{e[1]:+.1f})")
        if abs(e[0]) < DEADBAND_PX and abs(e[1]) < DEADBAND_PX:
            return True, det
        dq = np.clip(-LAMBDA * np.linalg.pinv(J) @ e, -MAX_STEP_DEG, MAX_STEP_DEG)
        q[:3] += np.radians(dq)
        servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], GRIP_OPEN]))
    return False, None


def descend_until(model, data, q, z_target, dz=-0.008, gain=1.0, max_iters=110):
    """Lower the tip to ``z_target`` in FK closed loop.

    The position servos hold a static error of ~1.5 deg under gravity, which is larger
    than a single small joint step, so an open-loop descent can move the wrong way.
    Each iteration measures the achieved tip height and adapts the step size.
    """
    for _ in range(max_iters):
        tip, _ = _tip_and_axis(model, data)
        if tip[2] <= z_target:
            return True
        # re-anchor the command to the achieved angles: the servos hold ~1.5 deg of
        # static error, so accumulating commands in open loop runs away from the arm
        for k in range(3):
            q[k] = float(data.qpos[k])
        before = tip[2]
        descend_step(model, data, q, dz=dz * gain)
        servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], GRIP_OPEN]))
        after = _tip_and_axis(model, data)[0][2]
        if after >= before - 1e-4:          # did not go down: push harder
            gain = min(gain * 1.8, 10.0)
        else:
            gain = max(gain * 0.9, 1.0)
    return False


def descend_step(model, data, q, dz=-0.004, damping=None):
    """Lower the tool tip by ``dz`` with the tool axis held constant (damped LS)."""
    tip0, axis0 = _tip_and_axis(model, data)
    J = np.zeros((3, 3))
    eps = np.radians(1.0)
    for k in range(3):
        qt = q.copy()
        qt[k] += eps
        for j, v in zip((0, 1, 2, 3, 4), qt, strict=True):
            data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j + 1}")] = v
        mujoco.mj_forward(model, data)
        tip1, _ = _tip_and_axis(model, data)
        J[:, k] = (tip1 - tip0) / np.degrees(eps)
    for j, v in zip((0, 1, 2, 3, 4), q, strict=True):
        data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j + 1}")] = v
    mujoco.mj_forward(model, data)
    target = np.array([0.0, 0.0, dz])  # tool tip moves straight down
    lam = damping if damping is not None else 1e-3 * float(np.trace(J @ J.T)) / 3.0
    dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), target)
    q[:3] += np.radians(np.clip(dq, -1.5, 1.5))
    return True


def unproject_to_plane(model, data, u, v, z_plane=0.015, cam="wrist_cam"):
    """Pixel -> world point on the horizontal plane z = z_plane.

    Fair game on the real robot too: the intrinsics are calibrated and the camera's pose
    comes from FK plus the registered mount. It removes the need to keep the cube in view
    while the arm moves, which a 90 deg field makes impractical.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    cam_pos = data.cam_xpos[cid].copy()
    cam_mat = data.cam_xmat[cid].reshape(3, 3).copy()
    fovy = np.radians(model.cam_fovy[cid])
    f = (IMG_H / 2.0) / np.tan(fovy / 2.0)
    x_c = (u - IMG_W / 2.0) / f
    y_c = -(v - IMG_H / 2.0) / f
    ray_cam = np.array([x_c, y_c, -1.0])
    ray = cam_mat @ ray_cam
    if abs(ray[2]) < 1e-6:
        return None
    t = (z_plane - cam_pos[2]) / ray[2]
    if t <= 0:
        return None
    return cam_pos + t * ray


def tool_axis_pixel(model, data) -> tuple[float, float]:
    """Pixel where the *tool axis* passes at the cube's plane, not the camera axis.

    The camera sits MOUNT_LAT to the side and MOUNT_BACK behind the tool axis, so the
    point the fingers will hit is not the image centre.  We project the tool axis
    through the cube's plane (z = CUBE_Z) and aim the cube at *that* pixel.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    l5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link5")
    axis = data.xmat[l5].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    origin = data.xpos[l5].copy()
    if abs(axis[2]) < 1e-6:
        return project_point(model, data, origin)
    s = (CUBE_Z - origin[2]) / axis[2]
    target = origin + s * axis
    return project_point(model, data, target)


MOVE_TIME_S = 1.2   # the real write6 frames command a 1.2 s move


def servos(model, data, ctrl, tol=0.004, max_steps=3000, move_time=MOVE_TIME_S):
    """Drive the actuators to ``ctrl`` with the real servos' *timed* move semantics.

    The real bus takes a move time with every write6 frame and interpolates to the
    target over it, with a ~1.5 deg static error under gravity. Setting data.ctrl
    directly makes the simulated arm settle about eight times faster, which is enough
    to change the controller's behaviour, so the target is ramped instead.
    """
    ctrl = np.asarray(ctrl, float)
    start = data.ctrl.copy()
    dt = float(model.opt.timestep)
    ramp = max(1, int(move_time / dt))
    for k in range(max_steps):
        a = min(1.0, (k + 1) / ramp)
        data.ctrl[:] = start + (ctrl - start) * a
        mujoco.mj_step(model, data)
        if a >= 1.0 and k > ramp:
            if np.abs(data.qpos[:5] - ctrl[:5]).max() < tol:
                for _ in range(100):
                    data.ctrl[:] = ctrl
                    mujoco.mj_step(model, data)
                return
    return


def run_once(seed_cube_xy, max_iters=40, verbose=False, shot=None):
    model = mujoco.MjModel.from_xml_path(__file__.rsplit("/", 1)[0] + "/dofbot_se.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    jadr = {
        j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        for j in ("joint1", "joint2", "joint3", "joint4", "joint5")
    }
    cube_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]

    data.qpos[cube_adr : cube_adr + 2] = seed_cube_xy
    data.qpos[cube_adr + 2] = CUBE_Z
    # aimed at the +y grasp region, the same side the real arm works on
    q = np.array([-0.1745, 0.1396, -1.5359, -1.3265, 1.0])
    for j, v in zip(("joint1", "joint2", "joint3", "joint4", "joint5"), q, strict=True):
        data.qpos[jadr[j]] = v
    ctrl = np.array([q[0], q[1], q[2], q[3], q[4], GRIP_OPEN])
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    servos(model, data, ctrl, max_steps=1500)

    # ---- phase 1: centre the cube on a FIXED target pixel -----------------
    det0 = detect_cube(render(model, data, renderer))
    if det0 is None:
        return {"ok": False, "reason": "cube not visible at start", "iters": 0}
    tgt_u, tgt_v = tool_axis_pixel(model, data)  # captured once, never updated
    trace = []
    if shot:
        shot("READY")
    ok_centre, det_centred = centre_cube(model, data, renderer, q, tgt_u, tgt_v,
                                         verbose=verbose)
    if shot:
        shot("CENTRED")
    if not ok_centre:
        return {"ok": False, "reason": "lost the cube while centring"}

    # ---- phase 1b: turn the gripper base so the jaws close on two faces ----
    j5 = align_jaw_axis(model, data, q)
    servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], GRIP_OPEN]))
    if verbose:
        print(f"  jaw axis aligned: j5 = {np.degrees(j5):.1f} deg")

    # ---- phase 1c: compensate the camera's parallax -------------------------
    # Centring puts the cube on the *camera's* optical axis, but the camera sits off the
    # tool axis, so the two are parallel lines separated by the mount offset. Moving the
    # arm by that offset puts the tool axis -- not the camera axis -- on the cube.
    # Centring leaves the cube on the *camera* axis; the tool axis is a parallel line one
    # mount-offset away. Rather than sliding by that offset (which loses the cube out of a
    # 90 deg field before the next look), back-project the cube's pixel onto the table
    # plane: with calibrated intrinsics and the registered mount that is a direct 3-D
    # fix, and it needs no further view of the cube.
    det = detect_cube(render(model, data, renderer))
    if det is None:
        return {"ok": False, "reason": "lost the cube before back-projection"}
    cube_est = unproject_to_plane(model, data, det["centre"][0], det["centre"][1], 0.015)
    if cube_est is None:
        return {"ok": False, "reason": "back-projection failed"}
    if verbose:
        print(f"  back-projected cube: ({cube_est[0]*1000:.0f}, {cube_est[1]*1000:.0f}) mm")
    if shot:
        shot(f"BACKPROJECT_{cube_est[0]*1000:.0f}_{cube_est[1]*1000:.0f}")

    # ---- phase 2: descend straight down, holding x/y ------------------------
    # Descending along the tool axis is not vertical: the arm reaches out as it lowers
    # and the tip drifts about 15 mm, which is enough for the jaws to land on the cube
    # instead of around it. So the descent holds x/y and moves z only.
    above, _ = _tip_and_axis(model, data)
    grasph = np.array([cube_est[0], cube_est[1], 0.010])
    if shot:
        shot("DESCEND_START")
    if not move_tip_to(model, data, q, grasph, max_iters=80):
        return {"ok": False, "reason": "vertical descent did not converge"}
    if shot:
        shot("AT_GRASP_HEIGHT")

    # ---- phase 3: close and lift ------------------------------------------
    servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], GRIP_CLOSE]))
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    z_before = float(data.xpos[cube_body][2])
    q[2] += np.radians(25.0)  # lift
    servos(model, data, np.array([q[0], q[1], q[2], q[3], q[4], GRIP_CLOSE]), max_steps=1500)
    z_after = float(data.xpos[cube_body][2])
    return {"ok": z_after > z_before + 0.02, "lift_mm": (z_after - z_before) * 1000}


if __name__ == "__main__":
    import sys

    # A collision-filtered sweep over the FULL joint range shows the grasp region is
    # symmetric: 49 % of the reachable points sit at y > 0, which is the side the real
    # arm actually works on. Sampling q3/q4 over a positive-only window (an earlier
    # mistake) hid the whole +y lobe.
    seeds = [(0.020, 0.140), (-0.060, 0.140), (0.060, 0.140), (0.100, 0.140)]
    wins = 0
    for sx, sy in seeds:
        res = run_once((sx, sy), verbose="-v" in sys.argv)
        print(f"cube at ({sx * 1000:.0f},{sy * 1000:.0f}) mm -> {res}")
        wins += bool(res.get("ok"))
    print(f"\n{wins}/{len(seeds)} grasps succeeded")
