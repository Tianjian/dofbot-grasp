"""The grasp controller: one sequence, two backends.

Phases, in order:

1. **centre** the cube on the tool-axis pixel by image-based servoing. The image Jacobian
   is re-identified every iteration: the wrist camera sits off the tool axis, so gains
   measured at another pose are wrong and sometimes have the wrong sign.
2. **align** the jaws by turning j5 until the opening axis is parallel to a world axis, so
   the fingers meet two opposite faces instead of corners.
3. **fix** the cube's 3-D position by back-projecting its pixel onto the table plane,
   using the calibrated intrinsics plus the registered mount.
4. **descend** vertically to the grasp height, holding x/y, then **close** and **lift**.

Only FK is used for geometry, so simulation and hardware follow the same arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from . import kinematics as K
from .params import GRASP, JOINTS, MOUNT


# ---------------------------------------------------------------- geometry
def joint_angles(pose: dict[int, float]) -> np.ndarray:
    return np.radians(np.array([pose[i] - JOINTS.offsets_deg[i] for i in range(1, 6)]))


def tip_and_axis(pose: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """Tool-tip position and tool axis in the base frame, from servo angles."""
    T = K.forward(joint_angles(pose))
    axis = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    tip = T[:3, 3] + (JOINTS.fingertip_below_tcp_m + 0.0) * axis
    return tip, axis


def jaw_axis(pose: dict[int, float]) -> np.ndarray:
    """The direction the jaws open along: link5's y, from the mesh's plate separation."""
    T = K.forward(joint_angles(pose))
    return T[:3, :3] @ np.array([0.0, 1.0, 0.0])


def camera_pose(
    pose: dict[int, float], roll_deg: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Wrist camera centre and rotation (camera -> base), from the registered mount."""
    T = K.forward(joint_angles(pose))
    p5 = T[:3, 3]
    R5 = T[:3, :3]
    roll = MOUNT.roll_deg if roll_deg is None else roll_deg
    return p5 + R5 @ cam_in_link5(), R5 @ cam_rotation_in_link5(roll)


LINK5_IN_LINK4 = np.array([-0.07385, -0.001, 0.0])
ROT_Y_MINUS_90 = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def cam_in_link5() -> np.ndarray:
    p_link4 = np.array(MOUNT.t_link4_cam_mm) / 1000.0
    return ROT_Y_MINUS_90.T @ (p_link4 - LINK5_IN_LINK4)


def cam_rotation_in_link5(roll_deg: float) -> np.ndarray:
    """Camera -> link5. OpenCV puts the optical axis on +z, the tool axis is +z too."""
    import cv2

    z = np.array([0.0, 0.0, 1.0])
    x0 = np.array([1.0, 0.0, 0.0])
    base = np.column_stack([x0, np.cross(z, x0), z])
    return base @ cv2.Rodrigues(z * math.radians(roll_deg))[0]


def backproject_to_table(
    u: float, v: float, pose: dict[int, float], intr: dict, z_plane: float | None = None
) -> np.ndarray | None:
    """Pixel -> base-frame point on the table plane."""
    import cv2

    Kmtx = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    dist = np.array(intr["dist"], float).reshape(-1, 1)
    und = cv2.undistortPoints(np.array([[[u, v]]], np.float32), Kmtx, dist).reshape(2)
    C, R = camera_pose(pose)
    ray = R @ np.array([und[0], und[1], 1.0])
    if abs(ray[2]) < 1e-6:
        return None
    plane = GRASP.table_z_m if z_plane is None else z_plane
    t = (plane - C[2]) / ray[2]
    return None if t <= 0 else C + t * ray


# ---------------------------------------------------------------- motion
def _tip_jacobian(pose: dict[int, float], eps_deg: float = 1.0) -> np.ndarray:
    """d(tip) / d(servo degrees), 3 x 5."""
    base, _ = tip_and_axis(pose)
    J = np.zeros((3, 5))
    for k in range(5):
        probe = dict(pose)
        probe[k + 1] += eps_deg
        J[:, k] = (tip_and_axis(probe)[0] - base) / eps_deg
    return J


def move_tip(io, delta_m: np.ndarray, *, gain: float = 1.6, max_deg: float = 2.0) -> None:
    """Command a tip translation with a damped least-squares step.

    Joints sitting on a limit that the step would push further out are dropped from the
    solve. Without that active set the solver kept pushing a saturated elbow and the tip
    stalled 25 mm above the grasp height instead of reaching it.
    """
    pose = io.read_joints()
    J = _tip_jacobian(pose)
    delta = np.asarray(delta_m, float)
    free = np.ones(5, dtype=bool)
    for k in range(5):
        lo = -JOINTS.offsets_deg[k + 1]
        hi = JOINTS.servo_range_deg[k + 1] - JOINTS.offsets_deg[k + 1]
        qk = pose[k + 1] - JOINTS.offsets_deg[k + 1]
        if qk <= lo + 0.5:
            free[k] = J[2, k] * delta[2] <= 0.0
        elif qk >= hi - 0.5:
            free[k] = J[2, k] * delta[2] >= 0.0
    if not free.any():
        return
    Jf = J[:, free]
    lam = 1e-2 * float(np.trace(Jf @ Jf.T)) / 3.0
    dqf = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam * np.eye(3), delta)
    target = dict(pose)
    column = 0
    for k in range(5):
        if not free[k]:
            continue
        target[k + 1] = pose[k + 1] + float(np.clip(dqf[column] * gain, -max_deg, max_deg))
        column += 1
    io.command(target)


def move_tip_to(
    io, target_xyz, *, max_iters: int = 140, tol: float = 0.005, contact_abort: bool = True
) -> bool:
    """Walk the fingertip onto ``target_xyz`` with small damped steps.

    Stops as soon as the tip stops following the command. Continuing to push after the
    fingers touch something drove the cube clean through the table (a recorded lift of
    -20 mm), so contact ends the descent instead of being averaged away.
    """
    target = np.asarray(target_xyz, float)
    prev_z = None
    stalled = 0
    for _ in range(max_iters):
        tip, _ = tip_and_axis(io.read_joints())
        err = target - tip
        if np.linalg.norm(err) < tol:
            return True
        if contact_abort and prev_z is not None:
            stalled = stalled + 1 if abs(tip[2] - prev_z) < 0.0004 else 0
            if stalled >= 3:
                return True  # blocked: treat the contact as the grasp height
        prev_z = float(tip[2])
        # Descend with a pure height command. Feeding the lateral error as well let the
        # solver spend its joint budget correcting x/y, and it stalled with the elbow on
        # its limit; the centring stage is what keeps the tool axis on the cube.
        move_tip(io, np.array([0.0, 0.0, float(np.clip(err[2], -0.010, 0.010))]))
    tip, _ = tip_and_axis(io.read_joints())
    return bool(np.linalg.norm(target - tip) < 0.015)


# ---------------------------------------------------------------- servoing
def centre(
    io, detect: Callable, target_px, *, max_iters: int = 25, log: Callable | None = None
) -> bool:
    """Drive the cube's centroid onto ``target_px`` with a per-step image Jacobian."""
    pose = io.read_joints()
    det = detect(io.capture_wrist())
    if det is None:
        return False
    point = io.unproject_to_table(det["centre"][0], det["centre"][1], GRASP.table_z_m)
    if point is None:
        return False
    best_err = float("inf")
    stalled = 0
    for it in range(max_iters):
        pose = io.read_joints()
        J = _image_jacobian(io, pose, point)
        if J is None:
            return False
        det = detect(io.capture_wrist())
        if det is None:
            return False
        e = np.array([det["centre"][0] - target_px[0], det["centre"][1] - target_px[1]])
        if log:
            log(f"  centre it{it}: err=({e[0]:+.1f},{e[1]:+.1f})")
        norm = float(np.linalg.norm(e))
        if abs(e[0]) < GRASP.deadband_px and abs(e[1]) < GRASP.deadband_px:
            return True
        if norm < best_err - 0.5:
            best_err, stalled = norm, 0
        else:
            stalled += 1
            if stalled >= 6:
                # The pixel map has local minima: with j4 pinned, the gradient settles 23 px
                # short while a 1.8 px solution sits ten degrees away on j2 and j4.
                jumped = _escape_local_minimum(io, detect, target_px, pose)
                best_err, stalled = float("inf"), 0
                if not jumped:
                    return False
                continue
        dq = np.clip(
            -GRASP.gain * np.linalg.pinv(J) @ np.asarray(e), -GRASP.max_step_deg, GRASP.max_step_deg
        )
        pose = io.read_joints()
        target = dict(pose)
        for k in range(3):
            target[k + 1] = pose[k + 1] + float(dq[k])
        io.command(target)
    return False


def _solve_pixel_step(J: np.ndarray, err: np.ndarray, pose: dict[int, float]) -> np.ndarray:
    """Pixel error -> joint step, skipping joints already on a limit.

    Leaving a saturated joint in the solve wastes the whole step on it: with j4 pinned at
    -90 deg the y error sat at 23 px for fifty iterations while nothing moved.
    """
    dq = _weighted_step(J, err, len(GRASP.joint_weights))
    for _ in range(3):
        for k in range(J.shape[1]):
            lo = -JOINTS.offsets_deg[k + 1]
            hi = JOINTS.servo_range_deg[k + 1] - JOINTS.offsets_deg[k + 1]
            qk = pose[k + 1] - JOINTS.offsets_deg[k + 1]
            if (qk <= lo + 0.5 and dq[k] < 0) or (qk >= hi - 0.5 and dq[k] > 0):
                J = J.copy()
                J[:, k] = 0.0
                dq = _weighted_step(J, err, len(GRASP.joint_weights))
    caps = np.asarray(GRASP.max_step_per_joint[: len(dq)], dtype=float)
    return np.clip(dq, -caps, caps)


def _weighted_step(J: np.ndarray, err: np.ndarray, n: int) -> np.ndarray:
    """Least-squares pixel step with per-joint cost.

    Unweighted, the solver happily folds the wrist to its limit and leaves the shoulder
    barely rotated, which parks the arm in a configuration it cannot reach past. Costing
    the wrist more than the shoulder and elbow makes it extend instead.
    """
    W = np.diag(np.asarray(GRASP.joint_weights[:n], dtype=float))
    Winv = np.linalg.inv(W)
    JT = J.T @ Winv
    lam = 1e-2 * float(np.trace(J @ Winv @ J.T)) / 2.0
    dq = -GRASP.gain * JT @ np.linalg.solve(J @ Winv @ J.T + lam * np.eye(2), err)
    return dq


def _escape_local_minimum(io, detect: Callable, target_px, pose: dict[int, float]) -> bool:
    """Small joint-space search used when the pixel gradient stalls.

    Costs a handful of moves, which is acceptable for a recovery and much cheaper than
    failing the grasp. Returns True when a better configuration was found and applied.
    """
    current = detect(io.capture_wrist())
    if current is None:
        return False
    best = float(np.linalg.norm(np.array(current["centre"]) - np.array(target_px)))
    best_q = None
    for dj2 in (-10.0, 0.0, 10.0, 20.0):
        for dj4 in (-10.0, 0.0, 10.0, 20.0):
            if dj2 == 0.0 and dj4 == 0.0:
                continue
            probe = dict(pose)
            probe[2] += dj2
            probe[4] += dj4
            if not (0.0 < probe[2] < 180.0 and 0.0 < probe[4] < 180.0):
                continue
            # probing must not drive the fingers into the cube; the escape exists to fix
            # aiming, not to test how hard the table is
            if tip_and_axis(probe)[0][2] < GRASP.table_z_m + GRASP.cube_edge_m + 0.015:
                continue
            io.command(probe)
            det = detect(io.capture_wrist())
            if det is None:
                continue
            err = float(np.linalg.norm(np.array(det["centre"]) - np.array(target_px)))
            if err < best:
                best, best_q = err, dict(probe)
    if best_q is None:
        io.command(pose)
        return False
    io.command(best_q)
    return True


def _image_jacobian(io, pose: dict[int, float], point_m: np.ndarray) -> np.ndarray | None:
    """2 x 3 pixel-per-degree Jacobian of a *known* world point.

    Computed from the camera model at perturbed poses, so it costs no motion. Probing the
    real arm instead cost six moves per refresh and each restore was imperfect, which
    accumulated drift and made the loop oscillate rather than settle.
    """
    base = io.project(point_m, pose)
    if base is None:
        return None
    cols = []
    for k in range(4):  # j4 is needed: these cube positions are a long reach away
        probe = dict(pose)
        probe[k + 1] += GRASP.probe_deg
        moved = io.project(point_m, probe)
        if moved is None:
            return None
        cols.append((np.array(moved) - np.array(base)) / GRASP.probe_deg)
    return np.column_stack(cols)


def align_jaws(io, offset_deg: float = 0.0) -> float:
    """Turn j5 until the jaw axis is parallel to a world axis.

    ``offset_deg`` rotates the gripper base a further quarter turn, which makes the jaws
    close on the other pair of faces. The fingers are 21.3 mm wide and 47.9 mm long, so
    which pair they straddle changes the clearance on the way down.
    """
    pose = io.read_joints()
    best = None
    for deg in np.linspace(-135.0, 135.0, 181):
        probe = dict(pose)
        probe[5] = JOINTS.offsets_deg[5] + deg
        jaw = jaw_axis(probe)
        horiz = np.array([jaw[0], jaw[1], 0.0])
        norm = np.linalg.norm(horiz)
        if norm < 1e-6:
            continue
        horiz /= norm
        score = max(abs(horiz[0]), abs(horiz[1]))
        if best is None or score > best[0]:
            best = (score, float(deg))
    if best is None:
        return 0.0
    target = dict(pose)
    target[5] = JOINTS.offsets_deg[5] + best[1] + offset_deg
    io.command(target)
    return float(best[1] + offset_deg)


def descend_guarded(io, detect: Callable, target_xyz, say) -> bool:
    """Lower onto the cube, stopping the moment the fingertip stops following.

    The reliable contact signal is mechanical: the tip stops descending while the command
    keeps asking for it. Watching the cube's pixel instead gave both false positives
    (fingers entering the view shift the silhouette centroid) and false negatives (the
    descent continued and pressed the cube through the table). The camera is kept as a
    log, not as the stop condition.
    """
    ref = detect(io.capture_wrist())
    prev_z = None
    stalled = 0
    for _ in range(80):
        tip, _ = tip_and_axis(io.read_joints())
        err = np.asarray(target_xyz, float) - tip
        if np.linalg.norm(err) < 0.005:
            return True
        if prev_z is not None:
            if abs(tip[2] - prev_z) < 0.0004:
                stalled += 1
                if stalled >= 2:
                    say("fingertip stopped descending -- contact")
                    return True
            else:
                stalled = 0
        prev_z = float(tip[2])
        det = detect(io.capture_wrist())
        if det is not None and ref is not None:
            shifted = float(
                np.linalg.norm(np.array(det["centre"], float) - np.array(ref["centre"], float))
            )
            if shifted > GRASP.cube_shift_abort_px:
                say(f"note: cube centroid {shifted:.0f} px from the start of the descent")
        move_tip(io, np.array([0.0, 0.0, float(np.clip(err[2], -0.010, 0.010))]))
    tip, _ = tip_and_axis(io.read_joints())
    return bool(np.linalg.norm(np.asarray(target_xyz, float) - tip) < 0.015)


# ---------------------------------------------------------------- sequence
def grasp(
    io,
    detect: Callable,
    *,
    jaw_offset_deg: float = 90.0,
    log: Callable | None = None,
    on_stage: Callable | None = None,
) -> dict:
    """Run the whole sequence and report what happened."""
    say = log or (lambda _m: None)
    stage = on_stage or (lambda _s: None)
    stage = on_stage or (lambda _s: None)
    pose = io.read_joints()
    tip, axis = tip_and_axis(pose)
    say(f"start TCP {np.round(tip * 1000, 0)} mm")

    stage("CENTRE")
    target_px = _tool_axis_pixel(io, pose, axis)
    if not centre(io, detect, target_px, log=say):
        return {"ok": False, "reason": "centring failed"}

    stage("ALIGN")
    j5 = align_jaws(io, jaw_offset_deg)
    say(f"jaws aligned at j5 = {j5:+.0f} deg")

    stage("FIX")
    det = detect(io.capture_wrist())
    if det is None:
        return {"ok": False, "reason": "cube lost before back-projection"}
    est = io.unproject_to_table(det["centre"][0], det["centre"][1], GRASP.table_z_m)
    if est is None:
        return {"ok": False, "reason": "back-projection failed"}
    say(f"cube at {np.round(est * 1000, 0)} mm")

    # Descend straight down from where the centring left the tool axis. Moving laterally to
    # the back-projected estimate first was tried and made things worse: that estimate
    # carries an 8 mm silhouette bias, and steering to it put the fingers beside the cube.
    stage("DESCEND")
    tip_now, _ = tip_and_axis(io.read_joints())
    target = np.array([tip_now[0], tip_now[1], GRASP.fingertip_grasp_z_m])
    if not descend_guarded(io, detect, target, say):
        return {"ok": False, "reason": "descent did not converge"}
    say("at grasp height")

    # Close with the arm frozen: re-command the pose it already holds, and change only the
    # gripper. Every extra tip command during the close nudges the cube.
    stage("CLOSE")
    io.command(None, gripper=1.0)
    say("closed, arm held still")

    stage("LIFT")
    lift = dict(io.read_joints())
    lift[3] = lift[3] + 25.0
    io.command(lift, gripper=1.0)
    return {"ok": True, "cube": est.tolist()}


def _tool_axis_pixel(io, pose: dict[int, float], axis: np.ndarray):
    """Pixel where the tool axis meets the cube's plane, not the image centre.

    The camera is mounted off the tool axis, so aiming the cube at the image centre leaves
    the fingers beside it. The projection itself belongs to the backend: MuJoCo and
    OpenCV disagree about which way the optical axis points, and encoding that here made
    every target pixel wrong.
    """
    T = K.forward(joint_angles(pose))
    origin = T[:3, 3]
    point = (
        origin if abs(axis[2]) < 1e-6 else origin + (GRASP.table_z_m - origin[2]) / axis[2] * axis
    )
    uv = io.project(point)
    return uv if uv is not None else (160.0, 120.0)
