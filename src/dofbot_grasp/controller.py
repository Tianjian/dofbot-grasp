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
    for k in np.nonzero(free)[0]:
        target[k + 1] = pose[k + 1] + float(
            np.clip(dqf[np.sum(free[:k])] * gain, -max_deg, max_deg)
        )
    io.command(target)


def move_tip_to(io, target_xyz, *, max_iters: int = 140, tol: float = 0.005) -> bool:
    """Walk the fingertip onto ``target_xyz`` with small damped steps.

    A single IK solve looked tempting but stalls when the elbow sits on its limit; small
    closed-loop steps keep working because each one is re-measured from the read-back.
    """
    target = np.asarray(target_xyz, float)
    for _ in range(max_iters):
        tip, _ = tip_and_axis(io.read_joints())
        err = target - tip
        if np.linalg.norm(err) < tol:
            return True
        move_tip(io, np.clip(err, -0.010, 0.010))
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
        if abs(e[0]) < GRASP.deadband_px and abs(e[1]) < GRASP.deadband_px:
            return True
        dq = np.clip(
            -GRASP.gain * np.linalg.pinv(J) @ np.asarray(e), -GRASP.max_step_deg, GRASP.max_step_deg
        )
        pose = io.read_joints()
        target = dict(pose)
        for k in range(3):
            target[k + 1] = pose[k + 1] + float(dq[k])
        io.command(target)
    return False


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
    for k in range(3):
        probe = dict(pose)
        probe[k + 1] += GRASP.probe_deg
        moved = io.project(point_m, probe)
        if moved is None:
            return None
        cols.append((np.array(moved) - np.array(base)) / GRASP.probe_deg)
    return np.column_stack(cols)


def align_jaws(io) -> float:
    """Turn j5 until the jaw axis is parallel to a world axis; returns the angle used."""
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
    target[5] = JOINTS.offsets_deg[5] + best[1]
    io.command(target)
    return float(best[1])


# ---------------------------------------------------------------- sequence
def grasp(io, detect: Callable, *, log: Callable | None = None) -> dict:
    """Run the whole sequence and report what happened."""
    say = log or (lambda _m: None)
    pose = io.read_joints()
    tip, axis = tip_and_axis(pose)
    say(f"start TCP {np.round(tip * 1000, 0)} mm")

    target_px = _tool_axis_pixel(io, pose, axis)
    if not centre(io, detect, target_px, log=say):
        return {"ok": False, "reason": "centring failed"}

    j5 = align_jaws(io)
    say(f"jaws aligned at j5 = {j5:+.0f} deg")

    det = detect(io.capture_wrist())
    if det is None:
        return {"ok": False, "reason": "cube lost before back-projection"}
    est = io.unproject_to_table(det["centre"][0], det["centre"][1], GRASP.table_z_m)
    if est is None:
        return {"ok": False, "reason": "back-projection failed"}
    say(f"cube at {np.round(est * 1000, 0)} mm")

    # Descend along the tool axis that the centring just aligned, not towards the
    # back-projected estimate: that estimate carries the silhouette-centroid bias (a few
    # millimetres), and descending to it pushed the cube 10 mm across the table instead of
    # straddling it.
    tip_now, _ = tip_and_axis(io.read_joints())
    target = np.array([tip_now[0], tip_now[1], GRASP.fingertip_grasp_z_m])
    if not move_tip_to(io, target, max_iters=80):
        return {"ok": False, "reason": "descent did not converge"}
    say("at grasp height")

    io.command(io.read_joints(), gripper=JOINTS.gripper_shut_deg)
    lift = dict(io.read_joints())
    lift[3] = lift[3] + 25.0
    io.command(lift, gripper=JOINTS.gripper_shut_deg)
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
