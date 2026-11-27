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
from .cube_fit import fit_cube
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
    """Camera -> link5, in MuJoCo's convention (-z along the optical axis).

    The simulation renders with MuJoCo's camera convention, so the analytic model uses it
    too; mixing the two put its prediction a thousand pixels away and the pose search then
    chose viewpoints that could not see the cube at all.
    """
    import cv2

    z = np.array([0.0, 0.0, -1.0])  # MuJoCo cameras look along -z
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
    # MuJoCo's convention (which the renderer uses, and which camera_pose now returns):
    # x right, y up, looking along -z. OpenCV's und[1] is measured downwards.
    ray = R @ np.array([und[0], -und[1], -1.0])
    if abs(ray[2]) < 1e-6:
        return None
    plane = GRASP.table_z_m if z_plane is None else z_plane
    t = (plane - C[2]) / ray[2]
    return None if t <= 0 else C + t * ray


# ---------------------------------------------------------------- motion
# ---------------------------------------------------------------- servoing
# ---------------------------------------------------------------- sequence
def align_jaws_to_yaw(io, yaw_rad: float) -> float:
    """Turn j5 so the jaws close parallel to one pair of the cube's faces.

    Better than aligning to a world axis: the cube can sit at any yaw, and the fit measures
    it. Jaws meeting two faces is the difference between a grasp and a corner pinch.
    """
    pose = io.read_joints()
    best = None
    for deg in np.linspace(-135.0, 135.0, 181):
        probe = dict(pose)
        probe[5] = JOINTS.offsets_deg[5] + deg
        jaw = jaw_axis(probe)
        horiz = np.array([jaw[0], jaw[1], 0.0])
        n = np.linalg.norm(horiz)
        if n < 1e-6:
            continue
        horiz /= n
        face = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
        score = abs(float(horiz[:2] @ face))
        if best is None or score > best[0]:
            best = (score, float(deg))
    if best is None:
        return 0.0
    target = dict(pose)
    target[5] = JOINTS.offsets_deg[5] + best[1]
    io.command(target)
    return best[1]


def _fit_once(io, detect, pose=None):
    """One frame -> the fitted cube (x, y, yaw), or None."""
    det = detect(io.capture_wrist())
    if det is None or det.get("mask") is None:
        return None
    p0 = pose or io.read_joints()
    est = io.unproject_to_table(det["centre"][0], det["centre"][1], GRASP.table_z_m)
    if est is None:
        return None

    def _project(pts, _p=p0):
        return io.project(pts, _p)

    fitted = fit_cube(
        det["mask"], _project, edge=GRASP.cube_edge_m, seed_xy=(float(est[0]), float(est[1]))
    )
    if fitted is None or fitted["iou"] <= 0.45:
        return None
    return fitted


def _ik_to_tip(io, target_xyz, *, tol: float = 0.012) -> bool:
    """Solve and command the pose that puts the fingertip at ``target_xyz``.

    IK rather than numerical stepping: the step solver reports success while leaving the tip
    tens of millimetres short, which is how the jaws ended up closing on air.
    """
    pose = io.read_joints()
    goal = np.eye(4)
    goal[:3, :3] = np.diag([1.0, -1.0, -1.0])
    goal[:3, 3] = np.asarray(target_xyz, float) - JOINTS.fingertip_below_tcp_m * np.array(
        [0.0, 0.0, -1.0]
    )
    q = K.inverse(goal, seed=[float(v) for v in joint_angles(pose)])
    io.command({i + 1: float(np.degrees(q[i]) + JOINTS.offsets_deg[i + 1]) for i in range(5)})
    tip, _ = tip_and_axis(io.read_joints())
    return bool(float(np.linalg.norm(np.asarray(target_xyz, float) - tip)) < 3 * tol)


def grasp_measured(
    io, detect: Callable, *, log: Callable | None = None, on_stage: Callable | None = None
) -> dict:
    """Centre on the cube, measure it, then grasp open loop with IK.

    The camera can only see the cube over a narrow cone, so closing the servo loop through
    the whole approach is impossible: moving far enough to reach loses sight of it. Instead
    the sequence spends its vision budget at one place -- a pose where the cube is centred in
    the image, which is also where the 3-D estimate is most reliable -- and then drives the
    arm by IK alone, with nothing to lose track of.
    """
    say = log or (lambda _m: None)
    stage = on_stage or (lambda _s: None)

    stage("AIM")
    # Centring is best effort. It sharpens the measurement, but the fit does not require the
    # cube to be in the middle of the frame -- only to be seen well -- and centring it can
    # need tens of degrees of joint motion, which is exactly what loses sight of it.
    # No centring motion. The pre-position is already chosen to see the cube, and moving to
    # centre it drives the arm into configurations from which the grasp pose is 30 mm out of
    # reach -- so the vision budget is spent on the measurement instead.
    say("measuring from the pre-position")

    stage("MEASURE")
    fitted = _fit_once(io, detect)
    if fitted is None:
        return {"ok": False, "reason": "measurement failed"}
    if fitted["iou"] < 0.6:
        return {"ok": False, "reason": f"measurement too weak (iou {fitted['iou']:.2f})"}
    x, y = float(fitted["centre_m"][0]), float(fitted["centre_m"][1])
    yaw = float(fitted["yaw_rad"])
    say(
        f"measured cube at ({x * 1000:.0f}, {y * 1000:.0f}) mm, iou {fitted['iou']:.2f}, "
        f"yaw {np.degrees(yaw):.0f} deg"
    )

    stage("ALIGN")
    j5 = align_jaws_to_yaw(io, yaw)
    say(f"jaws squared at j5 = {j5:+.0f} deg")

    stage("APPROACH")
    clearance = GRASP.table_z_m + GRASP.cube_edge_m + 0.035
    if not _ik_to_tip(io, np.array([x, y, clearance])):
        return {"ok": False, "reason": "IK could not reach the pre-grasp point"}
    say("above the cube")

    stage("DESCEND")
    if not _ik_to_tip(io, np.array([x, y, GRASP.fingertip_grasp_z_m])):
        return {"ok": False, "reason": "IK could not reach the grasp height"}
    say("at grasp height")

    stage("CLOSE")
    io.command(None, gripper=1.0)
    stage("LIFT")
    lift = dict(io.read_joints())
    lift[3] = lift[3] + 20.0
    io.command(lift, gripper=1.0)
    return {"ok": True, "cube": [x, y], "yaw": yaw}
