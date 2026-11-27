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


def _fit_once(io, detect, pose=None, *, retries: int = 1):
    """One frame -> the fitted cube (x, y, yaw), or None.

    Retries once. Observed on hardware: the first detection after a second camera thread
    starts comes back empty and the next one is fine, so a single retry turns a wasted run
    into a normal one. A measurement costs a second; a failed run costs the whole session.
    """
    det = None
    for _attempt in range(retries + 1):
        det = detect(io.capture_wrist())
        if det is not None and det.get("mask") is not None:
            break
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


def _ik_to_tip(io, target_xyz, *, tol: float = 0.004) -> bool:
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
    # Check the solve before commanding it. Sending a solution the arm cannot reach swings the
    # fingers through whatever is in the way on the way there -- one such command threw the
    # cube 419 mm across the cell. A pose that cannot be reached should not be ordered at all.
    tf = K.forward(q)
    predicted = tf[:3, 3] + JOINTS.fingertip_below_tcp_m * (tf[:3, :3] @ np.array([0.0, 0.0, 1.0]))
    predicted_error = float(np.linalg.norm(np.asarray(target_xyz, float) - predicted))
    if predicted_error >= 3 * tol:
        return False
    io.command({i + 1: float(np.degrees(q[i]) + JOINTS.offsets_deg[i + 1]) for i in range(5)})
    tip, _ = tip_and_axis(io.read_joints())
    error = float(np.linalg.norm(np.asarray(target_xyz, float) - tip))
    # 3 x tol = 12 mm: the jaws allow about 10 mm of lateral error, so anything looser is a
    # solve that reports success while the fingers close on air. 36 mm did exactly that.
    return bool(error < 3 * tol)


def measure_cube(io, detect: Callable, *, log: Callable | None = None) -> dict | None:
    """The cube's table position and yaw from a single frame, with no motion.

    Separate from the grasp so the arm can be asked what it sees without any of the
    consequences: the safest first question to ask of real hardware, and the gate the caller
    uses to decide whether a viewpoint is worth grasping from.
    """
    say = log or (lambda _m: None)
    fitted = _fit_once(io, detect)
    if fitted is None:
        return None
    if fitted["iou"] < 0.6:
        say(f"fit too weak (iou {fitted['iou']:.2f})")
        return None
    x, y = float(fitted["centre_m"][0]), float(fitted["centre_m"][1])
    yaw = float(fitted["yaw_rad"])
    say(
        f"cube at ({x * 1000:.0f}, {y * 1000:.0f}) mm, iou {fitted['iou']:.2f}, "
        f"yaw {np.degrees(yaw):.0f} deg"
    )
    return {"x": x, "y": y, "yaw": yaw, "iou": float(fitted["iou"])}


def grasp_measured(
    io,
    detect: Callable,
    *,
    log: Callable | None = None,
    on_stage: Callable | None = None,
    abort_check: Callable[[], None] | None = None,
    measure_only: bool = False,
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
    guard = abort_check or (lambda: None)

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

    if measure_only:
        return {"ok": True, "measured": True, "cube": [x, y], "yaw": yaw}

    guard()
    stage("ALIGN")
    j5 = align_jaws_to_yaw(io, yaw)
    say(f"jaws squared at j5 = {j5:+.0f} deg")

    stage("APPROACH")
    # The high pre-grasp point is not always reachable, and it is counter-intuitive which way
    # round that goes: with the cube 165 mm out, holding the tool axis vertical at 65 mm up
    # needs more extension than the arm has, while the grasp at 10 mm is comfortably inside
    # the workspace. So take the highest approach height that the kinematics actually allow,
    # and descend straight from it.
    clearance = GRASP.table_z_m + GRASP.cube_edge_m + 0.035
    for candidate in (clearance, 0.050, 0.035, 0.025):
        if _point_reachable(io, x, y, float(candidate)):
            clearance = float(candidate)
            break
    say(f"approaching at {clearance * 1000:.0f} mm")
    if not _ik_to_tip(io, np.array([x, y, clearance])):
        return {"ok": False, "reason": "IK could not reach the pre-grasp point"}
    say("above the cube")

    guard()
    stage("DESCEND")
    if not _ik_to_tip(io, np.array([x, y, GRASP.fingertip_grasp_z_m])):
        return {"ok": False, "reason": "IK could not reach the grasp height"}
    say("at grasp height")

    guard()
    stage("CLOSE")
    io.command(None, gripper=1.0)
    stage("LIFT")
    lift = dict(io.read_joints())
    lift[3] = lift[3] + 20.0
    io.command(lift, gripper=1.0)
    return {"ok": True, "cube": [x, y], "yaw": yaw}


def search_poses(
    intr: dict, seed_xy: tuple[float, float], *, limit: int = 12, step_deg: int = 10
) -> list[dict[int, float]]:
    """Servo poses from which the wrist camera should see ``seed_xy`` and can also grasp it.

    Purely geometric -- no simulation -- so the arm can use the same search the trial does.
    Two conditions, both necessary: the cube must land inside the frame, and the vertical
    grasp pose must be reachable from there. Poses that see the cube best are often ones from
    which the grasp is tens of millimetres out of reach, so both are checked.
    """
    f, cx, cy = intr["fy"], intr["cx"], intr["cy"]
    width, height = intr.get("image_size", [640, 480])
    scored: list[tuple[float, dict[int, float]]] = []
    for j1 in range(40, 141, step_deg):
        for j2 in range(30, 151, max(5, step_deg // 2)):
            for j3 in range(0, 181, step_deg):
                j4 = j2 + j3 - 90.0
                if not 0.0 <= j4 <= 180.0:
                    continue
                pose = {1: float(j1), 2: float(j2), 3: float(j3), 4: float(j4), 5: 135.0}
                tip, _ = tip_and_axis(pose)
                if tip[2] < 0.040:
                    continue
                cam_c, cam_r = camera_pose(pose)
                pc = (np.array([seed_xy[0], seed_xy[1], GRASP.table_z_m]) - cam_c) @ cam_r
                depth = -pc[2]
                if depth <= 1e-6:
                    continue
                u = cx + f * pc[0] / depth
                v = cy - f * pc[1] / depth
                if not (30.0 < u < width - 30.0 and 30.0 < v < height - 30.0):
                    continue
                goal = np.eye(4)
                goal[:3, :3] = np.diag([1.0, -1.0, -1.0])
                goal[:3, 3] = np.array(
                    [seed_xy[0], seed_xy[1], GRASP.fingertip_grasp_z_m]
                ) - JOINTS.fingertip_below_tcp_m * np.array([0.0, 0.0, -1.0])
                q_goal = K.inverse(goal, seed=[float(v) for v in joint_angles(pose)])
                tf = K.forward(q_goal)
                tip_goal = tf[:3, 3] + JOINTS.fingertip_below_tcp_m * (
                    tf[:3, :3] @ np.array([0.0, 0.0, 1.0])
                )
                reach = float(
                    np.linalg.norm(
                        tip_goal - np.array([seed_xy[0], seed_xy[1], GRASP.fingertip_grasp_z_m])
                    )
                )
                if reach > 0.012:
                    continue
                scored.append((float(np.hypot(u - 0.4 * width, v - 0.6 * height)), pose))
    scored.sort(key=lambda kv: kv[0])
    return [p for _d, p in scored[:limit]]


def camera_look_point(pose: dict[int, float]) -> np.ndarray | None:
    """Where the wrist camera's optical axis meets the table.

    The camera sits 57 mm off the tool axis and 95 mm above the fingertip, so it looks at a
    point well *beyond* the fingertips rather than under them -- roughly 56 mm past the tip
    when the tool axis is vertical. Planning a sweep from where the tip is therefore aims the
    camera at empty desk; this returns the point that actually matters.
    """
    cam_c, cam_r = camera_pose(pose)
    axis = -cam_r[:, 2]  # MuJoCo cameras look along -z
    if axis[2] >= -0.05:
        return None  # aimed at the horizon, not the desk
    t = (GRASP.table_z_m - cam_c[2]) / axis[2]
    if t <= 0:
        return None
    return cam_c + t * axis


def observer_sweep(
    *,
    min_radius_m: float = 0.10,
    max_radius_m: float = 0.30,
    tip_clearance_m: float = 0.045,
    limit: int = 12,
) -> list[dict[int, float]]:
    """Poses whose camera looks at the desk, chosen to cover it rather than to reach a point.

    Different question from :func:`search_poses`, which needs to know where the cube is. Here
    nothing is known yet, so the poses are picked so that their optical-axis footprints tile
    the area the arm can work in.
    """
    candidates: list[tuple[float, float, dict[int, float]]] = []
    for j1 in range(40, 141, 10):
        for j2 in range(65, 111, 5):
            for j3 in range(0, 51, 5):
                j4 = j2 + j3 - 90.0
                if not 0.0 <= j4 <= 180.0:
                    continue
                pose = {1: float(j1), 2: float(j2), 3: float(j3), 4: float(j4), 5: 135.0}
                tip, _ = tip_and_axis(pose)
                if tip[2] < tip_clearance_m:
                    continue
                hit = camera_look_point(pose)
                if hit is None:
                    continue
                r = float(np.hypot(hit[0], hit[1]))
                if not min_radius_m <= r <= max_radius_m:
                    continue
                candidates.append((r, float(np.arctan2(hit[1], hit[0])), pose))
    if not candidates:
        return []
    # Tile the reachable annulus: a few radii against several bearings, one pose per cell,
    # nearest to the cell's centre. A handful of hand-picked poses leaves holes big enough for
    # the cube to hide in, and the miss shows up later as "no pose gave a usable fit".
    radii = np.linspace(min_radius_m + 0.04, max_radius_m - 0.03, 3)
    bearings = np.linspace(-np.pi, np.pi, 6, endpoint=False)
    chosen: list[dict[int, float]] = []
    for target_r in radii:
        for bearing in bearings:
            best = min(
                candidates,
                key=lambda c: abs(c[0] - target_r) + 0.15 * abs(c[1] - bearing),
            )
            if best[2] not in chosen:
                chosen.append(best[2])
            if len(chosen) >= limit:
                return chosen
    return chosen
    return chosen


def project_points(C, R, intr: dict, points_m) -> np.ndarray:
    """Base-frame point(s) -> pixel(s), MuJoCo convention (optical axis -z).

    A free function so it can be tested without a robot: the real backend's version of this
    was wrong in two ways -- single point only, and mirrored about the image centre -- and
    neither could be seen until an arm was attached.
    """
    pts = np.atleast_2d(np.asarray(points_m, float))
    local = (pts - np.asarray(C, float)) @ np.asarray(R, float)
    depth = -local[:, 2]
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    dist = np.asarray(intr.get("dist", [0.0, 0.0]), float).ravel()
    k1 = float(dist[0]) if dist.size > 0 else 0.0
    k2 = float(dist[1]) if dist.size > 1 else 0.0
    out = np.full((local.shape[0], 2), np.nan)
    for i in range(local.shape[0]):
        if depth[i] <= 1e-6:
            continue
        xn = local[i, 0] / depth[i]
        yn = -local[i, 1] / depth[i]  # image y grows downward
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        out[i] = (cx + fx * xn * radial, cy + fy * yn * radial)
    return out


# The flow's fixed poses. PARK is where every failure returns; OBSERVE is where the arm waits
# with the camera pointed at the desk, looking at (0, 166) mm with a 70 mm footprint.
PARK_POSE: dict[int, float] = {1: 90.0, 2: 90.0, 3: 90.0, 4: 90.0, 5: 135.0}
#: Where the arm waits with the camera pointed at the desk. It looks at (0, 141) mm with a
#: 70 mm footprint. Aimed closer than about 150 mm on purpose: at 165 mm the arm sits at the
#: edge of what a vertical tool axis can reach and droops 17 mm below the commanded height,
#: which is enough to miss the cube. 140 mm is where it still has margin.
OBSERVE_POSE: dict[int, float] = {1: 90.0, 2: 90.0, 3: 0.0, 4: 0.0, 5: 135.0}


def is_yellow(frame: np.ndarray, mask: np.ndarray) -> bool:
    """Whether the segmented region is the yellow target rather than something square.

    Colour as a *gate on the segmenter's output*, never as the detector. Asked to find the
    cube, a colour threshold could not: the cube's own mean saturation measured 139 against a
    gate of 140. Asked to confirm what segmentation already found, it is reliable and it is
    what rejects the blue box the segmenter is happy to segment at the same spot.
    """
    if frame is None or mask is None or not mask.any():
        return False
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return False
    hue = float(hsv[ys, xs, 0].mean())
    sat = float(hsv[ys, xs, 1].mean())
    val = float(hsv[ys, xs, 2].mean())
    return 5.0 <= hue <= 45.0 and sat >= 90.0 and val >= 80.0


def watch_for_cube(
    io,
    detect: Callable,
    *,
    timeout_s: float = 30.0,
    period_s: float = 1.0,
    log: Callable | None = None,
    abort_check: Callable[[], None] | None = None,
) -> dict | None:
    """Wait at the observation pose for a yellow cube to enter the wrist camera's view.

    The contract is that the cube is placed so at least part of it is in frame; what this
    loop adds is the discipline of waiting for it rather than assuming it, and giving up with
    a reason when it does not come. A robot that waits forever is not a state machine.
    """
    import time

    say = log or (lambda _m: None)
    guard = abort_check or (lambda: None)
    deadline = time.time() + timeout_s
    attempts = 0
    while time.time() < deadline:
        guard()
        attempts += 1
        frame = io.capture_wrist()
        found = detect(frame) if frame is not None else None
        if found is not None and is_yellow(frame, found.get("mask")):
            say(
                f"cube in view after {attempts} attempt(s) at "
                f"{tuple(round(float(v)) for v in found['centre'])}"
            )
            return found
        if found is not None:
            say(f"  attempt {attempts}: something square but not yellow -- ignoring")
        time.sleep(period_s)
    say(f"nothing yellow in view after {attempts} attempts in {timeout_s:.0f} s")
    return None


def centre_cube(
    io,
    detect: Callable,
    *,
    max_iters: int = 6,
    max_step_deg: float = 3.0,
    tol_px: float = 25.0,
    intr: dict | None = None,
    seed_xy=None,
    log: Callable | None = None,
) -> dict | None:
    """Nudge the arm until the cube's centroid is near the image centre, without losing reach.

    Bounded on purpose. Unbounded centring is what previously drove the arm into a
    configuration from which the grasp pose was 30 mm out of reach, so every step is followed
    by a reachability check and undone if it makes the grasp impossible. A badly-framed cube
    that can be grasped beats a well-framed one that cannot.
    """
    say = log or (lambda _m: None)
    if intr is None:
        intr = io.intrinsics() if hasattr(io, "intrinsics") else None
    for step in range(max_iters + 1):
        frame = io.capture_wrist()
        found = detect(frame) if frame is not None else None
        if found is None:
            return None
        centre = np.asarray(found["centre"], float)
        err = centre - np.array([320.0, 240.0])
        if float(np.hypot(*err)) <= tol_px:
            say(f"centred after {step} step(s), {np.hypot(*err):.0f} px off")
            return found
        if step == max_iters or intr is None:
            say(f"stopping at {np.hypot(*err):.0f} px off (budget spent)")
            return found
        before = dict(io.read_joints())
        # small, bounded correction towards the centre; sign from the image Jacobian
        J = None
        if seed_xy is not None:
            point = np.array([seed_xy[0], seed_xy[1], GRASP.table_z_m])
            J = _image_jacobian_at(io, before, point)
        if J is None:
            say("no usable image Jacobian; keeping the current view")
            return found
        dq = np.clip(-0.5 * np.linalg.pinv(J) @ err, -max_step_deg, max_step_deg)
        target = dict(before)
        for k in range(min(4, len(dq))):
            target[k + 1] = before[k + 1] + float(dq[k])
        io.command(target)
        if seed_xy is not None and intr is not None:
            from . import controller as _self

            if not _self.search_poses(intr, seed_xy, limit=1):
                io.command(before)
                say("that step would put the grasp out of reach; undone")
                return found
    return None


def _image_jacobian_at(io, pose: dict[int, float], point_m) -> np.ndarray | None:
    """d(pixel)/d(servo) at a world point, by finite differences through the camera model."""
    base = io.project(np.asarray(point_m, float), pose)
    if base is None or not np.all(np.isfinite(base)):
        return None
    cols = []
    for k in range(1, 5):
        bumped = dict(pose)
        bumped[k] = pose[k] + 1.0
        uv = io.project(np.asarray(point_m, float), bumped)
        if uv is None or not np.all(np.isfinite(uv)):
            return None
        cols.append(np.asarray(uv, float) - np.asarray(base, float))
    return np.column_stack(cols)


def observer_cube_width(io) -> float | None:
    """Apparent width of the cube in the observer view, for verifying a lift."""
    frame = io.capture_observer()
    if frame is None:
        return None
    from .observer_vision import cube_side_px

    return cube_side_px(frame)


def hold_grasp(
    io,
    seconds: float,
    *,
    log: Callable | None = None,
    abort_check: Callable[[], None] | None = None,
) -> None:
    """Keep the jaws closed and the joints commanded, re-sending so the bus keeps torque on."""
    import time

    say = log or (lambda _m: None)
    guard = abort_check or (lambda: None)
    pose = dict(io.read_joints())
    deadline = time.time() + seconds
    while time.time() < deadline:
        guard()
        io.command(dict(pose), gripper=1.0)
        time.sleep(min(5.0, max(0.5, deadline - time.time())))
    say(f"held the grasp for {seconds:.0f} s")


def go_park(io, *, log: Callable | None = None) -> None:
    (log or (lambda _m: None))("returning to park")
    io.command(dict(PARK_POSE), gripper=JOINTS.gripper_open_deg)


def _point_reachable(io, x: float, y: float, z: float, *, tol_m: float = 0.012) -> bool:
    """Whether the fingertip can be put at (x, y, z) with the tool axis vertical, from here."""
    pose = io.read_joints()
    goal = np.eye(4)
    goal[:3, :3] = np.diag([1.0, -1.0, -1.0])
    goal[:3, 3] = np.array([x, y, z]) - JOINTS.fingertip_below_tcp_m * np.array([0.0, 0.0, -1.0])
    q = K.inverse(goal, seed=[float(v) for v in joint_angles(pose)])
    tf = K.forward(q)
    tip = tf[:3, 3] + JOINTS.fingertip_below_tcp_m * (tf[:3, :3] @ np.array([0.0, 0.0, 1.0]))
    return bool(float(np.linalg.norm(tip - np.array([x, y, z]))) < tol_m)


def _grasp_reachable(io, x: float, y: float, *, tol_m: float = 0.012) -> bool:
    """Whether the vertical grasp at (x, y) can be reached from where the arm is now."""
    pose = io.read_joints()
    goal = np.eye(4)
    goal[:3, :3] = np.diag([1.0, -1.0, -1.0])
    goal[:3, 3] = np.array(
        [x, y, GRASP.fingertip_grasp_z_m]
    ) - JOINTS.fingertip_below_tcp_m * np.array([0.0, 0.0, -1.0])
    q = K.inverse(goal, seed=[float(v) for v in joint_angles(pose)])
    tf = K.forward(q)
    tip = tf[:3, 3] + JOINTS.fingertip_below_tcp_m * (tf[:3, :3] @ np.array([0.0, 0.0, 1.0]))
    return bool(float(np.linalg.norm(tip - np.array([x, y, GRASP.fingertip_grasp_z_m]))) < tol_m)


def grasp_task(
    io,
    detect: Callable,
    *,
    log: Callable | None = None,
    on_stage: Callable | None = None,
    abort_check: Callable | None = None,
    intr: dict | None = None,
    watch_timeout_s: float = 30.0,
    hold_s: float = 60.0,
) -> dict:
    """park -> observe -> watch -> centre -> measure -> grasp -> verify -> hold or park."""
    say = log or (lambda _m: None)
    stage = on_stage or (lambda _s: None)
    guard = abort_check or (lambda: None)

    stage("PARK")
    go_park(io, log=say)
    guard()

    stage("OBSERVE")
    io.command(dict(OBSERVE_POSE), gripper=JOINTS.gripper_open_deg)
    say("watching for the cube")

    stage("WATCH")
    found = watch_for_cube(io, detect, timeout_s=watch_timeout_s, log=say, abort_check=guard)
    if found is None:
        go_park(io, log=say)
        return {"ok": False, "reason": "no yellow cube entered the wrist camera's view"}

    # "Centre the cube in the camera" is the one step that does not work on this arm.
    # Continuous centring needs tens of degrees of joint motion, and the poses that frame the
    # cube best are frequently ones from which the vertical grasp is 20-30 mm out of reach --
    # so an arm that has centred itself can no longer reach its target. What does work is the
    # small discrete set of poses that both see a point and can grasp it, so the flow picks
    # from that set instead of servoing.
    stage("CENTRE")
    rough = io.unproject_to_table(found["centre"][0], found["centre"][1], GRASP.table_z_m)
    seed_xy = None if rough is None else (float(rough[0]), float(rough[1]))
    aligned = found
    if seed_xy is not None and intr is not None:
        say(
            f"cube roughly at ({seed_xy[0] * 1000:.0f}, {seed_xy[1] * 1000:.0f}) mm; "
            f"choosing a view that can also reach it"
        )
        for cand in search_poses(intr, seed_xy, limit=8):
            io.command(dict(cand), gripper=JOINTS.gripper_open_deg)
            seen = detect(io.capture_wrist())
            if seen is None:
                continue
            again = measure_cube(io, detect)
            if again is None:
                continue
            if _point_reachable(io, again["x"], again["y"], GRASP.fingertip_grasp_z_m):
                where = {k: round(v) for k, v in cand.items()}
                say(
                    f"view {where} measures ({again['x'] * 1000:.0f}, "
                    f"{again['y'] * 1000:.0f}) mm and can reach it"
                )
                aligned = seen
                break
        else:
            say("no pose both saw the cube and could reach it")
    if aligned is None:
        go_park(io, log=say)
        return {"ok": False, "reason": "lost the cube while choosing a view"}

    stage("MEASURE")
    measured = measure_cube(io, detect, log=say)
    if measured is None:
        go_park(io, log=say)
        return {"ok": False, "reason": "measurement failed"}
    x, y, yaw = measured["x"], measured["y"], measured["yaw"]

    # The pose that frames the cube best is not always one that can reach it: the observation
    # and centring steps can leave the arm in a configuration whose vertical grasp is 20-30 mm
    # short, and the IK then "succeeds" while the fingers close on air. Check, and if the
    # current pose cannot reach, move to one that can and measure again.
    if not _point_reachable(io, x, y, GRASP.fingertip_grasp_z_m):
        say("this pose cannot reach the grasp; moving to one that can")
        for cand in search_poses(intr, (x, y), limit=6) if intr else []:
            io.command(cand, gripper=JOINTS.gripper_open_deg)
            again = measure_cube(io, detect)
            if again is not None and _grasp_reachable(io, again["x"], again["y"]):
                x, y, yaw = again["x"], again["y"], again["yaw"]
                say(
                    f"re-measured from { {k: round(v) for k, v in cand.items()} }: "
                    f"({x * 1000:.0f}, {y * 1000:.0f}) mm"
                )
                break

    stage("ALIGN")
    align_jaws_to_yaw(io, yaw)

    stage("APPROACH")
    # The high pre-grasp point is not always reachable, and it is counter-intuitive which way
    # round that goes: with the cube 165 mm out, holding the tool axis vertical at 65 mm up
    # needs more extension than the arm has, while the grasp at 10 mm is comfortably inside
    # the workspace. So take the highest approach height that the kinematics actually allow,
    # and descend straight from it.
    clearance = GRASP.table_z_m + GRASP.cube_edge_m + 0.035
    for candidate in (clearance, 0.050, 0.035, 0.025):
        if _point_reachable(io, x, y, float(candidate)):
            clearance = float(candidate)
            break
    say(f"approaching at {clearance * 1000:.0f} mm")
    if not _ik_to_tip(io, np.array([x, y, clearance])):
        go_park(io, log=say)
        return {"ok": False, "reason": "IK could not reach the pre-grasp point"}
    tip_now, _axis = tip_and_axis(io.read_joints())
    say(
        f"above the cube: tip ({tip_now[0] * 1000:.0f}, {tip_now[1] * 1000:.0f}, "
        f"{tip_now[2] * 1000:.0f}) mm, target ({x * 1000:.0f}, {y * 1000:.0f}, {clearance * 1000:.0f})"
    )
    guard()

    stage("DESCEND")
    if not _ik_to_tip(io, np.array([x, y, GRASP.fingertip_grasp_z_m])):
        go_park(io, log=say)
        return {"ok": False, "reason": "IK could not reach the grasp height"}
    tip_now, _axis = tip_and_axis(io.read_joints())
    say(
        f"at grasp height: tip ({tip_now[0] * 1000:.0f}, {tip_now[1] * 1000:.0f}, "
        f"{tip_now[2] * 1000:.0f}) mm, target z {GRASP.fingertip_grasp_z_m * 1000:.0f}"
    )
    guard()

    stage("CLOSE")
    io.command(None, gripper=1.0)

    stage("LIFT")
    lifted = dict(io.read_joints())
    lifted[3] = lifted[3] + 20.0
    io.command(lifted, gripper=1.0)

    # Verify from the wrist camera, in the same view the cube was measured from. Held in the
    # jaws it sits ~60 mm from the lens instead of ~150 mm on the table, so it covers several
    # times the area; a cube still on the table does not. That needs no extrinsics and no
    # second camera, and the difference is large enough to be unambiguous.
    stage("VERIFY")
    io.command(dict(OBSERVE_POSE), gripper=1.0)
    after = detect(io.capture_wrist())
    area_before = float(found.get("area", 0) or 0)
    area_after = 0.0 if after is None else float(after.get("area", 0) or 0)
    ratio = area_after / area_before if area_before > 0 else 0.0
    verified = ratio >= 2.0
    say(f"verify: cube area {area_before:.0f} -> {area_after:.0f} px (x{ratio:.1f})")
    if not verified:
        go_park(io, log=say)
        return {
            "ok": False,
            "reason": "the grasp did not verify",
            "cube": [x, y],
            "yaw": yaw,
            "area_ratio": ratio,
        }

    stage("HOLD")
    hold_grasp(io, hold_s, log=say, abort_check=guard)
    return {"ok": True, "cube": [x, y], "yaw": yaw, "area_ratio": ratio, "measured": measured}
