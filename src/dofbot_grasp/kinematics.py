"""Forward/inverse kinematics for the DOFBOT SE 5-DOF desktop arm.

The kinematic chain is a literal transcription of
``prior/arm_cam_control/dofbot_info/urdf/dofbot.urdf`` (read-only reference):

===========  ==========================  =========================  ========
joint        origin xyz (m)              origin rpy (rad)           axis
===========  ==========================  =========================  ========
joint1       (0, 0, 0.06605)             (-0.010805, 0, 0)          +Z
joint2       (0, -0.00031873, 0.04145)   (0, 1.5708, 0)             +Z
joint3       (-0.08285, 0, 0)            (0, 0, 0)                  +Z
joint4       (-0.08285, 0, 0)            (0, 0, 0.0083081)          +Z
joint5       (-0.07385, -0.001, 0)       (0, -1.57, 0)              +Z
===========  ==========================  =========================  ========

URDF semantics: ``T_parent_child = T_origin(xyz, rpy) @ R_axis(axis, q)``
with ``rpy`` a fixed-axis roll/pitch/yaw rotation (R = Rz(yaw) Ry(pitch) Rx(roll)).
No gripper link exists in the URDF, so the "end effector" frame is link5.
Gripper opening is a separate servo (ID 6) and never part of the chain.

Joint limits (URDF):
joint1..joint4 [-1.5708, 1.5708] rad, joint5 [-1.5708, 3.1416] rad.

This module is pure NumPy + stdlib by design so unit tests never need
MuJoCo, cv2 or a serial port.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "JOINT_AXES",
    "JOINT_LIMITS",
    "JOINT_LOWER",
    "JOINT_NAMES",
    "JOINT_ORIGINS_RPY",
    "JOINT_ORIGINS_XYZ",
    "JOINT_UPPER",
    "N_JOINTS",
    "SERVO_CENTER_DEG",
    "SERVO_IDS",
    "SERVO_RANGE_DEG",
    "SERVO_REVERSED_IDS",
    "TCP_OFFSET_LINK5_M",
    "clamp_joints",
    "clamp_servo_deg",
    "forward",
    "forward_all",
    "forward_tcp",
    "fraction_to_gripper_deg",
    "gripper_deg_to_fraction",
    "inverse",
    "inverse_position",
    "joint_deg_to_raw_servo",
    "joint_deg_to_servo",
    "joint_rad_to_servo",
    "make_transform",
    "raw_servo_to_joint_deg",
    "rot_log",
    "rpy_to_matrix",
    "servo_to_joint_deg",
    "servo_to_joint_rad",
]

N_JOINTS = 5

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5")

#: Origin translation of each joint, relative to its parent link (URDF xyz).
JOINT_ORIGINS_XYZ = np.array(
    [
        (0.0, 0.0, 0.06605),
        (0.0, -0.00031873, 0.04145),
        (-0.08285, 0.0, 0.0),
        (-0.08285, 0.0, 0.0),
        (-0.07385, -0.001, 0.0),
    ],
    dtype=float,
)

#: Origin fixed-axis rpy (roll, pitch, yaw), URDF ``rpy`` attribute.
JOINT_ORIGINS_RPY = np.array(
    [
        (-0.010805, 0.0, 0.0),
        (0.0, 1.5708, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0083081),
        (0.0, -1.57, 0.0),
    ],
    dtype=float,
)

#: All five URDF joints rotate about their local +Z axis.
JOINT_AXES = np.tile(np.array([0.0, 0.0, 1.0]), (N_JOINTS, 1))

#: Tool-centre-point offset, in the link5 frame: 50 mm down the tool (+Z) axis.
#: Mirrors the ``tcp`` site in ``dofbot_grasp/sim/dofbot_se.xml``.
TCP_OFFSET_LINK5_M = np.array([0.0, 0.0, 0.050])

#: (lower, upper) joint limits in radians, taken verbatim from the URDF.
JOINT_LIMITS = np.array(
    [
        (-1.5708, 1.5708),
        (-1.5708, 1.5708),
        (-1.5708, 1.5708),
        (-1.5708, 1.5708),
        (-1.5708, 3.1416),
    ],
    dtype=float,
)

JOINT_LOWER = JOINT_LIMITS[:, 0]
JOINT_UPPER = JOINT_LIMITS[:, 1]

# ---------------------------------------------------------------------------
# Servo <-> joint mapping (see prior/arm_cam_control/ANGLE_MAPPING.md)
# ---------------------------------------------------------------------------

#: Arm_Lib servo IDs: 1..5 are arm joints, 6 is the gripper.
SERVO_IDS = (1, 2, 3, 4, 5, 6)

#: Mechanical travel of each servo, in degrees (Arm_Lib.py / ANGLE_MAPPING.md).
#: IDs 1-4 and 6 are 0-180 deg; ID 5 is 0-270 deg.
SERVO_RANGE_DEG = (180.0, 180.0, 180.0, 180.0, 270.0, 180.0)

#: Servo IDs whose *raw* direction is reversed inside Arm_Lib
#: (``Arm_serial_servo_write`` / ``Arm_serial_servo_write6`` apply ``180 - angle``).
#: ``Arm_serial_servo_read`` already un-reverses them, so values returned by the
#: library are "logical" servo angles: for these IDs, joint = logical - 90.
SERVO_REVERSED_IDS = (2, 3, 4)

#: Logical servo angle that corresponds to joint angle 0 rad, per servo.
#: ID 1-4 -> 90 deg, ID 5 -> 135 deg (its mechanical mid-travel), ID 6 gripper -> 90.
SERVO_CENTER_DEG = (90.0, 90.0, 90.0, 90.0, 135.0, 90.0)

_JOINT_SERVO_CENTER = np.array(SERVO_CENTER_DEG[:N_JOINTS], dtype=float)


def clamp_joints(joints_rad: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``joints_rad`` clamped to the URDF joint limits (new array)."""
    q = np.asarray(joints_rad, dtype=float).reshape(-1)
    if q.shape != (N_JOINTS,):
        raise ValueError(f"expected {N_JOINTS} joint values, got shape {q.shape}")
    return np.clip(q, JOINT_LOWER, JOINT_UPPER)


def clamp_servo_deg(servo_deg: Sequence[float] | np.ndarray) -> np.ndarray:
    """Clamp servo degree command(s) (last axis = 6) to mechanical ranges."""
    s = np.asarray(servo_deg, dtype=float)
    if s.shape == () or s.shape[-1] != 6:
        raise ValueError(f"expected servo angles with last dimension 6, got shape {s.shape}")
    return np.clip(s, 0.0, np.asarray(SERVO_RANGE_DEG, dtype=float))


def servo_to_joint_deg(servo_deg: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert *logical* servo angles to joint degrees.

    Input is the 6-element (or 5-element) servo vector as returned by
    ``Arm_serial_servo_read`` / passed to ``Arm_serial_servo_write6``.
    Per ``ANGLE_MAPPING.md``:

    * joint1..joint4 = servo[0..3] - 90
    * joint5         = servo[4] - 135  (S5 travels 0-270, mid = 135)
    * ID 6 is the gripper and is ignored.

    Note: ``ANGLE_MAPPING.md`` is internally inconsistent for S5 -- its formula
    ``servo/270*270 - 135`` and its mid-point check (servo 135 -> joint 0) imply
    joint5 in [-135, 135] deg, while its endpoint examples claim servo 0 -> -90
    and servo 270 -> 180.  This module follows the formula and the mid-point
    check (the only self-consistent reading), so the URDF's joint5 range
    -90..+180 deg is only partly reachable through the servo (max +135 deg);
    values beyond that saturate at 270 deg when converted back to a command.

    IDs 2/3/4 are the mechanically reversed servos; ``Arm_Lib`` performs the
    ``180 - angle`` internally, so the logical value already follows the same
    ``-90`` offset as the others (applying ``90 - s`` again is the documented
    "double reverse" bug).
    """
    s = np.asarray(servo_deg, dtype=float)
    if s.ndim == 0 or s.shape[-1] not in (5, 6):
        raise ValueError(f"expected servo angles with last dimension 5 or 6, got shape {s.shape}")
    joints = s[..., :N_JOINTS] - _JOINT_SERVO_CENTER
    return joints


def joint_deg_to_servo(
    joint_deg: Sequence[float] | np.ndarray, gripper_deg: float = 90.0
) -> np.ndarray:
    """Inverse of :func:`servo_to_joint_deg` (logical servo angles, 6 elements)."""
    j = np.asarray(joint_deg, dtype=float)
    if j.ndim == 0 or j.shape[-1] != N_JOINTS:
        raise ValueError(
            f"expected joint values with last dimension {N_JOINTS}, got shape {j.shape}"
        )
    servo = np.empty(j.shape[:-1] + (6,), dtype=float)
    servo[..., :N_JOINTS] = j + _JOINT_SERVO_CENTER
    servo[..., 5] = float(gripper_deg)
    return clamp_servo_deg(servo)


def raw_servo_to_joint_deg(raw_servo_deg: Sequence[float]) -> np.ndarray:
    """Convert raw (physical) servo angles to joint degrees, applying reverse.

    "Raw" here means the mechanical 0-180/0-270 value before ``Arm_Lib``'s
    logical un-reversal.  IDs 2/3/4 are physically inverted, so their effective
    logical angle is ``180 - raw`` and the joint angle is ``effective - 90``.
    Useful when talking to the servo bus without Arm_Lib.
    """
    s = np.asarray(raw_servo_deg, dtype=float)
    if s.ndim == 0 or s.shape[-1] not in (5, 6):
        raise ValueError(f"expected servo angles with last dimension 5 or 6, got shape {s.shape}")
    logical = s[..., :N_JOINTS].copy()
    for sid in SERVO_REVERSED_IDS:  # IDs are 1-based
        logical[..., sid - 1] = 180.0 - logical[..., sid - 1]
    return logical - _JOINT_SERVO_CENTER


def joint_deg_to_raw_servo(joint_deg: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`raw_servo_to_joint_deg` for servo IDs 1-5 (raw degrees)."""
    j = np.asarray(joint_deg, dtype=float)
    if j.ndim == 0 or j.shape[-1] != N_JOINTS:
        raise ValueError(
            f"expected joint values with last dimension {N_JOINTS}, got shape {j.shape}"
        )
    raw = j + _JOINT_SERVO_CENTER
    for sid in SERVO_REVERSED_IDS:
        raw[..., sid - 1] = 180.0 - raw[..., sid - 1]
    return np.clip(raw, 0.0, np.asarray(SERVO_RANGE_DEG[:N_JOINTS], dtype=float))


def servo_to_joint_rad(servo_deg: Sequence[float]) -> np.ndarray:
    """Logical servo degrees -> joint radians."""
    return np.radians(servo_to_joint_deg(servo_deg))


def joint_rad_to_servo(joints_rad: Sequence[float], gripper_deg: float = 90.0) -> np.ndarray:
    """Joint radians -> logical servo degrees (6 elements)."""
    return joint_deg_to_servo(np.degrees(np.asarray(joints_rad, dtype=float)), gripper_deg)


def gripper_deg_to_fraction(gripper_deg: float) -> float:
    """Map gripper servo angle 0-180 deg to a 0..1 opening fraction (0 = closed)."""
    return float(np.clip(gripper_deg, 0.0, 180.0) / 180.0)


def fraction_to_gripper_deg(fraction: float) -> float:
    """Map a 0..1 gripper opening fraction to a 0-180 deg servo command."""
    return float(np.clip(fraction, 0.0, 1.0) * 180.0)


# ---------------------------------------------------------------------------
# Rigid-body helpers
# ---------------------------------------------------------------------------


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis rpy -> rotation matrix (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def make_transform(
    rotation: np.ndarray | None = None, translation: Sequence[float] = (0, 0, 0)
) -> np.ndarray:
    """Build a 4x4 homogeneous transform."""
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = np.asarray(rotation, dtype=float)
    t[:3, 3] = np.asarray(translation, dtype=float)
    return t


def rot_log(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis-angle, radians)."""
    r = np.asarray(rotation, dtype=float)
    cos_theta = (np.trace(r) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        return 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta > np.pi - 1e-6:
        # Near pi: recover the axis from the symmetric part R + I = 2 a a^T.
        a = np.sqrt(np.maximum((np.diag(r) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(a))
        sign = 1.0 if r[k, (k + 1) % 3] - r[(k + 1) % 3, k] >= 0 else -1.0
        axis = np.zeros(3)
        axis[k] = a[k]
        axis[(k + 1) % 3] = sign * a[(k + 1) % 3]
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        axis /= np.linalg.norm(axis)
        return axis * theta
    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


def _joint_transform(index: int, q: float) -> np.ndarray:
    """Fixed origin transform for joint ``index`` followed by its rotation."""
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*JOINT_ORIGINS_RPY[index])
    t[:3, 3] = JOINT_ORIGINS_XYZ[index]
    c, s = np.cos(q), np.sin(q)
    rz = np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return t @ rz


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def forward(joints_rad: Sequence[float] | np.ndarray) -> np.ndarray:
    """End-effector (link5) pose in the robot base frame as a 4x4 matrix.

    Parameters
    ----------
    joints_rad : sequence of 5 floats
        Joint angles in radians, ordered joint1..joint5.

    Returns
    -------
    np.ndarray, shape (4, 4)
        ``T_base_ee``; the frame origin is the link5 (gripper base) frame origin
        and its +X axis points along the gripper approach direction.
    """
    q = np.asarray(joints_rad, dtype=float).reshape(-1)
    if q.shape != (N_JOINTS,):
        raise ValueError(f"expected {N_JOINTS} joint values, got shape {q.shape}")
    t = np.eye(4)
    for i in range(N_JOINTS):
        t = t @ _joint_transform(i, q[i])
    return t


def forward_tcp(joints_rad: Sequence[float]) -> np.ndarray:
    """Tool-centre-point pose: link5 frame shifted 50 mm along its +Z tool axis.

    The URDF has no gripper link, so :func:`forward` stops at link5.  The
    sim/real contract defines the grasp point as the ``tcp`` site at
    ``(0, 0, 0.050)`` in link5; this helper gives that pose (validated against
    MuJoCo ``site_xpos``/``site_xmat`` of ``dofbot_se.xml``).
    """
    t = forward(joints_rad)
    tcp = t.copy()
    tcp[:3, 3] = t[:3, 3] + t[:3, :3] @ TCP_OFFSET_LINK5_M
    return tcp


def forward_all(joints_rad: Sequence[float] | np.ndarray) -> list[np.ndarray]:
    """Poses of every link frame: ``[base, link1, ..., link5]`` (6 matrices)."""
    q = np.asarray(joints_rad, dtype=float).reshape(-1)
    if q.shape != (N_JOINTS,):
        raise ValueError(f"expected {N_JOINTS} joint values, got shape {q.shape}")
    frames = [np.eye(4)]
    t = np.eye(4)
    for i in range(N_JOINTS):
        t = t @ _joint_transform(i, q[i])
        frames.append(t.copy())
    return frames


# ---------------------------------------------------------------------------
# Inverse kinematics: damped least squares with joint-limit clamping
# ---------------------------------------------------------------------------


def _pose_error(target_t: np.ndarray, q: np.ndarray, orientation_weight: float):
    """Return (error_vector, position_error, rotation_error)."""
    t = forward(q)
    e_p = target_t[:3, 3] - t[:3, 3]
    e_r = rot_log(target_t[:3, :3] @ t[:3, :3].T)
    err = np.concatenate([e_p, orientation_weight * e_r])
    return err, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))


def _numeric_jacobian(q: np.ndarray, orientation_weight: float, eps: float = 1e-6) -> np.ndarray:
    """6x5 numerical Jacobian of the weighted pose error w.r.t. joints."""
    t0 = forward(q)
    jac = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        qp = q.copy()
        qp[i] += eps
        tp = forward(qp)
        jac[:3, i] = (tp[:3, 3] - t0[:3, 3]) / eps
        jac[3:, i] = rot_log(tp[:3, :3] @ t0[:3, :3].T) * orientation_weight / eps
    return jac


def inverse(
    pose: np.ndarray,
    seed: Sequence[float] | None = None,
    *,
    tol: float = 1e-4,
    rot_tol: float = 1e-3,
    max_iters: int = 400,
    damping: float = 1e-3,
    orientation_weight: float = 0.1,
    max_step_rad: float = 0.25,
) -> np.ndarray:
    """Solve ``forward(q) == pose`` for joint angles via damped least squares.

    Parameters
    ----------
    pose : (4, 4) array
        Target end-effector pose in the base frame.
    seed : sequence of 5 floats, optional
        Initial guess (default zeros). Clamped to the joint limits.
    tol : float
        Position convergence tolerance in metres (default 0.1 mm).
    rot_tol : float
        Orientation convergence tolerance in radians.
    max_iters : int
        Iteration cap.
    damping : float
        Initial Levenberg-Marquardt damping (adaptive during the solve).
    orientation_weight : float
        Weight applied to the orientation rows.  With a 5-DOF arm the full 6-D
        pose is not always achievable; lowering this (e.g. 0.01) makes position
        tracking dominate, which is what the grasp pipeline wants.
    max_step_rad : float
        Per-iteration joint step clamp, for stability near singularities.

    Returns
    -------
    np.ndarray, shape (5,)
        Best joint solution, always inside the URDF joint limits.

    Notes
    -----
    The solver is a damped Gauss-Newton (Levenberg-Marquardt) iteration::

        dq = J^T (J J^T + lambda^2 I)^-1 e

    with ``J`` the numerical 6x5 pose Jacobian, ``e`` the weighted pose error
    and adaptive ``lambda`` (grows when a step increases the error).  Every
    candidate is clamped to ``JOINT_LIMITS`` so the returned joints are always
    commandable.
    """
    target = np.asarray(pose, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"pose must be 4x4, got {target.shape}")
    q = clamp_joints(np.zeros(N_JOINTS) if seed is None else seed)
    lam = float(damping)
    best_q = q.copy()
    best_err = np.inf
    for _ in range(int(max_iters)):
        err, pos_err, rot_err = _pose_error(target, q, orientation_weight)
        err_norm = float(np.linalg.norm(err))
        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()
        if pos_err < tol and rot_err < rot_tol:
            return q
        jac = _numeric_jacobian(q, orientation_weight)
        try:
            lhs = jac @ jac.T + (lam**2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(lhs, err)
        except np.linalg.LinAlgError:  # pragma: no cover - extremely defensive
            dq = jac.T @ err
        step = float(np.max(np.abs(dq))) if dq.size else 0.0
        if step > max_step_rad:
            dq *= max_step_rad / step
        q_new = clamp_joints(q + dq)
        _, new_pos, new_rot = _pose_error(target, q_new, orientation_weight)
        new_norm = new_pos**2 + (orientation_weight * new_rot) ** 2
        if np.isfinite(new_norm) and new_norm < pos_err**2 + (orientation_weight * rot_err) ** 2:
            q = q_new
            lam = max(lam / 3.0, 1e-6)
        else:
            q = q_new
            lam = min(lam * 10.0, 1e3)
    return best_q


def inverse_position(
    position: Sequence[float] | np.ndarray,
    seed: Sequence[float] | np.ndarray | None = None,
    *,
    tol: float = 1e-4,
    max_iters: int = 300,
    damping: float = 1e-3,
    max_step_rad: float = 0.25,
    restarts: int = 12,
) -> np.ndarray:
    """Solve ``forward(q)[:3, 3] == position`` (position-only IK).

    The 5-DOF arm has more freedom than a 3-D position constraint, so a
    position-only damped least squares solve converges much more reliably than
    :func:`inverse` when the full orientation cannot be achieved (which is the
    normal case for a wrist-mounted grasp).  Joints are clamped to the URDF
    limits on every iteration.
    """
    target = np.asarray(position, dtype=float).reshape(-1)
    if target.shape != (3,):
        raise ValueError(f"position must have 3 elements, got shape {target.shape}")
    base_seed = clamp_joints(np.zeros(N_JOINTS) if seed is None else seed)
    # Deterministic restarts make the solver robust against local minima when
    # the seed is far from a solution.  The pipeline itself always seeds with
    # the current joints (already close), so the common path returns on the
    # first attempt; far seeds explore base-yaw candidates derived from the
    # target azimuth plus a small grid of pitch configurations.
    azimuth = float(np.arctan2(target[1], target[0]))
    q1_candidates = [
        float(np.clip(azimuth - np.pi / 2.0, *JOINT_LIMITS[0])),
        float(np.clip(azimuth + np.pi / 2.0, *JOINT_LIMITS[0])),
        float(base_seed[0]),
        0.0,
    ]
    pitch_grid = [
        (0.0, 0.0, 0.0),
        (0.6, 0.6, 0.6),
        (-0.6, -0.6, -0.6),
        (1.2, 1.2, 1.2),
        (-1.2, -1.2, -1.2),
        (1.2, 0.0, 0.6),
        (-1.2, 0.0, -0.6),
        (0.0, 0.9, 0.9),
        (0.0, -0.9, -0.9),
        (1.2, -0.6, 0.6),
        (-1.2, 0.6, -0.6),
    ]
    restart_seeds = [base_seed, np.zeros(N_JOINTS)]
    for q1 in q1_candidates:
        for pitch in pitch_grid:
            restart_seeds.append(np.array([q1, pitch[0], pitch[1], pitch[2], 0.0]))
    best_q = base_seed.copy()
    best_err = np.inf
    attempts = min(len(restart_seeds), max(1, int(restarts) + 1))
    for attempt in range(attempts):
        q, err = _position_solve(
            target, restart_seeds[attempt], tol, max_iters, damping, max_step_rad
        )
        if err < best_err:
            best_err = err
            best_q = q
        if err < tol:
            return q
    return best_q


def _position_solve(
    target: np.ndarray,
    q: np.ndarray,
    tol: float,
    max_iters: int,
    damping: float,
    max_step_rad: float,
) -> tuple[np.ndarray, float]:
    """One damped-least-squares run for a 3-D position target."""
    q = clamp_joints(q)
    lam = float(damping)
    best_q = q.copy()
    best_err = np.inf
    for _ in range(int(max_iters)):
        p = forward(q)[:3, 3]
        err = target - p
        err_norm = float(np.linalg.norm(err))
        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()
        if err_norm < tol:
            return q, err_norm
        jac = np.zeros((3, N_JOINTS))
        eps = 1e-6
        for i in range(N_JOINTS):
            qp = q.copy()
            qp[i] += eps
            jac[:, i] = (forward(qp)[:3, 3] - p) / eps
        try:
            lhs = jac @ jac.T + (lam**2) * np.eye(3)
            dq = jac.T @ np.linalg.solve(lhs, err)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            dq = jac.T @ err
        step = float(np.max(np.abs(dq))) if dq.size else 0.0
        if step > max_step_rad:
            dq *= max_step_rad / step
        q_new = clamp_joints(q + dq)
        new_err = float(np.linalg.norm(target - forward(q_new)[:3, 3]))
        if np.isfinite(new_err) and new_err < err_norm:
            q = q_new
            lam = max(lam / 3.0, 1e-6)
        else:
            q = q_new
            lam = min(lam * 10.0, 1e3)
    return best_q, best_err


# ---------------------------------------------------------------------------
# Convenience poses used by the pipeline / tests
# ---------------------------------------------------------------------------

HOME_JOINTS_RAD = np.zeros(N_JOINTS)
"""All-zero joints: the arm stands straight up (EE at ~z = 0.347 m)."""
