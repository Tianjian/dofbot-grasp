"""Kinematics tests: FK golden poses, IK round-trip/clamping, servo mapping."""

from __future__ import annotations

import numpy as np
import pytest

from dofbot_grasp.kinematics import (
    JOINT_LIMITS,
    N_JOINTS,
    clamp_joints,
    clamp_servo_deg,
    forward,
    forward_all,
    inverse,
    inverse_position,
    joint_deg_to_raw_servo,
    joint_deg_to_servo,
    raw_servo_to_joint_deg,
    rot_log,
    rpy_to_matrix,
    servo_to_joint_deg,
)

# Golden values from an independent URDF-chain evaluation (metres).
HOME_P = np.array([0.00000088, 0.0011039, 0.34704362])
EXTENDED_P = np.array([0.0, 0.23967174, 0.10652627])  # q = [0, -pi/2, 0, 0, 0]
BASE90_P = np.array([0.00193224, 0.00303691, 0.34702273])  # q = [pi/2, 0, 0, 0, 0]


def assert_rotation(r: np.ndarray, atol: float = 1e-6) -> None:
    assert r.shape == (3, 3)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=atol)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=atol)


def test_forward_home_pose() -> None:
    t = forward(np.zeros(N_JOINTS))
    assert t.shape == (4, 4)
    np.testing.assert_allclose(t[:3, 3], HOME_P, atol=2e-4)
    # the arm stands straight up: all links stack along +Z
    assert t[2, 3] == pytest.approx(0.34704, abs=5e-4)
    np.testing.assert_allclose(t[:3, :3], np.eye(3), atol=3e-3)
    assert_rotation(t[:3, :3])


def test_forward_extended_pose() -> None:
    q = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0])
    t = forward(q)
    np.testing.assert_allclose(t[:3, 3], EXTENDED_P, atol=1e-4)
    # folded forward/sideways: the total link length (~0.347 m) projects onto
    # the plane, and the EE is no longer at full height
    assert np.linalg.norm(t[:3, 3]) < 0.30
    assert t[2, 3] < HOME_P[2]
    assert_rotation(t[:3, :3])


def test_forward_base_90_degrees() -> None:
    t = forward(np.array([np.pi / 2, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(t[:3, 3], BASE90_P, atol=1e-4)
    # the EE frame is yawed 90 deg: its +X axis now points along base +Y
    np.testing.assert_allclose(t[:3, 0], [0.0, 1.0, 0.0], atol=0.02)
    # and the arm still stands at (almost) the same height
    assert t[2, 3] == pytest.approx(HOME_P[2], abs=1e-3)
    assert_rotation(t[:3, :3])


def test_forward_all_returns_six_link_frames() -> None:
    frames = forward_all(np.array([0.1, -0.2, 0.3, -0.4, 0.5]))
    assert len(frames) == N_JOINTS + 1
    np.testing.assert_allclose(frames[0], np.eye(4))
    np.testing.assert_allclose(frames[-1], forward(np.array([0.1, -0.2, 0.3, -0.4, 0.5])))


def test_forward_validates_shape() -> None:
    with pytest.raises(ValueError):
        forward([0.0, 0.0])


def test_rotation_helpers_are_consistent() -> None:
    r = rpy_to_matrix(0.1, -0.2, 0.3)
    assert_rotation(r)
    v = rot_log(r)
    assert np.linalg.norm(v) == pytest.approx(0.374, abs=0.05)


def test_ik_round_trip_within_1mm() -> None:
    rng = np.random.default_rng(42)
    worst_pos = 0.0
    worst_rot = 0.0
    for _ in range(8):
        q_true = rng.uniform(JOINT_LIMITS[:, 0] + 0.2, JOINT_LIMITS[:, 1] - 0.25)
        target = forward(q_true)
        seed = q_true + rng.normal(0.0, 0.2, N_JOINTS)
        solution = inverse(target, seed)
        assert np.all(solution >= JOINT_LIMITS[:, 0] - 1e-9)
        assert np.all(solution <= JOINT_LIMITS[:, 1] + 1e-9)
        solved = forward(solution)
        pos_err = np.linalg.norm(solved[:3, 3] - target[:3, 3])
        rot_err = np.linalg.norm(rot_log(target[:3, :3] @ solved[:3, :3].T))
        worst_pos = max(worst_pos, pos_err)
        worst_rot = max(worst_rot, rot_err)
        assert pos_err < 1e-3, f"IK position error {pos_err * 1000:.3f} mm > 1 mm"
    assert worst_pos < 1e-3
    assert worst_rot < 5e-3


def test_ik_position_only_reaches_fk_points() -> None:
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(8):
        q_true = rng.uniform(JOINT_LIMITS[:, 0] + 0.3, JOINT_LIMITS[:, 1] - 0.35)
        target_position = forward(q_true)[:3, 3]
        seed = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        solution = inverse_position(target_position, seed)
        worst = max(worst, np.linalg.norm(forward(solution)[:3, 3] - target_position))
    assert worst < 1e-3, f"position-only IK error {worst * 1000:.3f} mm > 1 mm"


def test_ik_clamps_to_joint_limits_with_out_of_range_seed() -> None:
    q_true = np.array([0.4, -0.6, 0.9, 0.3, 0.7])
    solution = inverse(forward(q_true), [9.0, -9.0, 9.0, -9.0, 9.0])
    assert solution.shape == (N_JOINTS,)
    assert np.all(solution >= JOINT_LIMITS[:, 0] - 1e-9)
    assert np.all(solution <= JOINT_LIMITS[:, 1] + 1e-9)
    assert np.all(np.isfinite(solution))


def test_ik_unreachable_target_still_returns_valid_joints() -> None:
    target = np.eye(4)
    target[:3, 3] = [1.5, 0.0, 0.5]  # far outside the ~0.35 m workspace
    solution = inverse(target, np.zeros(N_JOINTS), max_iters=150)
    assert np.all(solution >= JOINT_LIMITS[:, 0] - 1e-9)
    assert np.all(solution <= JOINT_LIMITS[:, 1] + 1e-9)
    assert np.allclose(clamp_joints(solution), solution)


def test_clamp_joints() -> None:
    q = clamp_joints([3.0, -3.0, 0.0, 1.5708, 2.0])
    np.testing.assert_allclose(q, [1.5708, -1.5708, 0.0, 1.5708, 2.0])


def test_servo_mapping_mid_positions() -> None:
    # Arm_Lib mid-travel: servos 1-4 = 90 deg, servo 5 = 135 deg, gripper = 90
    np.testing.assert_allclose(
        servo_to_joint_deg([90, 90, 90, 90, 135, 90]), np.zeros(5), atol=1e-12
    )
    np.testing.assert_allclose(
        joint_deg_to_servo(np.zeros(5), gripper_deg=90), [90, 90, 90, 90, 135, 90]
    )


def test_servo_mapping_round_trip() -> None:
    rng = np.random.default_rng(3)
    joints = np.column_stack(
        [
            rng.uniform(-90.0, 90.0, size=20),
            rng.uniform(-90.0, 90.0, size=20),
            rng.uniform(-90.0, 90.0, size=20),
            rng.uniform(-90.0, 90.0, size=20),
            rng.uniform(-90.0, 135.0, size=20),
        ]
    )
    gripper = 45.0
    servo = joint_deg_to_servo(joints, gripper_deg=gripper)
    assert servo.shape == (20, 6)
    assert np.all(servo[:, 5] == gripper)
    np.testing.assert_allclose(servo_to_joint_deg(servo), joints, atol=1e-9)


def test_servo_mapping_s5_0_270() -> None:
    servo = joint_deg_to_servo([0, 0, 0, 0, 135], gripper_deg=90)
    assert servo[4] == pytest.approx(270.0)
    # ANGLE_MAPPING formula joint5 = servo5 - 135 (mid 135 deg -> joint 0)
    assert servo_to_joint_deg([90, 90, 90, 90, 0, 90])[4] == pytest.approx(-135.0)
    assert servo_to_joint_deg([90, 90, 90, 90, 270, 90])[4] == pytest.approx(135.0)
    # S5 range clamping
    clamped = clamp_servo_deg([200, 200, 200, 200, 300, 200])
    np.testing.assert_allclose(clamped, [180, 180, 180, 180, 270, 180])
    # joint5 = +180 deg (URDF upper limit) would need servo 315 > 270, so the
    # command saturates; only the servo-reachable part (-135..+135) round-trips.
    saturated = joint_deg_to_servo([0, 0, 0, 0, 180], gripper_deg=90)
    assert saturated[4] == pytest.approx(270.0)


def test_servo_reversal_of_ids_2_3_4() -> None:
    # logical servo angle 0..180; raw direction is inverted for IDs 2/3/4.
    raw = joint_deg_to_raw_servo([0.0, 0.0, 0.0, 0.0, 0.0])
    # joint 0 -> logical 90 -> raw 90 for every servo
    np.testing.assert_allclose(raw, [90, 90, 90, 90, 135])
    # joint2 = +90 -> logical 180 -> raw 0 (reversed)
    raw = joint_deg_to_raw_servo([0.0, 90.0, 0.0, 0.0, 0.0])
    assert raw[1] == pytest.approx(0.0)
    raw = joint_deg_to_raw_servo([0.0, -90.0, 0.0, 0.0, 0.0])
    assert raw[1] == pytest.approx(180.0)
    # and the inverse returns the joint angles
    for j in ([0, 90, -90, 0, 0], [10, -20, 30, -40, 50]):
        raw = joint_deg_to_raw_servo(j)
        np.testing.assert_allclose(raw_servo_to_joint_deg(raw), j, atol=1e-9)


def test_double_reversal_is_not_needed() -> None:
    # ANGLE_MAPPING.md's "common mistake": ID 2/3/4 are already un-reversed by
    # Arm_Lib's read; applying 90 - s would double-reverse.
    joints = servo_to_joint_deg([90, 90, 90, 90, 135, 90])
    np.testing.assert_allclose(joints, np.zeros(5))
