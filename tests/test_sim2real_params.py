"""The servo/joint mapping and the camera geometry the real run depends on.

These numbers decide where the arm goes and where it believes the cube is, so a silent sign
flip in any of them is only visible as "the grasp does not work".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dofbot_grasp import controller as C  # noqa: E402
from dofbot_grasp.params import GRASP, JOINTS, MOUNT, joint_to_servo, servo_to_joint  # noqa: E402

ZERO = {1: 90.0, 2: 90.0, 3: 90.0, 4: 90.0, 5: 135.0}


def test_zero_pose_mapping_round_trips():
    """u = [90, 90, 90, 90, 135] is the vertical pose, and it maps to q = 0."""
    q = servo_to_joint(ZERO)
    assert q == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-12)
    assert joint_to_servo(q) == pytest.approx(ZERO, abs=1e-9)


def test_joint_limits_match_the_bus():
    """j1..j4 cover a 180 deg servo, j5 a 270 deg one."""
    for i in (1, 2, 3, 4, 5):
        assert JOINTS.servo_range_deg[i] - 2 * JOINTS.offsets_deg[i] == pytest.approx(0.0)


def test_gripper_travel_closes_on_the_cube():
    """The jaw slide must be enough to reach the cube's faces."""
    inner_open = 0.028 - 0.003  # finger centre minus half thickness
    inner_shut = inner_open - GRASP.sim_jaw_travel_m
    assert inner_shut < GRASP.cube_edge_m / 2  # it must close past the face to grip it


def test_camera_sits_ahead_of_link5():
    """The mount is registered to the camera modelled in the mesh: ahead of link5."""
    offset = C.cam_in_link5()
    assert np.linalg.norm(offset) * 1000 == pytest.approx(60.0, abs=20.0)
    assert offset[2] < 0.0  # it hangs slightly behind, as the STL shows


def test_backprojection_round_trips_a_table_point():
    """A table point projected through the model comes back to itself."""
    intr = json.loads((ROOT / "calib" / "intrinsics_wrist.json").read_text())
    pose = {1: 80.3, 2: 96.5, 3: 2.0, 4: 14.2, 5: 134.8}
    target = np.array([0.10, 0.10, GRASP.table_z_m])

    C_cam, R_cam = C.camera_pose(pose)
    p_cam = R_cam.T @ (target - C_cam)
    depth = -p_cam[2]  # MuJoCo cameras look along -z
    if depth <= 0:
        pytest.skip("test point is behind the camera at this pose")
    uv = np.array(
        [intr["cx"] + intr["fy"] * p_cam[0] / depth, intr["cy"] - intr["fy"] * p_cam[1] / depth]
    )

    est = C.backproject_to_table(float(uv[0]), float(uv[1]), pose, intr, GRASP.table_z_m)
    assert est is not None
    assert est[:2] == pytest.approx(target[:2], abs=0.03)


def test_fingertip_convention_matches_the_fingers():
    """The offset is measured from link5 along the tool axis, not from the TCP.

    Palm 0..40 mm, fingers 40..80 mm, TCP at 50 mm: the fingertip is 80 mm out from link5,
    which is 30 mm beyond the TCP rather than below it.
    """
    assert JOINTS.fingertip_below_tcp_m == pytest.approx(0.080, abs=1e-9)


def test_camera_roll_is_recorded():
    """The roll is a measured quantity, not a placeholder."""
    assert MOUNT.roll_deg is not None
    assert -180.0 <= MOUNT.roll_deg <= 180.0


def test_projection_handles_batches_and_the_mujoco_convention():
    """The real backend projects whole cubes, and in MuJoCo's -z convention.

    Both faults were invisible without an arm attached: single-point-only raised "size 8 is
    different from 3" from inside the fit, and an OpenCV-convention projection would have
    mirrored every pixel about the image centre.
    """
    from dofbot_grasp.controller import project_points

    intr = json.loads((ROOT / "calib" / "intrinsics_wrist.json").read_text())
    C = np.array([0.0, 0.15, 0.15])
    R = np.eye(3)  # looking down -z, straight at the table
    points = np.array(
        [
            [0.0, 0.15, 0.0],  # straight below the camera
            [0.03, 0.15, 0.0],  # to +x, which must appear to the right
            [0.0, 0.12, 0.0],  # to -y, which must appear lower
        ]
    )
    uv = project_points(C, R, intr, points)
    assert uv.shape == (3, 2)
    # centre
    assert uv[0, 0] == pytest.approx(intr["cx"], abs=6.0)
    assert uv[0, 1] == pytest.approx(intr["cy"], abs=6.0)
    # +x is to the right of centre, not mirrored to the left
    assert uv[1, 0] > uv[0, 0] + 20
    # -y appears below centre (image y grows downward)
    assert uv[2, 1] > uv[0, 1] + 20


def test_projection_marks_points_behind_the_camera():
    from dofbot_grasp.controller import project_points

    intr = json.loads((ROOT / "calib" / "intrinsics_wrist.json").read_text())
    uv = project_points(np.zeros(3), np.eye(3), intr, np.array([[0.0, 0.0, 0.5]]))
    assert np.isnan(uv[0]).all()
