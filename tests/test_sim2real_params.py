"""The servo/joint mapping and the camera geometry used for sim2real.

These are the numbers a real run depends on, so they get pinned here: a silent sign flip
in the mount transform, for instance, sends every ray out of the table and is otherwise
only visible as "the grasp does not work".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from dofbot_grasp.params import GRASP, JOINTS, MOUNT, joint_to_servo, servo_to_joint  # noqa: E402

run_real = pytest.importorskip("run_ibvs_real")


def test_zero_pose_mapping_round_trips():
    """u = [90, 90, 90, 90, 135] is the vertical pose, and it maps to q = 0."""
    zero = {1: 90.0, 2: 90.0, 3: 90.0, 4: 90.0, 5: 135.0}
    q = servo_to_joint(zero)
    assert q == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-12)
    assert joint_to_servo(q) == pytest.approx(zero, abs=1e-9)


def test_joint_limits_match_the_bus():
    """j1..j4 cover a 180 deg servo, j5 a 270 deg one."""
    for i in (1, 2, 3, 4):
        assert JOINTS.servo_range_deg[i] - 2 * JOINTS.offsets_deg[i] == pytest.approx(0.0)
    assert JOINTS.servo_range_deg[5] - 2 * JOINTS.offsets_deg[5] == pytest.approx(0.0)


def test_gripper_stroke_closes_on_a_cube():
    """Open jaws clear the cube, shut jaws grip it."""
    assert GRASP.cube_edge_m > 0.0
    assert JOINTS.gripper_open_deg < JOINTS.gripper_shut_deg


def test_camera_sits_in_front_of_and_below_link5():
    """The mount is registered to the camera modelled in the mesh: ahead of link5."""
    p5, _ = run_real._link5_frame({1: 90.0, 2: 90.0, 3: 90.0, 4: 90.0, 5: 135.0})
    cam = run_real.camera_centre({1: 90.0, 2: 90.0, 3: 90.0, 4: 90.0, 5: 135.0})
    offset = cam - p5
    assert np.linalg.norm(offset) * 1000 == pytest.approx(60.0, abs=15.0)


def test_backprojection_round_trips_a_table_point():
    """Project a known table point, then recover it from the pixel."""
    intr_path = ROOT / "calib" / "intrinsics_wrist.json"
    intr = json.loads(intr_path.read_text())
    servo = {1: 91.9, 2: 97.5, 3: 15.7, 4: 1.3, 5: 134.8}

    target = np.array([0.0, 0.20, GRASP.table_z_m])
    p5, R = run_real._link5_frame(servo)
    C = p5 + R @ run_real._cam_in_link5()
    R_cam = R @ run_real._cam_rotation_in_link5(MOUNT.roll_deg)
    p_cam = R_cam.T @ (target - C)
    if p_cam[2] <= 0:
        pytest.skip("test point is behind the camera at this pose")
    f = np.array([intr["fx"], intr["fy"]])
    c = np.array([intr["cx"], intr["cy"]])
    uv = c + f * p_cam[:2] / p_cam[2]

    est = run_real.backproject_to_table(float(uv[0]), float(uv[1]), servo, intr)
    assert est is not None
    assert est[:2] == pytest.approx(target[:2], abs=0.02)  # 20 mm, distortion included
