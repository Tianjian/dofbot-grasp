"""Camera abstraction tests (no OpenCV required)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dofbot_grasp.camera import (
    Intrinsics,
    SimCamera,
    UsbCamera,
    ee_to_wrist_camera,
    intrinsics_from_fovy,
    project_point,
    unproject_pixel,
    wrist_camera_to_ee,
)


def test_sim_camera_raises_until_wired() -> None:
    camera = SimCamera()
    with pytest.raises(RuntimeError, match="not wired"):
        camera.read()
    camera.wire(lambda: np.zeros((4, 4, 3), dtype=np.uint8))
    frame = camera.read()
    assert frame.shape == (4, 4, 3)


def test_sim_camera_accepts_camera_like_object() -> None:
    class Dummy:
        def read(self):
            return np.full((2, 2, 3), 7, dtype=np.uint8)

    camera = SimCamera(frame_fn=Dummy())
    np.testing.assert_array_equal(camera.read(), np.full((2, 2, 3), 7, dtype=np.uint8))


def test_intrinsics_scaling_and_matrix() -> None:
    intr = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    scaled = intr.scaled(1280, 960)
    assert scaled.fx == pytest.approx(1200.0)
    assert scaled.cx == pytest.approx(640.0)
    k = intr.as_matrix()
    np.testing.assert_allclose(k, [[600, 0, 320], [0, 600, 240], [0, 0, 1]])


def test_wrist_camera_transform_matches_sim_contract() -> None:
    t = wrist_camera_to_ee()
    r = t[:3, :3]
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
    # sim/dofbot_se.xml: camera at (0.035, 0, 0.010) looking down +Z of link5
    np.testing.assert_allclose(t[:3, 3], [0.035, 0.0, 0.010])
    np.testing.assert_allclose(r, np.eye(3))
    np.testing.assert_allclose(ee_to_wrist_camera(), np.linalg.inv(t))


def test_intrinsics_from_fovy() -> None:
    intr = intrinsics_from_fovy(75.0, 640, 480)
    assert intr.fx == pytest.approx(312.774, abs=1e-3)
    assert intr.cx == pytest.approx(320.0)
    assert intr.cy == pytest.approx(240.0)


def test_project_unproject_round_trip() -> None:
    intr = Intrinsics(fx=500.0, fy=520.0, cx=300.0, cy=200.0, width=640, height=480)
    point = np.array([0.03, -0.02, 0.4])
    uv = project_point(point, intr)
    recovered = unproject_pixel(uv[0], uv[1], point[2], intr)
    np.testing.assert_allclose(recovered, point, atol=1e-12)


def test_usb_camera_guarded_cv2_import() -> None:
    if importlib.util.find_spec("cv2") is not None:
        pytest.skip("OpenCV installed; cannot exercise the missing-dependency path")
    with pytest.raises(RuntimeError, match="OpenCV"):
        UsbCamera()
