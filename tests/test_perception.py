"""Classical perception tests on synthetic NumPy-drawn frames (no cv2 needed)."""

from __future__ import annotations

import numpy as np
import pytest

from dofbot_grasp.camera import Intrinsics
from dofbot_grasp.perception import (
    DetectConfig,
    bgr_to_hsv,
    convex_hull,
    cube_pose_estimate,
    detect_cube,
    min_area_rect,
)

F_PX = 600.0
CUBE_M = 0.030


def synth_frame(
    side_px: float,
    center=(320.0, 240.0),
    size=(480, 640),
    bg=(35, 35, 35),
    color=(0, 128, 255),
) -> np.ndarray:
    """Draw an axis-aligned square (the face of the cube) on a plain table."""
    h, w = size
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :] = bg
    half = side_px / 2.0
    x0 = max(0, int(round(center[0] - half)))
    x1 = min(w, int(round(center[0] + half)))
    y0 = max(0, int(round(center[1] - half)))
    y1 = min(h, int(round(center[1] + half)))
    img[y0:y1, x0:x1] = color
    return img


@pytest.mark.parametrize("distance", [0.20, 0.35, 0.50])
def test_distance_error_below_10_percent(distance: float) -> None:
    side = F_PX * CUBE_M / distance
    frame = synth_frame(side)
    obs = detect_cube(frame, cube_size_m=CUBE_M, cfg=DetectConfig(f_px=F_PX))
    assert obs is not None
    error = abs(obs.distance_m - distance) / distance
    assert error < 0.10, f"distance error {error * 100:.1f}% >= 10%"
    assert obs.width_px == pytest.approx(side, rel=0.08)
    assert obs.bearing_rad == pytest.approx(0.0, abs=0.01)
    assert obs.bbox[2] > 0 and obs.bbox[3] > 0
    np.testing.assert_allclose(obs.center_px, (320.0, 240.0), atol=2.0)


def test_bearing_sign_and_center_offset() -> None:
    side = F_PX * CUBE_M / 0.30
    frame = synth_frame(side, center=(420.0, 180.0))
    obs = detect_cube(frame, cfg=DetectConfig(f_px=F_PX))
    assert obs is not None
    assert obs.center_px[0] == pytest.approx(420.0, abs=2.0)
    assert obs.center_px[1] == pytest.approx(180.0, abs=2.0)
    assert obs.bearing_rad > 0.0
    assert obs.bearing_rad == pytest.approx(np.arctan2(420.0 - 320.0, F_PX), abs=0.01)


def test_no_detection_on_blank_frame() -> None:
    frame = np.full((480, 640, 3), 35, dtype=np.uint8)
    assert detect_cube(frame, cfg=DetectConfig(f_px=F_PX)) is None


def test_no_detection_on_tiny_speck() -> None:
    frame = synth_frame(8.0)
    assert detect_cube(frame, cfg=DetectConfig(f_px=F_PX, min_area_px=100)) is None


def test_adaptive_fallback_bright_cube_on_dark_table() -> None:
    frame = synth_frame(F_PX * CUBE_M / 0.30, bg=(30, 30, 30), color=(200, 200, 200))
    obs = detect_cube(frame, cfg=DetectConfig(f_px=F_PX))
    assert obs is not None
    assert abs(obs.distance_m - 0.30) / 0.30 < 0.10


def test_adaptive_fallback_dark_cube_on_bright_table() -> None:
    frame = synth_frame(F_PX * CUBE_M / 0.30, bg=(205, 205, 205), color=(40, 40, 40))
    obs = detect_cube(frame, cfg=DetectConfig(f_px=F_PX))
    assert obs is not None
    assert abs(obs.distance_m - 0.30) / 0.30 < 0.10


def test_custom_hsv_colour() -> None:
    # saturated green cube with a green HSV range
    frame = synth_frame(F_PX * CUBE_M / 0.35, bg=(30, 30, 30), color=(0, 180, 0))
    cfg = DetectConfig(f_px=F_PX, hsv_lower=(40, 80, 60), hsv_upper=(80, 255, 255))
    obs = detect_cube(frame, cfg=cfg)
    assert obs is not None


def test_distance_uses_intrinsics_fx() -> None:
    distance = 0.4
    other_f = 900.0
    side = other_f * CUBE_M / distance
    frame = synth_frame(side, center=(640.0, 360.0), size=(720, 1280))
    intr = Intrinsics(fx=other_f, fy=other_f, cx=640.0, cy=360.0, width=1280, height=720)
    obs = detect_cube(frame, cube_size_m=CUBE_M, intrinsics=intr)
    assert obs is not None
    assert abs(obs.distance_m - distance) / distance < 0.10
    assert obs.center_px[0] == pytest.approx(640.0, abs=2.0)


def test_cube_pose_estimate_recovers_camera_frame_point() -> None:
    intr = Intrinsics(fx=F_PX, fy=F_PX, cx=320.0, cy=240.0, width=640, height=480)
    point_cam = np.array([0.02, -0.015, 0.30])
    u = intr.fx * point_cam[0] / point_cam[2] + intr.cx
    v = intr.fy * point_cam[1] / point_cam[2] + intr.cy
    side = intr.fx * CUBE_M / point_cam[2]
    frame = synth_frame(side, center=(u, v))
    obs = detect_cube(frame, cube_size_m=CUBE_M, intrinsics=intr)
    assert obs is not None
    pose = cube_pose_estimate(obs, intr)
    assert pose.shape == (4, 4)
    np.testing.assert_allclose(pose[:3, 3], point_cam, atol=0.01)
    np.testing.assert_allclose(pose[3], [0, 0, 0, 1])
    # mapping through a camera_to_base transform shifts the position accordingly
    t_base_cam = np.eye(4)
    t_base_cam[:3, 3] = [0.1, -0.2, 0.3]
    pose_base = cube_pose_estimate(obs, intr, camera_to_base=t_base_cam)
    np.testing.assert_allclose(pose_base[:3, 3], [0.12, -0.215, 0.60], atol=0.01)


def test_hsv_conversion_conventions() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., :] = (0, 0, 255)  # pure red BGR
    hsv = bgr_to_hsv(bgr)
    assert hsv[0, 0, 0] == pytest.approx(0.0, abs=1.0)  # red hue = 0 in OpenCV
    assert hsv[0, 0, 1] == pytest.approx(255.0, abs=1.0)  # saturation
    assert hsv[0, 0, 2] == pytest.approx(255.0, abs=1.0)  # value


def test_min_area_rect_and_hull_on_rotated_square() -> None:
    # a 40x40 square rotated by 30 degrees: min-area rect recovers ~40x40
    side = 40.0
    theta = np.radians(30.0)
    base = np.array([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = base @ rot.T + np.array([100.0, 100.0])
    center, (w, h), angle = min_area_rect(points)
    true_center = rot @ np.array([side / 2.0, side / 2.0]) + np.array([100.0, 100.0])
    np.testing.assert_allclose(center, true_center, atol=0.5)
    assert w * h == pytest.approx(side * side, rel=0.02)
    hull = convex_hull(points)
    assert len(hull) == 4


def test_invalid_frame_raises() -> None:
    with pytest.raises(ValueError):
        detect_cube(np.zeros((10, 10), dtype=np.uint8))
