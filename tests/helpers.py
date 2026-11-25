"""Shared test doubles: a synthetic cube scene and a camera backed by it.

The mock scene is deliberately self-consistent with the real geometry:

* the cube is a fixed point in the robot base frame;
* the camera sits at the link5 origin and looks along +X of link5;
* a frame is rendered by projecting the cube with the pinhole model from the
  *current* joint state of the mock :class:`~dofbot_grasp.bus.SimBus`.

That means the pipeline closes its loop through exactly the same math a real
wrist camera would, instead of playing back a canned observation.
"""

from __future__ import annotations

import numpy as np

from dofbot_grasp.bus import SimBus
from dofbot_grasp.camera import Intrinsics, wrist_camera_to_ee
from dofbot_grasp.kinematics import forward
from dofbot_grasp.pipeline import PipelineConfig


def default_scene_cube(
    cfg: PipelineConfig | None = None, table_cube_z: float = 0.015
) -> np.ndarray:
    """A 30 mm cube sitting on the table, centred on the home camera ray.

    Uses the contract camera mount (35 mm beside / 10 mm along the tool axis)
    and intersects the home optical axis with the cube-centre plane
    ``z = table_cube_z`` (15 mm = half a 30 mm cube on a table at z = 0).
    """
    cfg = cfg or PipelineConfig()
    home = cfg.home_joints_rad()
    t_base_ee = forward(home)
    camera_pos = t_base_ee[:3, 3] + t_base_ee[:3, :3] @ np.array([0.035, 0.0, 0.010])
    optical_axis = t_base_ee[:3, 2]
    depth = (float(table_cube_z) - camera_pos[2]) / optical_axis[2]
    return camera_pos + depth * optical_axis


class CubeScene:
    """A static 30 mm cube rendered from the current arm joints."""

    def __init__(
        self,
        cube_world=None,
        intrinsics: Intrinsics | None = None,
        camera_to_ee: np.ndarray | None = None,
        cube_size_m: float = 0.030,
        bg_bgr=(35, 35, 35),
        cube_bgr=(0, 128, 255),
        noise: int = 0,
    ):
        self.cube_world = None if cube_world is None else np.asarray(cube_world, dtype=float)
        self.intrinsics = intrinsics or Intrinsics(
            fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480
        )
        self.camera_to_ee = (
            wrist_camera_to_ee() if camera_to_ee is None else np.asarray(camera_to_ee, dtype=float)
        )
        self.cube_size_m = float(cube_size_m)
        self.bg_bgr = tuple(int(v) for v in bg_bgr)
        self.cube_bgr = tuple(int(v) for v in cube_bgr)
        self.noise = int(noise)
        self._rng = np.random.default_rng(1234)
        self.frames_rendered = 0

    # -- geometry ---------------------------------------------------------
    def cube_camera_point(self, joints_rad) -> np.ndarray | None:
        if self.cube_world is None:
            return None
        t_base_ee = forward(joints_rad)
        t_base_cam = t_base_ee @ self.camera_to_ee
        p_cam = np.linalg.inv(t_base_cam) @ np.append(self.cube_world, 1.0)
        if p_cam[2] <= 0.02:
            return None
        return p_cam[:3]

    def project_cube(self, joints_rad):
        """Return ``(u, v, side_px)`` of the cube, or ``None`` if not visible."""
        p_cam = self.cube_camera_point(joints_rad)
        if p_cam is None:
            return None
        z = p_cam[2]
        u = self.intrinsics.fx * p_cam[0] / z + self.intrinsics.cx
        v = self.intrinsics.fy * p_cam[1] / z + self.intrinsics.cy
        side = self.intrinsics.fx * self.cube_size_m / z
        h, w = self.intrinsics.height, self.intrinsics.width
        if side < 2 or side > min(h, w) * 0.98:
            return None
        if -side < u - w / 2 and u + w / 2 < 2 * w and -side < v < 2 * h:
            return float(u), float(v), float(side)
        return None

    # -- rendering --------------------------------------------------------
    def render(self, joints_rad) -> np.ndarray:
        self.frames_rendered += 1
        img = np.empty((self.intrinsics.height, self.intrinsics.width, 3), dtype=np.uint8)
        img[:, :] = self.bg_bgr
        projected = self.project_cube(joints_rad)
        if projected is not None:
            u, v, side = projected
            half = side / 2.0
            x0 = max(0, int(round(u - half)))
            x1 = min(img.shape[1], int(round(u + half)))
            y0 = max(0, int(round(v - half)))
            y1 = min(img.shape[0], int(round(v + half)))
            if x1 > x0 and y1 > y0:
                img[y0:y1, x0:x1] = self.cube_bgr
        if self.noise > 0:
            noise = self._rng.integers(-self.noise, self.noise + 1, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img


class SceneCamera:
    """Camera that renders :class:`CubeScene` from the bus's current joints."""

    def __init__(self, scene: CubeScene, bus: SimBus):
        self.scene = scene
        self.bus = bus

    @property
    def intrinsics(self) -> Intrinsics:
        return self.scene.intrinsics

    def read(self) -> np.ndarray:
        return self.scene.render(self.bus.read_joints())


_DEFAULT_CUBE = object()


def make_system(
    cube_world=_DEFAULT_CUBE,
    cfg: PipelineConfig | None = None,
    noise: int = 0,
    **scene_kwargs,
):
    """Return ``(scene, bus, camera, cfg)`` for pipeline tests.

    ``cube_world`` omitted -> cube placed along the home camera axis;
    ``cube_world=None`` -> empty scene (no cube).
    """
    cfg = cfg or PipelineConfig()
    if cube_world is _DEFAULT_CUBE:
        cube = default_scene_cube(cfg)
    else:
        cube = None if cube_world is None else np.asarray(cube_world, dtype=float)
    bus = SimBus()
    scene = CubeScene(cube_world=cube, noise=noise, **scene_kwargs)
    camera = SceneCamera(scene, bus)
    return scene, bus, camera, cfg
