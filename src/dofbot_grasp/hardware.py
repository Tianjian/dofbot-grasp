"""One interface, two backends: the MuJoCo cell and the real arm.

The grasp controller talks only to :class:`RobotIO`, so the same phase code runs in
simulation and on hardware. What differs between them is exactly what you would expect:
how a frame is captured, and how a joint command is delivered (a ramped ``ctrl`` in
simulation, a guarded ``write6`` frame on the bus).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .params import BUS, GRASP, JOINTS


class RobotIO(Protocol):
    """Everything the controller needs from a robot."""

    def read_joints(self) -> dict[int, float]:
        """Servo angles in degrees, ids 1..5."""

    def command(self, target: dict[int, float], gripper: float | None = None) -> dict[int, float]:
        """Move to ``target`` (servo degrees) and return the settled read-back."""

    def capture_wrist(self) -> np.ndarray | None:
        """One RGB frame from the wrist camera."""

    def capture_observer(self) -> np.ndarray | None:
        """One RGB frame from the fixed observer camera, for safety checks."""

    def project(self, point_m: np.ndarray) -> tuple[float, float] | None:
        """Base-frame point -> pixel in the wrist frame (None if behind the camera)."""

    def unproject_to_table(self, u: float, v: float, z_plane: float) -> np.ndarray | None:
        """Wrist pixel -> base-frame point on the horizontal plane ``z_plane``."""


class SimRobot:
    """MuJoCo cell: the model in ``sim/dofbot_se.xml``.

    ``command`` ramps ``data.ctrl`` over the commanded move time instead of jumping, so
    the simulated arm settles in 1.5-2 s like the real one rather than in 0.2 s. That
    difference is large enough to change how the controller behaves.
    """

    def __init__(
        self,
        model,
        data,
        wrist_camera="wrist_cam",
        observer_camera="overview",
        img_w=320,
        img_h=240,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(model, height=img_h, width=img_w)
        self.wrist_camera = wrist_camera
        self.observer_camera = observer_camera
        self.img_w = img_w
        self.img_h = img_h

    # -- joints ---------------------------------------------------------
    def _addr(self, name: str) -> int:
        return self.model.jnt_qposadr[
            self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
        ]

    def read_joints(self) -> dict[int, float]:
        out = {}
        for i in range(1, 6):
            q = float(self.data.qpos[self._addr(f"joint{i}")])
            out[i] = float(np.degrees(q) + JOINTS.offsets_deg[i])
        return out

    def command(self, target: dict[int, float], gripper: float | None = None) -> dict[int, float]:
        """Ramp ``ctrl`` to the target over the commanded move time, then settle.

        Teleporting the joints would settle the arm in 0.2 s against the hardware's
        1.5-2 s, and that difference is enough to change what the controller does.
        """
        dt = float(self.model.opt.timestep)
        ramp = max(1, int(BUS.move_time_s / dt))
        start = self.data.ctrl.copy()
        goal = np.array(
            [target[i] - JOINTS.offsets_deg[i] for i in range(1, 6)]
            + [gripper if gripper is not None else GRASP.deadband_px * 0.0]
        )
        goal[:5] = np.radians(goal[:5])
        if gripper is None:
            goal[5] = start[5]
        for k in range(ramp):
            a = (k + 1) / ramp
            self.data.ctrl[:] = start + (goal - start) * a
            self._mj.mj_step(self.model, self.data)
        for _ in range(int(BUS.settle_timeout_s / dt * 0.25)):
            self.data.ctrl[:] = goal
            self._mj.mj_step(self.model, self.data)
        return self.read_joints()

    # -- vision ---------------------------------------------------------
    def _render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def capture_wrist(self) -> np.ndarray | None:
        return self._render(self.wrist_camera)

    def capture_observer(self) -> np.ndarray | None:
        return self._render(self.observer_camera)

    # -- camera model ---------------------------------------------------
    # The projection lives with the backend: MuJoCo and OpenCV disagree about which way
    # the optical axis points (+z vs -z), and encoding that once in the controller caused
    # a roll mismatch that made every target pixel wrong.
    def _cam(self):
        cid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_CAMERA, self.wrist_camera)
        return self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy()

    def _focal(self) -> float:
        cid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_CAMERA, self.wrist_camera)
        fovy = float(np.radians(self.model.cam_fovy[cid]))
        return (self.img_h / 2.0) / np.tan(fovy / 2.0)

    def project(self, point_m: np.ndarray) -> tuple[float, float] | None:
        C, R = self._cam()
        local = R.T @ (np.asarray(point_m, float) - C)
        depth = -local[2]  # MuJoCo cameras look along -z
        if depth <= 1e-6:
            return None
        f = self._focal()
        return float(self.img_w / 2.0 + f * local[0] / depth), float(
            self.img_h / 2.0 - f * local[1] / depth
        )

    def unproject_to_table(self, u: float, v: float, z_plane: float) -> np.ndarray | None:
        C, R = self._cam()
        f = self._focal()
        ray = R @ np.array([(u - self.img_w / 2.0) / f, -(v - self.img_h / 2.0) / f, -1.0])
        if abs(ray[2]) < 1e-9:
            return None
        t = (z_plane - C[2]) / ray[2]
        return None if t <= 0 else C + t * ray


class SerialRobot:
    """The real arm over the bus, with the timing rules from ``docs/hardware-notes.md``."""

    def __init__(
        self,
        port: str = BUS.port,
        baud: int = BUS.baud,
        wrist_device: str | None = None,
        observer_device: str | None = None,
    ) -> None:
        from .params import CAMERAS

        self.arm = _make_arm(port, baud)
        self.wrist_device = wrist_device or CAMERAS.wrist_device
        self.observer_device = observer_device or CAMERAS.observer_device

    def read_joints(self) -> dict[int, float]:
        return self.arm.pose()

    def command(self, target: dict[int, float], gripper: float | None = None) -> dict[int, float]:
        return self.arm.move(target, gripper=gripper)

    def capture(self, device: str) -> np.ndarray | None:
        import cv2

        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def capture_wrist(self) -> np.ndarray | None:
        return self.capture(self.wrist_device)

    def capture_observer(self) -> np.ndarray | None:
        return self.capture(self.observer_device)

    # -- camera model ---------------------------------------------------
    def _intrinsics(self) -> dict:
        import json
        from pathlib import Path

        from .params import CAMERAS

        return json.loads(
            (Path(__file__).resolve().parents[2] / CAMERAS.wrist_intrinsics).read_text()
        )

    def _camera(self):
        """Camera centre and camera->base rotation, from FK plus the registered mount."""
        import importlib

        ctrl = importlib.import_module("dofbot_grasp.controller")
        return ctrl.camera_pose(self.read_joints())

    def project(self, point_m: np.ndarray) -> tuple[float, float] | None:
        import cv2

        C, R = self._camera()
        local = R.T @ (np.asarray(point_m, float) - C)
        if local[2] <= 1e-6:
            return None
        intr = self._intrinsics()
        Kmtx = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
        dist = np.array(intr["dist"], float).reshape(-1, 1)
        uv, _ = cv2.projectPoints(
            np.asarray(point_m, float).reshape(1, 3),
            cv2.Rodrigues(R.T)[0],
            (-R.T @ C).reshape(3, 1),
            Kmtx,
            dist,
        )
        return float(uv[0, 0]), float(uv[0, 1])

    def unproject_to_table(self, u: float, v: float, z_plane: float) -> np.ndarray | None:
        import importlib

        ctrl = importlib.import_module("dofbot_grasp.controller")
        return ctrl.backproject_to_table(u, v, self.read_joints(), self._intrinsics(), z_plane)

    def close(self) -> None:
        self.arm.close()


def _make_arm(port: str, baud: int):
    """Import the guarded bus wrapper lazily so the sim path needs no serial port."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "run_ibvs_real.py"
    spec = importlib.util.spec_from_file_location("_real_arm", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the serial wrapper from {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Arm(port=port, baud=baud)


def grasp_ready(io: RobotIO) -> bool:
    """Cheap pre-flight: the arm is inside the servo travel and the cube is reachable."""
    pose = io.read_joints()
    if any(not (0.0 < pose[i] < JOINTS.servo_range_deg[i]) for i in range(1, 6)):
        return False
    return GRASP.deadband_px > 0
