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

    def read_gripper(self) -> float | None:
        """The jaws' own position, in degrees. Blocked by a cube, they stop short.

        This is the one grasp signal that needs no camera: the arm knows whether the jaws
        closed on something or on nothing, and at the grasp height the wrist view cannot see
        the cube at all.
        """

    def command(
        self, target: dict[int, float] | None, gripper: float | None = None
    ) -> dict[int, float]:
        """Move to ``target`` (servo degrees) and return the settled read-back.

        ``target=None`` means "leave the joints exactly as commanded and only change the
        gripper" -- that is what closing the jaws needs, and re-sending the read-back as a
        target instead nudges the arm by its own static error.

        ``gripper`` is a closure **fraction**, 0 = jaws apart, 1 = shut. Each backend maps
        it to its own units: the simulator drives a slide in metres, the arm a servo in
        degrees. Passing degrees to the simulator used to work only because ``ctrlrange``
        clipped the number.
        """

    def capture_wrist(self) -> np.ndarray | None:
        """One RGB frame from the wrist camera."""

    def capture_observer(self) -> np.ndarray | None:
        """One RGB frame from the fixed observer camera, for safety checks."""

    def project(self, point_m: np.ndarray, pose: dict[int, float] | None = None):
        """Base-frame point -> pixel in the wrist frame (None if behind the camera).

        ``pose`` asks for a kinematic query at another joint configuration -- used to build
        the image Jacobian without moving the arm.
        """

    def unproject_to_table(self, u: float, v: float, z_plane: float) -> np.ndarray | None:
        """Wrist pixel -> base-frame point on the horizontal plane ``z_plane``."""


class RealCameraLike(Protocol):
    """What ``sim/real_camera.py`` provides for resampling through real intrinsics."""

    width: int
    height: int

    def render_geometry(self) -> tuple[int, int, float]: ...

    def sample_offsets(self) -> tuple[int, int]: ...

    def apply(self, ideal_render: np.ndarray) -> np.ndarray: ...


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
        real_cameras: dict[str, RealCameraLike] | None = None,
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
        # Optional RealCamera per view. A MuJoCo camera is a symmetric pinhole with no
        # distortion; the real wrist lens has its principal point 50 px off centre and
        # k1 -0.50. Rendering through the measured intrinsics makes the simulated frame
        # have the same geometry as the real one, which is what a fair comparison needs.
        self.real_cameras = real_cameras or {}

    # -- joints ---------------------------------------------------------
    def _addr(self, name: str) -> int:
        return self.model.jnt_qposadr[
            self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
        ]

    def read_gripper(self) -> float | None:
        sides = []
        for name in ("gripper_left", "gripper_right"):
            try:
                sides.append(float(self.data.qpos[self._addr(name)]))
            except Exception:
                return None
        if not sides:
            return None
        return float(np.degrees(np.mean(sides)))

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
        goal = np.zeros(6)
        if target is None:
            goal[:5] = start[:5]  # joints frozen: only the jaw moves
        else:
            goal[:5] = np.radians([target[i] - JOINTS.offsets_deg[i] for i in range(1, 6)])
        # the simulated jaw is a slide; full closure is sim_jaw_travel() metres
        if gripper is not None and not 0.0 <= float(gripper) <= 1.0:
            raise ValueError(
                f"gripper must be a fraction 0..1, got {gripper}. Servo degrees belong in "
                f"the joint dict; passing JOINTS.gripper_open_deg here used to be clipped to "
                f"1.0, so every 'open' command closed the jaws."
            )
        goal[5] = start[5] if gripper is None else float(gripper) * sim_jaw_travel()
        for k in range(ramp):
            a = (k + 1) / ramp
            self.data.ctrl[:] = start + (goal - start) * a
            self._mj.mj_step(self.model, self.data)
        # settle until the joints stop moving, not for a fixed count: a folded pose under
        # gravity needed longer than the old 2.5 s and stopped 6 deg short of its command,
        # which put the fingertip tens of millimetres off target
        max_steps = int(BUS.settle_timeout_s / dt)
        prev = self.data.qpos.copy()
        still = 0
        for _ in range(max_steps):
            self.data.ctrl[:] = goal
            self._mj.mj_step(self.model, self.data)
            # Include the jaws. Settling on qpos[:5] returned while the fingers were still
            # closing, so "close then lift" was really "close and lift at the same time" -- and
            # a grasp that had not closed yet is a grasp that was never made.
            change = float(np.abs(self.data.qpos[:7] - prev[:7]).max())
            prev = self.data.qpos.copy()
            still = still + 1 if change < 1e-5 else 0
            if still >= 50:
                break
        return self.read_joints()

    # -- vision ---------------------------------------------------------
    def _render(self, camera: str) -> np.ndarray:
        """One frame in BGR, the convention every backend shares.

        MuJoCo renders RGB and OpenCV -- and therefore the real camera, the segmenter and the
        colour gate -- works in BGR. Converting here means one convention inside the shared
        controller instead of a per-backend surprise: the yellow gate silently failed in
        simulation for exactly that reason.
        """
        rc = self.real_cameras.get(camera)
        if rc is None:
            self.renderer.update_scene(self.data, camera=camera)
            return self.renderer.render()[:, :, ::-1].copy()
        width, height, fovy = rc.render_geometry()
        cid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_CAMERA, camera)
        saved_fovy = float(self.model.cam_fovy[cid])
        big = self._mj.Renderer(self.model, height=height, width=width)
        try:
            self.model.cam_fovy[cid] = fovy
            big.update_scene(self.data, camera=camera)
            frame = big.render().copy()
        finally:
            self.model.cam_fovy[cid] = saved_fovy
        ox, oy = rc.sample_offsets()
        ox = min(ox, max(0, width - rc.width))
        oy = min(oy, max(0, height - rc.height))
        cropped = frame[oy : oy + rc.height, ox : ox + rc.width]
        out = rc.apply(cropped) if cropped.shape[:2] == (rc.height, rc.width) else frame
        return out[:, :, ::-1].copy()

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

    def project(self, point_m: np.ndarray, pose: dict[int, float] | None = None):
        arr = np.asarray(point_m, float)
        if arr.ndim == 2:  # batch: (N, 3) -> (N, 2)
            out = [self.project(p, pose) for p in arr]
            if any(o is None for o in out):
                return None
            return np.array(out, float)
        if pose is not None:
            saved = self.data.qpos.copy()
            for i in range(1, 6):
                self.data.qpos[self._addr(f"joint{i}")] = np.radians(
                    pose[i] - JOINTS.offsets_deg[i]
                )
            self._mj.mj_forward(self.model, self.data)
            try:
                return self.project(point_m)
            finally:
                self.data.qpos[:] = saved
                self._mj.mj_forward(self.model, self.data)
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

    def read_gripper(self) -> float | None:
        try:
            return float(self.arm.read(6))
        except Exception:
            return None

    def read_joints(self) -> dict[int, float]:
        return self.arm.pose()

    def command(
        self, target: dict[int, float] | None, gripper: float | None = None
    ) -> dict[int, float]:
        deg = None
        if gripper is not None:
            if not 0.0 <= float(gripper) <= 1.0:
                raise ValueError(f"gripper must be a fraction 0..1, got {gripper}")
            frac = float(gripper)
            deg = JOINTS.gripper_open_deg + frac * (
                JOINTS.gripper_shut_deg - JOINTS.gripper_open_deg
            )
        if target is None:
            self.arm.gripper(deg if deg is not None else self.arm.cmd.get(6, 20.0))
            return self.arm.pose()
        return self.arm.move(target, gripper=deg)

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

    def project(self, point_m: np.ndarray, pose: dict[int, float] | None = None):
        """Base-frame point(s) -> pixel(s); see controller.project_points for the convention."""
        import importlib

        ctrl = importlib.import_module("dofbot_grasp.controller")
        C, R = ctrl.camera_pose(self.read_joints() if pose is None else pose)
        single = np.asarray(point_m, float).ndim == 1
        uv = ctrl.project_points(C, R, self._intrinsics(), point_m)
        return (float(uv[0, 0]), float(uv[0, 1])) if single else uv

    def unproject_to_table(self, u: float, v: float, z_plane: float) -> np.ndarray | None:
        import importlib

        ctrl = importlib.import_module("dofbot_grasp.controller")
        return ctrl.backproject_to_table(u, v, self.read_joints(), self._intrinsics(), z_plane)

    def close(self) -> None:
        self.arm.close()


def _make_arm(port: str, baud: int):
    """The guarded bus wrapper, imported lazily so the sim path needs no serial port."""
    from .arm import Arm

    return Arm(port=port, baud=baud)


def sim_jaw_travel() -> float:
    """Jaw slide that corresponds to fully shut in the simulated model (metres)."""
    from .params import GRASP

    return GRASP.sim_jaw_travel_m


def grasp_ready(io: RobotIO) -> bool:
    """Cheap pre-flight: the arm is inside the servo travel and the cube is reachable."""
    pose = io.read_joints()
    if any(not (0.0 < pose[i] < JOINTS.servo_range_deg[i]) for i in range(1, 6)):
        return False
    return GRASP.deadband_px > 0
