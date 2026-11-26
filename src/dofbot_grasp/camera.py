"""Camera abstractions for the wrist-mounted DOFBOT camera.

Contract (see ``docs/controller-architecture.md``): a camera is anything with

    read() -> np.ndarray  # BGR image, shape (H, W, 3), uint8

``UsbCamera`` wraps OpenCV's ``VideoCapture``; OpenCV is imported lazily so the
pure perception/kinematics unit tests run without it installed.  ``SimCamera``
is the placeholder the MuJoCo/sim workstream wires to the simulator's renderer:
by itself it raises ``RuntimeError`` on ``read()``, rather than silently
returning a black frame that perception could mis-detect.

The wrist camera is mounted on link5 exactly as in the fixed sim/real contract
(``docs/controller-architecture.md`` / ``sim/dofbot_se.xml``)::

    <camera name="wrist_cam" pos="0.035 0 0.010" quat="0 1 0 0" fovy="75"/>

i.e. 35 mm beside the tool and 10 mm along it, looking **down the tool axis**
(+Z of link5).  Because the pinhole convention used here is +Z forward, +X
right, +Y down, the camera frame coincides with the link5 frame orientation::

    x_cam = +X_ee   (image right)
    y_cam = +Y_ee   (image down)
    z_cam = +Z_ee   (optical axis / tool direction)

and the translation is ``(0.035, 0, 0.010)`` in link5.  This was cross-checked
against MuJoCo: ``cam_xmat`` is diag(1, -1, -1) relative to link5, whose local
-Z is the viewing direction, and ``cam_xpos - link5_pos`` equals the offset
above.  The transform is exposed as :func:`wrist_camera_to_ee` so the pipeline
can lift a detected cube from the camera frame into the robot base frame
exactly; per-camera overrides go through the pipeline ``camera_to_ee`` config.

``intrinsics_from_fovy`` converts the MJCF ``fovy``/resolution into the pinhole
intrinsics perception needs (fovy 75 deg at 480x640 -> fx = fy ~313 px).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Intrinsics",
    "SimCamera",
    "UsbCamera",
    "WristCamera",
    "ee_to_wrist_camera",
    "intrinsics_from_fovy",
    "project_point",
    "unproject_pixel",
    "wrist_camera_to_ee",
]


@runtime_checkable
class WristCamera(Protocol):
    """Anything that can return a BGR frame."""

    def read(self) -> np.ndarray:
        """Return a BGR image (``uint8``, shape ``(H, W, 3)``)."""
        ...


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels (``cx``/``cy`` is the principal point)."""

    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    def scaled(self, width: int, height: int) -> Intrinsics:
        """Return intrinsics rescaled to a new image resolution."""
        sx = width / float(self.width)
        sy = height / float(self.height)
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
        )

    def as_matrix(self) -> np.ndarray:
        """3x3 camera matrix ``K``."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )


#: Wrist-camera mount offset in the link5 frame (metres), from the sim contract.
CAMERA_OFFSET_LINK5_M = np.array([0.035, 0.0, 0.010])

#: Camera axes expressed in the link5 (EE) frame; columns are the camera axes.
#: The MJCF camera looks down -Z_local and is rotated 180 deg about X relative
#: to link5, so the pinhole (+Z forward, +X right, +Y down) frame is the link5
#: frame orientation itself.
_R_EE_CAM = np.eye(3)


def wrist_camera_to_ee() -> np.ndarray:
    """4x4 transform from the wrist-camera frame to the link5 (EE) frame.

    ``T_ee_cam = [[I, (0.035, 0, 0.010)], [0, 1]]``: pinhole axes aligned with
    link5 axes (optical +Z down the tool axis), mounted beside the tool.
    """
    t = np.eye(4)
    t[:3, :3] = _R_EE_CAM
    t[:3, 3] = CAMERA_OFFSET_LINK5_M
    return t


def ee_to_wrist_camera() -> np.ndarray:
    """Inverse of :func:`wrist_camera_to_ee` (4x4)."""
    t = np.eye(4)
    t[:3, :3] = _R_EE_CAM.T
    t[:3, 3] = -_R_EE_CAM.T @ CAMERA_OFFSET_LINK5_M
    return t


def intrinsics_from_fovy(
    fovy_deg: float,
    width: int,
    height: int,
    cx: float | None = None,
    cy: float | None = None,
) -> Intrinsics:
    """Pinhole intrinsics from a vertical field of view (MJCF ``fovy``).

    Uses square pixels: ``fy = (height / 2) / tan(fovy / 2)`` and ``fx = fy``.
    """
    fy = (float(height) / 2.0) / np.tan(np.radians(float(fovy_deg)) / 2.0)
    return Intrinsics(
        fx=fy,
        fy=fy,
        cx=float(width) / 2.0 if cx is None else float(cx),
        cy=float(height) / 2.0 if cy is None else float(cy),
        width=int(width),
        height=int(height),
    )


def project_point(point_cam: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Project a 3D camera-frame point to ``(u, v)`` pixels.

    Points behind the camera (``z <= 0``) are projected by the same formula but
    callers should reject them; a ``z`` of zero raises ``ValueError``.
    """
    p = np.asarray(point_cam, dtype=float).reshape(3)
    if p[2] == 0.0:
        raise ValueError("cannot project a point with z == 0")
    u = intrinsics.fx * p[0] / p[2] + intrinsics.cx
    v = intrinsics.fy * p[1] / p[2] + intrinsics.cy
    return np.array([u, v], dtype=float)


def unproject_pixel(u: float, v: float, depth_m: float, intrinsics: Intrinsics) -> np.ndarray:
    """Back-project pixel ``(u, v)`` at ``depth_m`` to a camera-frame point."""
    z = float(depth_m)
    x = (float(u) - intrinsics.cx) * z / intrinsics.fx
    y = (float(v) - intrinsics.cy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=float)


class UsbCamera:
    """OpenCV-backed USB camera (``cv2.VideoCapture``), imported lazily.

    Parameters
    ----------
    index : int
        ``VideoCapture`` device index (0 is usually the first USB camera).
    width, height : int
        Requested capture resolution.
    warmup_frames : int
        Frames to discard after opening; many UVC cameras return an
        unexposed/black first frame.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, warmup_frames: int = 3):
        try:  # guarded: cv2 is optional for the pure control tests
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "UsbCamera requires OpenCV (pip install opencv-python); import cv2 failed"
            ) from exc
        self._cv2 = cv2
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self._cap: cv2.VideoCapture | None = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"could not open USB camera index {self.index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        for _ in range(int(warmup_frames)):
            self._cap.read()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.release()
            raise RuntimeError(f"USB camera index {self.index} opened but returned no frame")
        self._intrinsics = Intrinsics(
            cx=frame.shape[1] / 2.0,
            cy=frame.shape[0] / 2.0,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
        )

    @property
    def intrinsics(self) -> Intrinsics:
        """Default intrinsics (principal point at the image centre).

        ``fx``/``fy`` are not calibrated by this class; override the returned
        intrinsics in the pipeline config after running a calibration.
        """
        return self._intrinsics

    def read(self) -> np.ndarray:
        cap = self._cap
        if cap is None:
            raise RuntimeError("camera is closed")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("USB camera read failed")
        return frame

    def release(self) -> None:
        cap = self._cap
        if cap is not None:
            cap.release()
            self._cap = None

    def __enter__(self) -> UsbCamera:
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


class SimCamera:
    """Placeholder camera for the simulator.

    The sim workstream owns the MuJoCo model; it wires a frame provider into
    this class (a callable returning a BGR ``uint8`` image, e.g. from
    ``mujoco.Renderer``).  Until it is wired, ``read()`` raises instead of
    returning garbage, so a misconfigured run fails loudly.

    Parameters
    ----------
    frame_fn : callable, optional
        ``frame_fn() -> BGR ndarray``.  May also be an object with ``read()``.
    intrinsics : Intrinsics, optional
        Simulator camera intrinsics; pipeline uses it for distance/pose math.
    """

    def __init__(
        self,
        frame_fn: Callable[[], np.ndarray] | WristCamera | None = None,
        intrinsics: Intrinsics | None = None,
    ):
        self._frame_fn = frame_fn
        self._intrinsics = intrinsics or Intrinsics()

    @property
    def intrinsics(self) -> Intrinsics:
        return self._intrinsics

    def wire(self, frame_fn: Callable[[], np.ndarray] | WristCamera) -> None:
        """Attach the simulator frame source after construction."""
        self._frame_fn = frame_fn

    def read(self) -> np.ndarray:
        if self._frame_fn is None:
            raise RuntimeError(
                "SimCamera is not wired: pass frame_fn=... or call wire() with a "
                "callable returning a BGR frame (see the sim workstream's adapter)"
            )
        if hasattr(self._frame_fn, "read"):
            return self._frame_fn.read()
        return self._frame_fn()

    def __enter__(self) -> SimCamera:
        return self

    def __exit__(self, *exc_info) -> None:
        return None
