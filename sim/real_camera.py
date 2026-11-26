"""Make the simulated camera see what the real one sees.

A MuJoCo camera is a symmetric pinhole: the principal point is the image centre and there
is no lens distortion. The real wrist camera has its principal point 50 px left of centre
and strong barrel distortion (k1 -0.50, k2 +0.51), which is a 2.8 deg pointing offset and
roughly 13 px of edge error -- enough to matter against a 10 mm grasp tolerance.

So the rendered frame is resampled through the real intrinsics: for every pixel of the
output, take its ray through the real camera (undistorted), and fetch it from the ideal
pinhole render. That gives the controller the same image geometry on both sides.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class RealCamera:
    """Maps an ideal pinhole render onto a measured camera's pixel grid."""

    def __init__(self, intrinsics_path: str | Path) -> None:
        intr = json.loads(Path(intrinsics_path).read_text())
        self.width, self.height = intr["image_size"]
        self.K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
        self.dist = np.array(intr["dist"], float).reshape(1, -1)
        self._maps: tuple[np.ndarray, np.ndarray] | None = None

    def _ideal(self) -> tuple[np.ndarray, np.ndarray]:
        """For each output pixel, the ideal-pinhole coordinates to sample."""
        if self._maps is not None:
            return self._maps
        u, v = np.meshgrid(
            np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32)
        )
        pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)
        self._maps = (
            ideal[:, 0].reshape(self.height, self.width).astype(np.float32),
            ideal[:, 1].reshape(self.height, self.width).astype(np.float32),
        )
        return self._maps

    def apply(self, ideal_render: np.ndarray) -> np.ndarray:
        """Resample an ideal pinhole render onto the real pixel grid."""
        mx, my = self._ideal()
        return cv2.remap(
            ideal_render,
            mx,
            my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    @property
    def focal_px(self) -> float:
        return float(self.K[1, 1])

    def render_geometry(self, margin_px: int = 40) -> tuple[int, int, float]:
        """Render size and vertical field of view that put the pixels at the real scale.

        A MuJoCo camera's pixel scale is set by its height and field: f = (H/2)/tan(fovy/2).
        To sample through the real intrinsics the render must use the *real* focal length,
        so the field is derived from the render height rather than fixed.
        """
        mx, my = self._ideal()
        need_w = int(np.ceil(mx.max())) + 1 + margin_px
        need_h = int(np.ceil(my.max())) + 1 + margin_px
        height = max(need_h, self.height)
        width = max(need_w, self.width)
        fovy = 2.0 * np.degrees(np.arctan((height / 2.0) / self.focal_px))
        return width, height, float(fovy)

    def sample_offsets(self) -> tuple[int, int]:
        """Where the real grid's (0,0) sits inside the render frame."""
        mx, my = self._ideal()
        return int(-min(0, np.floor(mx.min()))), int(-min(0, np.floor(my.min())))

    def covers_ideal(self, fovy_deg: float) -> bool:
        """True when an ideal pinhole at ``fovy_deg`` has pixels outside the output grid.

        The real principal point sits off centre, so the ideal render must be wider than
        the output or the resample pulls in empty pixels.
        """
        mx, my = self._ideal()
        return bool(
            mx.min() < 0 or my.min() < 0 or mx.max() > self.width - 1 or my.max() > self.height - 1
        )
