"""Semantic checks on the observer view, on the GPU when one is available.

Pixel differencing answers "did the picture change", which cannot tell a person from a shadow.
Two semantic questions matter while the arm is moving:

* **is somebody in the room with it?** A face detector answers that cheaply and reliably --
  the observer camera looks at the desk from where the operator sits.
* **is the cube still where it was?** SAM2 segments it, so a shove or an unexpected lift is
  visible as the cube's centroid moving, not as pixels flickering.

Both are slower than the pixel check by two orders of magnitude (SAM2 costs 134-863 ms a
frame), so they are meant to be called at a low rate -- a few times a second -- while the
cheap check keeps running at video rate. Everything degrades to "unavailable" rather than
raising when torch is missing, because the monitor must not depend on the GPU being free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SemanticState:
    """What the last semantic pass saw; empty when the models are unavailable."""

    available: bool = False
    person: bool = False
    cube_checked: bool = False  # whether segmentation actually ran
    cube_centre: tuple[float, float] | None = None
    cube_was_at: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)

    def cube_shifted(self, threshold_px: float = 40.0) -> float | None:
        if self.cube_centre is None or self.cube_was_at is None:
            return None
        return float(
            np.hypot(
                self.cube_centre[0] - self.cube_was_at[0], self.cube_centre[1] - self.cube_was_at[1]
            )
        )


class ObserverVision:
    """Face presence and cube segmentation for the observer camera."""

    def __init__(
        self,
        root: Path,
        *,
        device: str = "cuda:0",
        want_person: bool = True,
        want_cube: bool = True,
    ) -> None:
        self.state = SemanticState()
        self._sam = None
        self._face = None
        try:
            import torch

            if not torch.cuda.is_available():
                return
        except Exception:
            return
        try:
            if want_person:
                candidates = list(
                    (root.parents[1] / "face_tracker" / "models").glob("yolo*face*.pt")
                )
                in_tree = list((root / "models").glob("yolo*face*.pt"))
                weights = (in_tree + candidates)[0] if (in_tree or candidates) else None
                if weights is not None:
                    from ultralytics import YOLO

                    self._face = YOLO(str(weights))
            if want_cube:
                cube_weights = root / "models" / "sam2_t.pt"
                if cube_weights.exists():
                    from ultralytics import SAM

                    self._sam = SAM(str(cube_weights))
        except Exception:
            self._sam = None
            self._face = None
            return
        self.state.available = self._face is not None or self._sam is not None

    # -- checks ---------------------------------------------------------
    def check(
        self,
        frame: np.ndarray,
        *,
        cube_seed: tuple[float, float] | None = None,
        expect_cube: bool = True,
    ) -> SemanticState:
        """One semantic pass. Cheap enough to call a few times a second, not every frame."""
        st = SemanticState(available=self.state.available)
        if not st.available:
            return st
        if self._face is not None:
            try:
                res = self._face.predict(frame, verbose=False, conf=0.35)
                boxes = res[0].boxes if res else None
                st.person = bool(boxes is not None and len(boxes) > 0)
            except Exception:
                st.notes.append("face model failed")
        if self._sam is not None and expect_cube:
            # The target's top face is yellow but its four sides carry white stickers, and the
            # observer looks at it from the side -- so a colour seed never finds it in this
            # view. The seed has to come from somewhere that knows where the cube is.
            seed = cube_seed
            if seed is not None:
                st.cube_checked = True
                try:
                    res = self._sam.predict(
                        frame, points=[[seed[0], seed[1]]], labels=[1], verbose=False
                    )
                    masks = res[0].masks
                    if masks is not None and len(masks.data) > 0:
                        seg = masks.data[0].cpu().numpy() > 0.5
                        if int(seg.sum()) > 300:
                            ys, xs = np.nonzero(seg)
                            st.cube_centre = (float(xs.mean()), float(ys.mean()))
                            st.cube_was_at = self.state.cube_centre or st.cube_centre
                except Exception:
                    st.notes.append("cube segmentation failed")
        self.state = st
        return st


def cube_side_px(frame: np.ndarray, *, roi_top_fraction: float = 0.55) -> float | None:
    """How wide the cube looks in the observer frame, in pixels.

    Image-space only, deliberately: the observer's intrinsics are calibrated but its extrinsics
    are not, so a pixel cannot be turned into a base-frame point. Its apparent size can still
    be turned into a *distance*, because the cube's edge is a known 30 mm -- and that is enough
    to check a position measured by another camera for gross disagreement.
    """
    import cv2

    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    roi = frame[int(roi_top_fraction * h) :, :]
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: float | None = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 400.0 < area < 20000.0:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh == 0:
            continue
        aspect = bw / float(bh)
        if not 0.55 < aspect < 1.9:  # a cube face seen from the side stays roughly square
            continue
        side = (bw + bh) / 2.0
        if side > (best or 0.0):
            best = float(side)
    return best


def distance_from_side_px(side_px: float, intr: dict, cube_edge_m: float = 0.030) -> float | None:
    """Distance implied by the cube's apparent width, or None if the input is nonsense."""
    if side_px is None or side_px <= 1.0:
        return None
    return float(cube_edge_m * intr["fx"] / side_px)
