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
                st.person = bool(res and len(res[0].boxes) > 0)
            except Exception:
                st.notes.append("face model failed")
        if self._sam is not None and expect_cube:
            seed = cube_seed or self._colour_seed(frame)
            if seed is not None:
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

    @staticmethod
    def _colour_seed(frame: np.ndarray) -> tuple[float, float] | None:
        import cv2

        if frame is None or frame.size == 0:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (12, 90, 80), (45, 255, 255))
        # the observer looks down at the desk from above, so the cube is in the lower part of
        # the frame; without this the seed lands on packaging on a shelf behind it
        h = frame.shape[0]
        mask[: int(0.60 * h), :] = 0
        ys, xs = np.nonzero(mask)
        if len(xs) < 200:
            return None
        # largest blob, not the mean of all yellow pixels
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count < 2:
            return None
        # the cube covers a few hundred pixels from where the observer sits; larger blobs are
        # packaging or a poster, smaller ones are noise
        areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]
        plausible = [(a, i) for i, a in enumerate(areas, start=1) if 150 <= a <= 6000]
        if not plausible:
            return None
        biggest = max(plausible)[1]
        ys, xs = np.nonzero(labels == biggest)
        return float(xs.mean()), float(ys.mean())
