"""SAM2-based cube tracker for the wrist camera.

Why SAM2: the cube's top face is pale (S~43) and less saturated than the desk mat
(S~104), so a colour threshold only sees the lit front face.  SAM2 segments the
whole visible silhouette from a single point prompt, and re-prompting with the
previous centroid keeps the mask stable across frames.

The tracker is deliberately independent of any camera extrinsic: it reports the
pixel centroid and mask area only.  Image-based servoing consumes those directly.
"""

from __future__ import annotations

import numpy as np


class CubeTracker:
    """Point-prompted SAM2 tracker seeded by a colour prior."""

    def __init__(self, weights: str = "sam2_t.pt", device: str | None = None) -> None:
        from ultralytics import SAM

        self.model = SAM(weights)
        if device:
            self.model.to(device)
        self.last: tuple[float, float] | None = None

    @staticmethod
    def _colour_seed(frame: np.ndarray) -> tuple[float, float] | None:
        """Cheap saturated-yellow centroid, used only to place the first prompt."""
        import cv2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (12, 110, 80), (45, 255, 255))
        ys, xs = np.nonzero(mask)
        if len(xs) < 300:
            return None
        return float(xs.mean()), float(ys.mean())

    def update(self, frame: np.ndarray, prompt: tuple[float, float] | None = None) -> dict | None:
        """Segment the cube in ``frame``; returns centroid, area and the mask."""
        import cv2

        p = prompt or self.last or self._colour_seed(frame)
        if p is None:
            return None
        res = self.model.predict(frame, points=[[p[0], p[1]]], labels=[1], verbose=False)
        masks = res[0].masks
        if masks is None or len(masks.data) == 0:
            return None
        seg = (masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
        if int(seg.sum()) < 300:
            return None
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        m = cv2.moments(hull)
        if m["m00"] <= 0:
            return None
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        x, y, w, h = cv2.boundingRect(hull)
        self.last = (cx, cy)
        clipped = bool(
            x <= 2 or y <= 2 or x + w >= frame.shape[1] - 2 or y + h >= frame.shape[0] - 2
        )
        return {
            "centre": (cx, cy),
            "area": float(cv2.contourArea(hull)),
            "bbox": (x, y, w, h),
            "clipped": clipped,
            "mask": seg,
        }

    def reset(self) -> None:
        self.last = None
