"""Cube detection front-end: SAM2 on the GPU when it is available, colour otherwise.

The colour rule only sees the cube's lit front face -- the pale top face is less saturated
than the desk mat -- so its centroid sits off the cube's centre. That bias is what a
back-projected pixel turns into a few millimetres of aim error. SAM2 segments the whole
visible silhouette from a point prompt, and re-prompting with the previous centroid keeps
the mask stable across frames, which is what image-based servoing wants.

The tracker returns pixels only. No extrinsics, no range: the controller back-projects.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def gpu_available() -> bool:
    """True when a CUDA-capable torch build can actually see a device."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class SamCubeDetector:
    """Find the cube by segmenting everything and picking the cube-shaped mask.

    Prompt-free on purpose. The classical path needed a colour threshold, and on the real desk
    that threshold was doing two jobs badly: the cube is pale butter-yellow (mean saturation
    139 under indoor light) while the paper around it sits at 89, so any threshold sits within
    a few units of the thing it is meant to find. Segmentation does not care about colour.

    A cube seen from a camera looks like a filled, roughly square quadrilateral, which is a
    shape test rather than a colour test -- so it survives the stickers, the lighting and the
    papers, none of which the threshold did.
    """

    name = "sam2"

    def __init__(
        self, weights: str = "models/sam2_t.pt", imgsz: int = 512, device: str | None = None
    ) -> None:
        from pathlib import Path

        from ultralytics import SAM

        root = Path(__file__).resolve().parents[2]
        self.model = SAM(str(root / weights) if (root / weights).exists() else weights)
        if device:
            self.model.to(device)
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray | None) -> dict | None:
        if frame is None:
            return None
        res = self.model.predict(frame, imgsz=self.imgsz, verbose=False)
        masks = res[0].masks
        if masks is None or len(masks.data) == 0:
            return None
        boxes = res[0].boxes.xyxy.cpu().numpy()
        segs = masks.data.cpu().numpy()
        frame_area = float(frame.shape[0] * frame.shape[1])
        best = None
        for seg, box in zip(segs, boxes, strict=False):
            binary = seg > 0.5
            area = float(binary.sum())
            if not (800.0 <= area <= 0.55 * frame_area):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            aspect = w / h
            if not 0.65 <= aspect <= 1.55:
                continue
            fill = area / (w * h)
            if fill < 0.45:
                continue
            # prefer filled and large, which favours the cube over a strip of paper
            score = fill + 0.4 * min(area / (0.25 * frame_area), 1.0)
            if best is None or score > best[0]:
                ys, xs = np.nonzero(binary)
                best = (score, float(xs.mean()), float(ys.mean()), area, binary)
        if best is None:
            return None
        _score, u, v, area, mask = best
        # A full-frame boolean mask, not a point list: the 30 mm prior is fitted by overlapping
        # the projected cube with this array, so its shape has to be the image's.
        return {
            "centre": np.array([u, v]),
            "mask": mask,
            "area": area,
            "width_px": float(np.sqrt(area)),
        }


def make_detector(prefer_gpu: bool = True, weights: str = "models/sam2_t.pt") -> Callable:
    """The GPU segmenter. Detection is segmentation now; there is no colour fallback.

    A fallback that quietly produced worse answers was worse than an error: it made a wrong
    measurement look like a working system.
    """
    return SamCubeDetector(weights=weights)
