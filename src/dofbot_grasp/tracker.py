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


class ColourDetector:
    """CPU fallback: threshold, largest blob, centroid."""

    name = "colour"

    def __init__(self, min_blob_saturation: float = 140.0) -> None:
        self.min_blob_saturation = min_blob_saturation

    def __call__(self, frame: np.ndarray | None) -> dict | None:
        if frame is None:
            return None
        from .perception import detect_cube

        obs = detect_cube(frame, white_balance=True, min_blob_saturation=self.min_blob_saturation)
        if obs is None:
            return None
        return {
            "centre": (float(obs.center_px[0]), float(obs.center_px[1])),
            "area": float(obs.mask_area_px),
            "source": self.name,
        }

    def reset(self) -> None:
        return None


class SamTracker:
    """SAM2 point-prompted tracker, seeded by the colour detector.

    Costs 134-863 ms per frame on an RTX 2060, so it is used where accuracy matters (the
    back-projection) and the colour blob still provides the seed.
    """

    name = "sam2"

    def __init__(self, weights: str = "models/sam2_t.pt", device: str | None = None) -> None:
        from pathlib import Path

        from ultralytics import SAM

        root = Path(__file__).resolve().parents[2]
        self.model = SAM(str(root / weights) if (root / weights).exists() else weights)
        if device:
            self.model.to(device)
        self.seed = ColourDetector()
        self.last: tuple[float, float] | None = None

    def __call__(self, frame: np.ndarray | None) -> dict | None:
        if frame is None:
            return None
        prompt = self.last
        if prompt is None:
            seed = self.seed(frame)
            if seed is None:
                return None
            prompt = seed["centre"]
        res = self.model.predict(frame, points=[[prompt[0], prompt[1]]], labels=[1], verbose=False)
        masks = res[0].masks
        if masks is None or len(masks.data) == 0:
            self.last = None
            return None
        seg = (masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
        if int(seg.sum()) < 300:
            self.last = None
            return None
        ys, xs = np.nonzero(seg)
        centre = (float(xs.mean()), float(ys.mean()))
        self.last = centre
        return {"centre": centre, "area": float(seg.sum()), "source": self.name, "mask": seg}

    def reset(self) -> None:
        self.last = None


def make_detector(prefer_gpu: bool = True, weights: str = "models/sam2_t.pt") -> Callable:
    """GPU tracker when one is usable, colour otherwise. Never raises on import."""
    if prefer_gpu and gpu_available():
        try:
            return SamTracker(weights=weights)
        except Exception:
            pass
    return ColourDetector()
