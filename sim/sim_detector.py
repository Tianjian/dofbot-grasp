"""Cube detector for rendered frames.

The simulator draws an unlit orange-brown cube on a grey floor, so the simplest possible
rule works and nothing needs a model. This lives apart from the hardware detector
(`dofbot_grasp.tracker`, which thresholds HSV saturation on real frames) because the two
sees different colours: a rule tuned on one does not fire on the other.
"""

from __future__ import annotations

import numpy as np


def detect_cube(img: np.ndarray) -> dict | None:
    """Return centroid, area fraction and bbox of the rendered cube, or None."""
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    mask = (r > 90) & (r > g + 25) & (g > b) & (b < 140)
    ys, xs = np.nonzero(mask)
    if len(xs) < 60:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "centre": (float(xs.mean()), float(ys.mean())),
        "area_frac": float(len(xs)) / mask.size,
        "bbox": (x0, y0, x1 - x0, y1 - y0),
    }
