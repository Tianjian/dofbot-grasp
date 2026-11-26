"""Recover a cube's centre by fitting the cube, not by averaging its pixels.

A silhouette centroid is only the object's centre when the whole silhouette is visible and
symmetric. Here it often is not: the pale top face falls below a saturation threshold, and
the jaws cut across the sides as the arm descends. The centroid then slides towards whatever
fragment was measured, and a back-projected pixel inherits that bias.

The object is a 30 mm cube resting on a table, so five of its six degrees of freedom are
already known and only three are free: where it sits on the table and its yaw. Fitting those
three by overlapping the projected cube with the observed mask gives the centre directly,
and a partially hidden cube still fits as long as enough of it is seen.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cube_corners(edge: float) -> np.ndarray:
    """The eight corners of a cube sitting on z = 0, centred on the origin in x/y."""
    h = edge / 2.0
    return np.array(
        [
            [sx * h, sy * h, sz * edge]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (0.0, 1.0)
        ]
    )


def silhouette(points_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Filled convex hull of the projected cube, as a uint8 mask."""
    import cv2

    hull = cv2.convexHull(points_px.astype(np.float32))
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    return mask


def fit_cube(
    observed: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray | None],
    *,
    edge: float = 0.030,
    seed_xy: tuple[float, float] | None = None,
    yaw_steps: int = 24,
    refine: int = 60,
) -> dict | None:
    """Fit a 30 mm cube's table position and yaw to ``observed``.

    ``project`` maps base-frame points to pixels and may return None when the point is
    behind the camera. ``seed_xy`` should be a rough position (the old centroid
    back-projection is good enough); without one the fit cannot start.
    """
    if seed_xy is None:
        return None
    shape = observed.shape[:2]
    corners = cube_corners(edge)

    def score(x: float, y: float, yaw: float) -> float:
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pts = corners @ rot.T + np.array([x, y, 0.0])
        px = project(pts)
        if px is None or not np.all(np.isfinite(px)):
            return 0.0
        model = silhouette(px, shape)
        inter = float(np.logical_and(model, observed).sum())
        union = float(np.logical_or(model, observed).sum())
        return inter / union if union else 0.0

    best = (0.0, np.array([seed_xy[0], seed_xy[1], 0.0]))
    for k in range(yaw_steps):
        yaw = np.pi / 2.0 * k / yaw_steps  # a square repeats every 90 deg
        val = score(seed_xy[0], seed_xy[1], yaw)
        if val > best[0]:
            best = (val, np.array([seed_xy[0], seed_xy[1], yaw]))

    # local refinement: Nelder-Mead over (x, y, yaw)
    params = best[1].copy()
    steps = np.array([0.004, 0.004, 0.08])
    val = best[0]
    for _ in range(refine):
        improved = False
        for i in range(3):
            for sign in (+1.0, -1.0):
                trial = params.copy()
                trial[i] += sign * steps[i]
                v = score(*trial)
                if v > val:
                    params, val, improved = trial, v, True
                    break
            if improved:
                break
        if not improved:
            steps *= 0.5
            if steps.max() < 1e-4:
                break
    return {
        "centre_m": (float(params[0]), float(params[1])),
        "yaw_rad": float(params[2]),
        "iou": float(val),
        "corners_px": project(
            corners
            @ np.array(
                [
                    [np.cos(params[2]), -np.sin(params[2]), 0.0],
                    [np.sin(params[2]), np.cos(params[2]), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ).T
            + np.array([params[0], params[1], 0.0])
        ),
    }


def mask_centroid(observed: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(observed)
    return None if len(xs) < 30 else (float(xs.mean()), float(ys.mean()))
