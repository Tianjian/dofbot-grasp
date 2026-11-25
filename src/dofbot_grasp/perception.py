"""Classical (no deep-learning) cube perception for the DOFBOT wrist camera.

Pipeline: BGR frame -> HSV mask (or adaptive/Otsu fallback when the colour
mask is empty) -> binary morphology -> connected components -> convex hull ->
minimum-area rotated rectangle.  The cube is the component whose rotated
rectangle best matches a square-ish blob (rectangularity + aspect ratio).

Distance comes from the pinhole model.  A cube of edge ``S`` seen face-on at
depth ``Z`` projects to ``width_px = f_px * S / Z``, so::

    Z = f_px * S / width_px            (S = 30 mm for the DOFBOT cube)

Bearing is the horizontal pixel offset converted to an angle::

    bearing = atan2(u_center - cx, f_px)

Requirement: this module must run with NumPy + stdlib only (no cv2, no
scikit-image), because the unit tests feed it synthetic frames drawn with NumPy
arrays.  Everything below is therefore implemented from scratch and kept
deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .camera import Intrinsics, unproject_pixel

__all__ = [
    "CubeObservation",
    "DetectConfig",
    "bgr_to_hsv",
    "connected_components",
    "convex_hull",
    "cube_pose_estimate",
    "cube_position_m",
    "detect_cube",
    "min_area_rect",
]


@dataclass
class DetectConfig:
    """Thresholds and camera parameters for :func:`detect_cube`.

    All lengths are SI (metres); image quantities are pixels.
    """

    cube_size_m: float = 0.030
    #: Focal length in pixels used for the pinhole distance estimate when no
    #: :class:`~dofbot_grasp.camera.Intrinsics` is supplied.
    f_px: float = 600.0
    #: Principal point; ``None`` means "centre of the frame".
    cx: float | None = None
    cy: float | None = None
    #: OpenCV-convention HSV bounds (H 0-179, S/V 0-255).  Default: orange.
    hsv_lower: tuple[int, int, int] = (0, 70, 60)
    hsv_upper: tuple[int, int, int] = (35, 255, 255)
    #: Minimum / maximum accepted blob area, in pixels.
    min_area_px: int = 100
    max_area_frac: float = 0.5
    #: Reject blobs whose area / rotated-rect-area is below this.
    min_rectangularity: float = 0.50
    #: Reject blobs whose long/short side ratio exceeds this.
    max_aspect: float = 2.0
    #: Apply a 3x3 open+close before component labelling (suppresses speckle).
    morphology: bool = True
    #: If the HSV mask is empty, fall back to Otsu thresholding of V.
    adaptive_fallback: bool = True
    #: Fallback class preference: "smaller" (works for dark-on-bright and
    #: bright-on-dark) or "brighter".
    adaptive_prefer: str = "smaller"
    #: If the mask covers more than this fraction of the frame, skip (avoids
    #: flood-filling the whole image).
    max_mask_frac: float = 0.6


@dataclass
class CubeObservation:
    """Detected cube in image + camera geometry."""

    bbox: tuple[int, int, int, int]  # axis-aligned x, y, w, h
    center_px: tuple[float, float]
    width_px: float
    distance_m: float
    bearing_rad: float
    angle_deg: float
    mask_area_px: int
    #: rotated-rect long/short sides (pixels) and rectangularity, for QA
    rect_size_px: tuple[float, float] = (0.0, 0.0)
    rectangularity: float = 0.0

    def pixel_offset_px(self, cx: float, cy: float) -> tuple[float, float]:
        """Signed horizontal/vertical offset from the principal point."""
        return (self.center_px[0] - cx, self.center_px[1] - cy)

    def as_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "center_px": [float(v) for v in self.center_px],
            "width_px": float(self.width_px),
            "distance_m": float(self.distance_m),
            "bearing_rad": float(self.bearing_rad),
            "angle_deg": float(self.angle_deg),
            "mask_area_px": int(self.mask_area_px),
        }


# ---------------------------------------------------------------------------
# Low-level NumPy image ops
# ---------------------------------------------------------------------------


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR ``uint8``/float image to OpenCV-convention HSV.

    Returns float32 ``(H, W, 3)`` with H in [0, 180), S and V in [0, 255].
    """
    img = np.asarray(bgr)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected a BGR image with 3 channels, got shape {img.shape}")
    rgb = img[..., :3][..., ::-1].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = np.max(rgb, axis=-1)
    diff = v - np.min(rgb, axis=-1)
    s = np.where(v > 0.0, diff / np.maximum(v, 1e-6) * 255.0, 0.0)

    h = np.zeros_like(v)
    nonzero = diff > 1e-6
    # branch masks (avoid division warnings where diff == 0)
    rmax = nonzero & (v == r)
    gmax = nonzero & (v == g) & ~rmax
    bmax = nonzero & (v == b) & ~rmax & ~gmax
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(nonzero, diff, 1.0)
        h_r = 30.0 * ((g - b) / denom)  # 60 deg * (...), /2 for OpenCV range
        h_g = 30.0 * (2.0 + (b - r) / denom)
        h_b = 30.0 * (4.0 + (r - g) / denom)
    h = np.where(rmax, h_r, np.where(gmax, h_g, np.where(bmax, h_b, 0.0)))
    h = np.mod(h, 180.0)
    return np.stack([h, s, np.clip(v, 0.0, 255.0)], axis=-1).astype(np.float32)


def _in_range(hsv: np.ndarray, lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    """OpenCV-style inRange; a wrapped hue range (lower > upper) is supported."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h_lo, h_hi = float(lower[0]), float(upper[0])
    if h_lo <= h_hi:
        hmask = (h >= h_lo) & (h <= h_hi)
    else:
        hmask = (h >= h_lo) | (h <= h_hi)
    smask = (s >= float(lower[1])) & (s <= float(upper[1]))
    vmask = (v >= float(lower[2])) & (v <= float(upper[2]))
    return hmask & smask & vmask


def _otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's threshold for an 8-bit-ish grayscale image (NumPy only)."""
    values = np.clip(np.asarray(gray, dtype=np.float32), 0.0, 255.0)
    hist, _ = np.histogram(values, bins=256, range=(0.0, 256.0))
    total = hist.sum()
    if total == 0:
        return 127.5
    p = hist.astype(np.float64) / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256.0))
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0.0] = 1e-12
    sigma_b = (mu_total * omega - mu) ** 2 / denom
    return float(np.argmax(sigma_b))


def _binary_open(mask: np.ndarray) -> np.ndarray:
    """3x3 binary erosion followed by dilation (isolated speckle removal)."""
    m = _erode3(mask)
    return _dilate3(m)


def _binary_close(mask: np.ndarray) -> np.ndarray:
    """3x3 binary dilation followed by erosion (fill pinholes)."""
    m = _dilate3(mask)
    return _erode3(m)


def _erode3(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    if m.shape[0] < 3 or m.shape[1] < 3:
        return out
    core = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    out[1:-1, 1:-1] = core
    return out


def _dilate3(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    if m.shape[0] < 3 or m.shape[1] < 3:
        return out
    core = m[1:-1, 1:-1] | m[:-2, 1:-1] | m[2:, 1:-1] | m[1:-1, :-2] | m[1:-1, 2:]
    out[1:-1, 1:-1] = core
    return out


def connected_components(mask: np.ndarray, max_components: int = 64) -> list[np.ndarray]:
    """Label a boolean mask and return each component as an ``(N, 2)`` yx array.

    Uses an iterative 8-connected flood fill (no cv2 / scipy).  Components are
    returned in scan order; at most ``max_components`` are materialised.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {m.shape}")
    h, w = m.shape
    visited = np.zeros((h, w), dtype=bool)
    # Pad so neighbour arithmetic never leaves the array.
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = m
    components: list[np.ndarray] = []
    ys_all, xs_all = np.nonzero(m)
    for y0, x0 in zip(ys_all.tolist(), xs_all.tolist(), strict=True):
        if visited[y0, x0]:
            continue
        comp: list[tuple[int, int]] = []
        stack = [(y0, x0)]
        visited[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    # check bounds BEFORE indexing: an object touching the image
                    # border used to raise IndexError on real frames
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not visited[ny, nx] and padded[ny + 1, nx + 1]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if comp:
            components.append(np.array(comp, dtype=np.int32))
            if len(components) >= max_components:
                break
    return components


# ---------------------------------------------------------------------------
# Convex hull + minimum-area rectangle (rotating calipers)
# ---------------------------------------------------------------------------


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain hull of ``(N, 2)`` points -> ``(M, 2)`` points.

    Points are returned counter-clockwise in ``(x, y)`` order.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return pts.copy()
    # lexicographic unique sort by (x, y)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    unique = [pts[0]]
    for p in pts[1:]:
        if not np.allclose(p, unique[-1]):
            unique.append(p)
    if len(unique) < 3:
        return np.asarray(unique)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def min_area_rect(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Minimum-area rotated rectangle of a point set.

    Returns ``((cx, cy), (w, h), angle_deg)`` where the rectangle is aligned
    with the hull edge at ``angle_deg`` (``w`` along that edge).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("min_area_rect needs at least one point")
    if len(pts) < 3:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = (mn + mx) / 2.0
        size = mx - mn
        return (float(center[0]), float(center[1])), (float(size[0]), float(size[1])), 0.0
    hull = convex_hull(pts)
    if len(hull) < 3:
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = (mn + mx) / 2.0
        size = mx - mn
        return (float(center[0]), float(center[1])), (float(size[0]), float(size[1])), 0.0

    best_area = np.inf
    best: tuple = ((0.0, 0.0), (0.0, 0.0), 0.0)
    n = len(hull)
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        length = np.linalg.norm(edge)
        if length < 1e-9:
            continue
        u = edge / length
        v = np.array([-u[1], u[0]])
        proj_u = hull @ u
        proj_v = hull @ v
        w = float(proj_u.max() - proj_u.min())
        h = float(proj_v.max() - proj_v.min())
        area = w * h
        if area < best_area:
            best_area = area
            center = (
                u * (proj_u.max() + proj_u.min()) / 2.0 + v * (proj_v.max() + proj_v.min()) / 2.0
            )
            angle = float(np.degrees(np.arctan2(u[1], u[0])))
            best = ((float(center[0]), float(center[1])), (w, h), angle)
    return best


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _build_mask(bgr: np.ndarray, cfg: DetectConfig) -> np.ndarray | None:
    """Colour mask, or None if nothing plausible is found."""
    hsv = bgr_to_hsv(bgr)
    mask = _in_range(hsv, cfg.hsv_lower, cfg.hsv_upper)
    if cfg.morphology and mask.any():
        mask = _binary_open(mask)
        mask = _binary_close(mask)
    if mask.sum() >= cfg.min_area_px:
        return mask
    if not cfg.adaptive_fallback:
        return None
    # Adaptive fallback: Otsu on the value channel, pick the plausible class.
    v = np.clip(bgr_to_hsv(bgr)[..., 2], 0.0, 255.0)
    thr = _otsu_threshold(v)
    bright = v > thr
    dark = ~bright
    if cfg.adaptive_prefer == "brighter":
        candidates = [bright]
    else:
        candidates = [bright, dark]
    best_mask: np.ndarray | None = None
    best_area = np.inf
    for cand in candidates:
        if cfg.morphology:
            cand = _binary_close(_binary_open(cand))
        area = int(cand.sum())
        if area < cfg.min_area_px:
            continue
        if cfg.adaptive_prefer == "smaller" and area >= best_area:
            continue
        best_area = area
        best_mask = cand
    return best_mask


def _score_component(points_yx: np.ndarray, shape: tuple[int, int], cfg: DetectConfig):
    """Return a candidate dict if the component looks like the cube, else None."""
    h_img, w_img = shape
    comp_area = int(len(points_yx))
    if comp_area < cfg.min_area_px:
        return None
    if comp_area > cfg.max_area_frac * h_img * w_img:
        return None
    # (x, y) pixel coordinates
    pts_xy = np.stack([points_yx[:, 1].astype(float), points_yx[:, 0].astype(float)], axis=1)
    center, (rw, rh), angle_deg = min_area_rect(pts_xy)
    if rw <= 1.0 or rh <= 1.0:
        return None
    rect_area = rw * rh
    rectangularity = comp_area / rect_area if rect_area > 1.0 else 0.0
    if rectangularity < cfg.min_rectangularity:
        return None
    long_side, short_side = max(rw, rh), min(rw, rh)
    aspect = long_side / max(short_side, 1e-6)
    if aspect > cfg.max_aspect:
        return None
    # sanity: the blob must actually be big enough to be the 30 mm cube but
    # not absurdly large (the area filter above already covers this).
    x0 = int(np.floor(pts_xy[:, 0].min()))
    y0 = int(np.floor(pts_xy[:, 1].min()))
    x1 = int(np.ceil(pts_xy[:, 0].max()))
    y1 = int(np.ceil(pts_xy[:, 1].max()))
    return {
        "bbox": (x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)),
        "center": (float(center[0]), float(center[1])),
        "rect": (float(rw), float(rh)),
        "angle_deg": float(angle_deg),
        "rectangularity": float(rectangularity),
        "area": comp_area,
        "long_side": float(long_side),
        "aspect": float(aspect),
    }


def normalize_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Gray-world white balance, so HSV thresholds survive auto-WB drift.

    The USB camera used on the bench has no working manual white balance: when a
    large yellow object fills the frame it swings the whole scene cyan and a
    fixed hue threshold stops matching. Dividing each channel by its own mean
    removes the cast, which makes the cube's hue stable across frames.
    """
    f = np.asarray(bgr, dtype=np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    return np.clip(f * (means.mean() / means), 0.0, 255.0).astype(np.uint8)


def detect_cube(
    bgr: np.ndarray,
    cube_size_m: float | None = None,
    intrinsics: Intrinsics | None = None,
    cfg: DetectConfig | None = None,
    white_balance: bool = False,
    **overrides,
) -> CubeObservation | None:
    """Find a single cube in a BGR frame; return ``None`` when none is found.

    Parameters
    ----------
    bgr : ndarray
        ``uint8`` BGR image, ``(H, W, 3)``.
    cube_size_m : float, optional
        Cube edge length (default 30 mm from the sim contract).  Only used for
        the pinhole distance estimate, not for detection.
    intrinsics : Intrinsics, optional
        When supplied, its ``fx``/``cx``/``cy`` override ``cfg.f_px`` etc.
    cfg : DetectConfig, optional
        Threshold configuration; individual fields can be overridden through
        ``**overrides`` (e.g. ``hsv_lower=...``).
    white_balance : bool, optional
        Apply :func:`normalize_white_balance` before thresholding. Recommended
        for live USB cameras whose auto white balance drifts.
    """
    frame = np.asarray(bgr)
    if frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
        raise ValueError(f"expected a non-empty BGR image, got shape {frame.shape}")
    params = replace(cfg or DetectConfig(), **overrides)
    if cube_size_m is not None:
        params.cube_size_m = float(cube_size_m)

    f_px = float(intrinsics.fx) if intrinsics is not None else float(params.f_px)
    if intrinsics is not None:
        cx = float(intrinsics.cx)
    else:
        cx = float(params.cx) if params.cx is not None else frame.shape[1] / 2.0
    if f_px <= 0.0:
        raise ValueError("focal length f_px must be positive")

    mask = _build_mask(normalize_white_balance(frame) if white_balance else frame, params)
    if mask is None:
        return None
    if mask.sum() > params.max_mask_frac * mask.size:
        return None
    components = connected_components(mask)
    if not components:
        return None

    best = None
    best_score = -np.inf
    for points in components:
        cand = _score_component(points, mask.shape, params)
        if cand is None:
            continue
        # Prefer square-ish, well-filled blobs; break ties by detected size,
        # capped so a huge background blob does not dominate.
        score = cand["rectangularity"] - 0.05 * (cand["aspect"] - 1.0)
        score += 0.05 * min(cand["area"], 20000) / 20000.0
        if score > best_score:
            best_score = score
            best = cand
    if best is None:
        return None

    width_px = max(1.0, best["long_side"])
    distance = f_px * float(params.cube_size_m) / width_px
    u, v = best["center"]
    bearing = float(np.arctan2(u - cx, f_px))
    return CubeObservation(
        bbox=best["bbox"],
        center_px=(float(u), float(v)),
        width_px=float(width_px),
        distance_m=float(distance),
        bearing_rad=bearing,
        angle_deg=float(best["angle_deg"]),
        mask_area_px=int(best["area"]),
        rect_size_px=(float(best["rect"][0]), float(best["rect"][1])),
        rectangularity=float(best["rectangularity"]),
    )


# ---------------------------------------------------------------------------
# Camera-frame / base-frame pose estimates
# ---------------------------------------------------------------------------


def cube_pose_estimate(
    observation: CubeObservation,
    intrinsics: Intrinsics,
    camera_to_base: np.ndarray | None = None,
    cube_size_m: float = 0.030,
) -> np.ndarray:
    """Estimate the cube pose as a 4x4 homogeneous transform.

    Position: back-projected pixel centre at the pinhole ``distance_m``.
    Orientation: rotation about the optical axis from the rotated-rect angle
    (a cube's yaw is observable only up to 90-degree symmetry; the estimate is
    the face-normal-aligned minimum rotation, not a full 6-DoF pose).

    ``camera_to_base`` maps camera coordinates into the robot base frame
    (``T_base_cam`` = FK(joints) @ ``wrist_camera_to_ee()``).  When omitted the
    pose is returned in the camera frame.
    """
    u, v = observation.center_px
    p_cam = unproject_pixel(u, v, observation.distance_m, intrinsics)
    theta = np.radians(observation.angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    r_cam = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t_cam = np.eye(4)
    t_cam[:3, :3] = r_cam
    t_cam[:3, 3] = p_cam
    if camera_to_base is None:
        return t_cam
    t_base_cam = np.asarray(camera_to_base, dtype=float)
    if t_base_cam.shape != (4, 4):
        raise ValueError(f"camera_to_base must be 4x4, got {t_base_cam.shape}")
    return t_base_cam @ t_cam


def cube_position_m(
    observation: CubeObservation,
    intrinsics: Intrinsics,
    camera_to_base: np.ndarray | None = None,
) -> np.ndarray:
    """Estimated cube centre in the camera frame (or base frame if given)."""
    p = unproject_pixel(
        observation.center_px[0], observation.center_px[1], observation.distance_m, intrinsics
    )
    if camera_to_base is None:
        return p
    t = np.asarray(camera_to_base, dtype=float)
    return t[:3, :3] @ p + t[:3, 3]
