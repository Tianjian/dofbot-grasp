"""Calibrate the observer camera's pose from a chessboard lying on the desk.

The fingertip is the wrong target: the gripper hides it, its edge is ambiguous, and its
position is itself extrapolated along the tool axis -- so it would fold the error being
measured into the calibration. A printed chessboard on the table is unambiguous: its corners
are exact points on the desk plane, so their 3-D coordinates follow from the board's measured
placement plus the known square size.

    # 1. put the table chessboard (9x6 squares of 20 mm) flat on the desk and check it is seen
    python scripts/calibrate_observer_extrinsics.py --detect-only

    # 2. measure where the board's top-left corner sits, in mm from the arm's base, and how
    #    far its long edge is rotated from +x; then solve
    python scripts/calibrate_observer_extrinsics.py --origin 60,180 --yaw 0

Writes calib/observer_extrinsics.json: the camera's position and orientation in base
coordinates, plus the reprojection error it achieved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SQUARE_MM = 20.0
PATTERN = (8, 5)  # inner corners of the 9x6 board


def grab(device: str, *, settle: float = 0.6) -> np.ndarray | None:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    time.sleep(settle)
    frame = None
    for _ in range(4):
        ok, frame = cap.read()
        if not ok:
            frame = None
    cap.release()
    return frame


def find_board(frame: np.ndarray) -> np.ndarray | None:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(grey, PATTERN, flags)
    if not ok or corners is None:
        return None
    corners = cv2.cornerSubPix(
        grey,
        corners.astype(np.float32),
        (9, 9),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
    )
    return corners.reshape(-1, 2)


def board_points(origin_mm: tuple[float, float], yaw_deg: float) -> np.ndarray:
    """The board's inner corners in base coordinates, on the table plane."""
    cols, rows = PATTERN
    us = np.arange(cols, dtype=float) * SQUARE_MM
    vs = np.arange(rows, dtype=float) * SQUARE_MM
    grid = np.array([[u, v, 0.0] for v in vs for u in us])
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    grid = grid @ rot.T
    grid[:, 0] += origin_mm[0] / 1000.0
    grid[:, 1] += origin_mm[1] / 1000.0
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video2")
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--origin", default="0,200", help="board's top-left corner, x,y mm")
    ap.add_argument("--yaw", type=float, default=0.0, help="board rotation about z, degrees")
    ap.add_argument("--save-frame", default=None)
    args = ap.parse_args()

    intr_path = ROOT / "calib" / "intrinsics_observer.json"
    intr = json.loads(intr_path.read_text())
    frame = grab(args.device)
    if frame is None:
        print(f"no frame from {args.device}")
        return 2
    if args.save_frame:
        cv2.imwrite(args.save_frame, frame)

    corners = find_board(frame)
    if corners is None:
        print(f"no {PATTERN[0]}x{PATTERN[1]} inner-corner chessboard in view")
        print("the sheet must be flat on the desk, unoccluded, and fill a good part of the frame")
        return 1
    print(
        f"board found: {len(corners)} corners, "
        f"first {tuple(round(float(v)) for v in corners[0])}, "
        f"last {tuple(round(float(v)) for v in corners[-1])}"
    )

    vis = frame.copy()
    for i, (u, v) in enumerate(corners):
        cv2.circle(vis, (int(u), int(v)), 3, (0, 0, 255), -1)
        if i % 10 == 0:
            cv2.putText(
                vis,
                str(i),
                (int(u) + 4, int(v) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
    out = ROOT / "sim" / "observer_chessboard.png"
    cv2.imwrite(str(out), vis)
    print(f"annotated: {out}")

    if args.detect_only:
        return 0

    ox, oy = (float(v) for v in args.origin.split(","))
    object_points = board_points((ox, oy), args.yaw)
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    dist = np.array(intr["dist"], float).reshape(-1, 1)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, corners.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        print("solvePnP failed")
        return 1
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    err = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
    rot, _ = cv2.Rodrigues(rvec)
    payload = {
        "source": "chessboard on the table, solvePnP",
        "device": args.device,
        "board": {
            "pattern": list(PATTERN),
            "square_mm": SQUARE_MM,
            "origin_mm": [ox, oy],
            "yaw_deg": args.yaw,
        },
        "position_m": [float(v) for v in tvec.ravel()],
        "rotation": [[float(v) for v in row] for row in rot],
        "reprojection_rms_px": float(np.sqrt(np.mean(err**2))),
        "reprojection_max_px": float(err.max()),
        "corners_used": int(len(corners)),
    }
    dest = ROOT / "calib" / "observer_extrinsics.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"camera at ({tvec[0] * 1000:.0f}, {tvec[1] * 1000:.0f}, {tvec[2] * 1000:.0f}) mm")
    print(
        f"reprojection: rms {payload['reprojection_rms_px']:.2f} px, "
        f"max {payload['reprojection_max_px']:.2f} px"
    )
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
