#!/usr/bin/env python3
"""Calibrate a camera's intrinsics from the printed A4 chessboard.

The sheet from ``scripts/make_calibration_sheet.py`` carries two boards:

    A  wrist camera   : 7x5 squares of 11 mm  ->  --cols 6 --rows 4 --square-mm 11
    B  observer/table : 9x6 squares of 20 mm  ->  --cols 8 --rows 5 --square-mm 20

Hold the sheet in front of the camera and move it around (tilt it towards each
corner, vary the distance). The script keeps any view in which the board is found
and stops after ``--shots`` accepted views, then prints the intrinsics and writes
them as JSON.

    python scripts/calibrate_camera.py --camera /dev/video0 \
        --cols 6 --rows 4 --square-mm 11 --shots 20 --out calib/intrinsics_wrist.json

No display is needed; it prints progress on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--cols", type=int, default=6, help="inner corners along width")
    ap.add_argument("--rows", type=int, default=4, help="inner corners along height")
    ap.add_argument("--square-mm", type=float, default=11.0, help="square edge in mm")
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument(
        "--min-interval-s",
        type=float,
        default=1.0,
        help="minimum time between accepted views",
    )
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--out", default="calib/intrinsics.json")
    ap.add_argument("--save-frames", default=None, help="directory for accepted frames")
    args = ap.parse_args(argv)

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * (
        args.square_mm / 1000.0
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"error: cannot open {args.camera}", file=sys.stderr)
        return 2

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    save_dir = Path(args.save_frames) if args.save_frames else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    shape = None
    last = 0.0
    t0 = time.time()
    print(
        f"looking for a {pattern[0]}x{pattern[1]} inner-corner board on {args.camera} ..."
    )
    while len(objpoints) < args.shots:
        if time.time() - t0 > args.timeout_s:
            print(
                f"timeout after {args.timeout_s:.0f}s with {len(objpoints)} views",
                file=sys.stderr,
            )
            break
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        shape = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not found:
            time.sleep(0.02)
            continue
        if time.time() - last < args.min_interval_s:
            continue
        last = time.time()
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(-1, 2))
        n = len(objpoints)
        print(
            f"  view {n}/{args.shots}  centroid=({corners[..., 0].mean():.0f},"
            f"{corners[..., 1].mean():.0f})  spread={corners[..., 0].std():.0f}px"
        )
        if save_dir:
            cv2.drawChessboardCorners(frame, pattern, corners, found)
            cv2.imwrite(str(save_dir / f"view_{n:02d}.png"), frame)

    cap.release()
    if len(objpoints) < 6:
        print(
            f"error: need at least 6 usable views, got {len(objpoints)}",
            file=sys.stderr,
        )
        return 1

    rms, K, dist, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        shape[::-1],
        None,
        None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-9),
    )

    # per-view reprojection error, so one bad view can be spotted and dropped
    per_view = []
    for obj, img in zip(objpoints, imgpoints, strict=True):
        ok_pnp, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        if not ok_pnp:
            per_view.append(float("nan"))
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        per_view.append(
            round(float(np.linalg.norm(img - proj.reshape(-1, 2), axis=1).mean()), 4)
        )

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    report = {
        "camera": args.camera,
        "pattern_inner_corners": list(pattern),
        "square_mm": args.square_mm,
        "views": len(objpoints),
        "rms_reprojection_px": round(float(rms), 4),
        "image_size": [int(shape[1]), int(shape[0])],
        "fx": round(fx, 3),
        "fy": round(fy, 3),
        "cx": round(cx, 3),
        "cy": round(cy, 3),
        "dist": [round(float(v), 6) for v in dist.reshape(-1)],
        "per_view_error_px": per_view,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    verdict = "good" if rms < 0.5 else ("usable" if rms < 1.0 else "poor - recalibrate")
    print(f"rms {rms:.3f} px -> {verdict}")
    print(f"wrote {out}")
    return 0 if rms < 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
