#!/usr/bin/env python3
"""Capture a checkpoint frame pair for vision-gate review.

The judgement itself is made by a multimodal model (the harness agent), not by
this script: colour thresholds on this bench scene produced false positives
(a pale desk-mat patch reads as "yellow"), so every checkpoint that matters is
confirmed by looking at the picture.

This script only does the deterministic half:

1. capture the wrist camera and the external observer camera,
2. run the classical detector as a *pre-filter* and draw what it thinks it found,
3. write both raw and annotated frames under a tag, and print the paths.

Usage (from the repository root):
    python scripts/checkpoint.py --tag before_grasp
    python scripts/checkpoint.py --tag after_lift --observer /dev/video2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dofbot_grasp.perception import detect_cube  # noqa: E402


def capture(device: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def annotate(frame: np.ndarray, obs) -> np.ndarray:
    out = frame.copy()
    if obs is not None:
        x, y, w, h = obs.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            out,
            f"pre-filter w={obs.width_px:.0f}px d={obs.distance_m * 100:.1f}cm",
            (x, max(14, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    cv2.putText(
        out,
        "UNVERIFIED: confirm with the vision gate",
        (8, out.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 255),
        1,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="short label, e.g. before_grasp")
    ap.add_argument("--wrist", default="/dev/video0")
    ap.add_argument("--observer", default=None, help="external camera, e.g. /dev/video2")
    ap.add_argument("--out", default=None, help="output directory (default: /tmp/gate)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out or "/tmp/gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    base = f"{args.tag}_{stamp}"

    report: dict[str, object] = {"tag": args.tag, "stamp": stamp, "frames": {}}

    wrist = capture(args.wrist)
    if wrist is None:
        print(f"error: could not read {args.wrist}", file=sys.stderr)
        return 2
    obs = detect_cube(wrist, white_balance=True, min_blob_saturation=140.0)
    raw_path = out_dir / f"{base}_wrist.png"
    ann_path = out_dir / f"{base}_wrist_annotated.png"
    cv2.imwrite(str(raw_path), wrist)
    cv2.imwrite(str(ann_path), annotate(wrist, obs))
    report["frames"]["wrist"] = str(raw_path)
    report["frames"]["wrist_annotated"] = str(ann_path)
    report["prefilter"] = (
        None
        if obs is None
        else {
            "center_px": [round(float(c), 1) for c in obs.center_px],
            "width_px": round(float(obs.width_px), 1),
            "distance_m": round(float(obs.distance_m), 4),
        }
    )

    if args.observer:
        obs_frame = capture(args.observer)
        if obs_frame is not None:
            obs_path = out_dir / f"{base}_observer.png"
            cv2.imwrite(str(obs_path), obs_frame)
            report["frames"]["observer"] = str(obs_path)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
