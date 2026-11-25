#!/usr/bin/env python3
"""Bench bring-up for the DOFBOT SE: read state, capture the wrist camera.

Nothing moves unless you ask it to. `probe` only reads servo angles and grabs a
frame; `move` nudges one servo by a few degrees and reads it back, which is how
you verify joint direction before trusting the controller.

Examples
--------
    # read-only: servos + camera
    python scripts/hw_probe.py probe --port /dev/ttyUSB0 --camera /dev/video0

    # nudge servo 1 by +5 degrees (small, reversible)
    python scripts/hw_probe.py move --port /dev/ttyUSB0 --servo 1 --delta 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dofbot_grasp.bus import (
    SERVO_REVERSED_IDS,
    SerialBus,
    SerialProtocolError,
    build_read_frame,
    pwm_to_servo_angle,
)


def read_raw_servos(bus: SerialBus, retries: int = 3) -> dict[int, float]:
    """Read servo 1-5 angles the same way SerialBus does, but keep them raw."""
    angles: dict[int, float] = {}
    for sid in range(1, 6):
        last: Exception | None = None
        for _ in range(retries):
            try:
                bus._write(build_read_frame(sid))
                time.sleep(0.001)
                ext_type, data = bus._read_reply()
                if ext_type != 0x0A:
                    raise SerialProtocolError(f"unexpected reply type 0x{ext_type:02X}")
                position = (data[0] << 8) | data[1]
                angle = pwm_to_servo_angle(sid, position)
                if sid in SERVO_REVERSED_IDS:
                    angle = 180.0 - angle
                angles[sid] = round(angle, 1)
                last = None
                break
            except SerialProtocolError as exc:
                last = exc
        if last is not None:
            raise last
    return angles


def cmd_probe(args: argparse.Namespace) -> int:
    out: dict[str, object] = {"port": args.port}
    with SerialBus(port=args.port, baudrate=args.baud, timeout=args.timeout) as bus:
        t0 = time.time()
        raw = read_raw_servos(bus)
        dt = time.time() - t0
        joints = bus.read_joints()
        out["servo_deg"] = raw
        out["joint_rad"] = [round(float(v), 4) for v in joints]
        out["joint_deg"] = [round(float(v), 1) for v in np.degrees(joints)]
        out["read_seconds"] = round(dt, 3)
        print(f"port            : {args.port}")
        print(f"servo angles    : {raw}")
        print(f"joint deg (mapped): {out['joint_deg']}")
        print(f"read time       : {dt:.3f} s for 5 servos")

    if args.camera:
        import cv2

        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"camera          : FAILED to read {args.camera}", file=sys.stderr)
            return 2
        out["frame_shape"] = list(frame.shape)
        print(f"camera          : {frame.shape[1]}x{frame.shape[0]}")
        if args.save:
            import cv2 as _cv2

            _cv2.imwrite(args.save, frame)
            print(f"frame saved     : {args.save}")
        try:
            from dofbot_grasp.perception import detect_cube

            obs = detect_cube(frame)
            out["cube"] = None if obs is None else obs.as_dict()
            print(f"cube detection  : {'none' if obs is None else obs.as_dict()}")
        except Exception as exc:  # pragma: no cover - exploratory
            print(f"cube detection  : error {type(exc).__name__}: {exc}")
    print(json.dumps(out, indent=2))
    return 0


def cmd_move(args: argparse.Namespace) -> int:
    with SerialBus(port=args.port, baudrate=args.baud, timeout=args.timeout) as bus:
        raw = read_raw_servos(bus)
        sid = int(args.servo)
        before = raw[sid]
        target = float(np.clip(before + args.delta, 0.0, 270.0 if sid == 5 else 180.0))
        servo = [raw[1], raw[2], raw[3], raw[4], raw[5], args.gripper]
        print(f"servo {sid}: {before} deg -> {target} deg (delta {args.delta:+.1f})")
        print("WATCH THE ARM: this is a small, reversible move.")
        time.sleep(1.0)
        servo[sid - 1] = target
        bus.write_servo_deg(servo, duration_s=args.duration)
        time.sleep(args.duration + 0.4)
        after = read_raw_servos(bus)
        print(f"commanded       : {target} deg")
        print(f"read back       : {after[sid]} deg")
        print(f"error           : {after[sid] - target:+.1f} deg")
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", default="/dev/ttyUSB0")
    common.add_argument("--baud", type=int, default=115200)
    common.add_argument("--timeout", type=float, default=0.3)

    p = sub.add_parser("probe", parents=[common], help="read servos and grab a camera frame")
    p.add_argument("--camera", default="/dev/video0")
    p.add_argument("--save", default=None, help="write the captured frame to this path")
    p.set_defaults(func=cmd_probe)

    m = sub.add_parser("move", parents=[common], help="nudge one servo and read it back")
    m.add_argument("--servo", type=int, required=True, choices=[1, 2, 3, 4, 5])
    m.add_argument("--delta", type=float, default=5.0)
    m.add_argument("--duration", type=float, default=0.8)
    m.add_argument("--gripper", type=float, default=90.0)
    m.set_defaults(func=cmd_move)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
