"""Bus wrapper with the measured timing rules baked in.

Every rule here came from a measurement that went wrong first:

* a read frame within ~0.3 s of a ``write6`` cancels the timed move, so nothing may poll
  inside that window;
* a 1.2 s commanded move can still be travelling at 1.6 s, so "sleep then read" reads the
  arm mid-move -- poll until two samples agree instead;
* the first reply after an idle period is stale, so it is discarded and asked for twice.

Lives in the package rather than in a script so that both the serial backend and any future
entry point share one implementation.
"""

from __future__ import annotations

import time

from .bus import (
    SerialBus,
    build_read_frame,
    build_single_servo_frame,
    build_write6_frame,
    pwm_to_servo_angle,
)
from .params import BUS, JOINTS, SAFETY


class Arm:
    """Serial wrapper with the measured timing rules baked in."""

    def __init__(self, port: str = BUS.port, baud: int = BUS.baud) -> None:
        self.bus = SerialBus(port=port, baudrate=baud)
        self.bus.__enter__()
        self.cmd = {i: 90.0 for i in range(1, 6)}
        self.cmd[5] = 135.0

    def close(self) -> None:
        self.bus.__exit__(None, None, None)

    def read(self, sid: int) -> float:
        for _ in range(2):  # the first reply after idle is stale
            self.bus._drain_input()
            self.bus._write(build_read_frame(sid))
            time.sleep(0.02)
            _, data = self.bus._read_reply()
        ang = pwm_to_servo_angle(sid, (data[0] << 8) | data[1])
        return 180.0 - ang if sid in (2, 3, 4) else ang

    def pose(self) -> dict[int, float]:
        return {i: self.read(i) for i in range(1, 6)}

    def wait_settled(self) -> dict[int, float]:
        """Poll until two consecutive samples agree, per the real settling time."""
        prev = self.pose()
        t0 = time.time()
        while time.time() - t0 < BUS.settle_timeout_s:
            time.sleep(0.25)
            cur = self.pose()
            if max(abs(cur[i] - prev[i]) for i in range(1, 6)) < BUS.settle_tolerance_deg:
                return cur
            prev = cur
        return prev

    def move(self, target_deg: dict[int, float], gripper: float | None = None) -> dict[int, float]:
        """One write6 frame, then a guarded settle. Never reads inside the guard."""
        ctrl = [target_deg[i] for i in range(1, 6)]
        if gripper is None:
            gripper = self.cmd.get(6, JOINTS.gripper_open_deg)
        self.bus._write(build_write6_frame([*ctrl, gripper], int(BUS.move_time_s * 1000)))
        time.sleep(BUS.guard_after_write_s)
        settled = self.wait_settled()
        lag = max(abs(settled[i] - ctrl[i - 1]) for i in range(1, 6))
        if lag > SAFETY.max_servo_lag_deg:
            raise RuntimeError(f"servo lag {lag:.1f} deg exceeds the limit -- arm is blocked")
        for i in range(1, 6):
            self.cmd[i] = ctrl[i - 1]
        self.cmd[6] = gripper
        return settled

    def gripper(self, angle_deg: float) -> None:
        self.bus._write(build_single_servo_frame(6, angle_deg, int(BUS.move_time_s * 1000)))
        time.sleep(BUS.guard_after_write_s + BUS.move_time_s)
        self.cmd[6] = angle_deg


# ---------------------------------------------------------------- kinematics
