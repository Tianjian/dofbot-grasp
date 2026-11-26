"""Servo-bus abstractions for the DOFBOT SE arm.

The sim/real contract (``docs/controller-architecture.md``) requires:

    ServoBus.write_joints(joints_rad, duration_s)
    ServoBus.read_joints() -> joints_rad

``SimBus`` delegates to a callable joint-target function (the sim workstream
wires it to MuJoCo qpos/actuators).  ``SerialBus`` speaks the real Arm_Lib
serial frame protocol at 115200 baud; pyserial is imported lazily so unit tests
never need it.

Wire protocol (source: ``prior/arm_cam_control/Arm_Lib.py``)
------------------------------------------------------------
``Arm_serial_servo_write6`` (Arm_Lib.py lines 146-191) emits a 19-byte frame::

    index  bytes                          meaning
    0      0xFF                           frame head (__HEAD)
    1      0xFC                           device id (__DEVICE_ID)
    2      0x11                           length: 17 = 2 + 14 data + 1 checksum
    3      0x1D                           function: write 6 servos
    4..17  S1H S1L S2H S2L ... S6H S6L   positions, big-endian
    18     TH TL and checksum? -> actually indices 16,17 are TH,TL
    18     checksum = (5 + sum(frame[0:18])) & 0xFF

More precisely: ``cmd = [0xFF, 0xFC, 0x11, 0x1D, S1H,S1L, ..., S6H,S6L, TH,TL]``
(18 bytes), then ``checksum = sum(cmd, 5) & 0xFF`` and append it (Arm_Lib.py
line 186: ``checksum = sum(cmd,5) & 0xff``).  ``sum(cmd, 5)`` starts the
accumulator at 5 and adds every byte; the firmware receiver mirrors this as
``check_sum = length + type + sum(data)`` (= 0x11 + 0x1D + 14 data bytes = 46 +
sum(data), and 5 + 0xFF + 0xFC = 0x200 = 512 ≡ 0 mod 256, so the two agree).

Position mapping (Arm_Lib.py lines 152-175)::

    servo 1,2,3,4,6: raw = int((3100-900)*angle/180 + 900)     (900..3100 us)
    servo 5:         raw = int((3700-380)*angle/270 + 380)     (380..3700 us)

Before the PWM conversion, ``Arm_serial_servo_write6`` reverses IDs 2/3/4::

    s2, s3, s4 = 180 - s, 180 - s, 180 - s                     (lines 156-169)

A single-servo write frame (``Arm_serial_servo_write``, Arm_Lib.py lines
100-101 / 118-119 / 136-137) is::

    [0xFF, 0xFC, 0x07, 0x10 + id, PH, PL, TH, TL, checksum]

and a read request (Arm_serial_servo_read, lines 363-364) is::

    [0xFF, 0xFC, 0x03, id + 0x30, checksum]    # e.g. id 1 -> 0x31

The firmware answers ``[0xFF, 0xFB, len, type, data..., checksum]`` where
``0xFB = DEVICE_ID - 1``.  For a servo read, ``type = 0x0A`` and ``data``
carries ``position_H, position_L, id`` (Arm_Lib ``__parse_data``).  The
returned position is converted back to degrees with the inverse of the PWM
mapping and IDs 2/3/4 are un-reversed (``d = 180 - d``, Arm_Lib.py lines
389-405); :func:`dofbot_grasp.kinematics.servo_to_joint_deg` then converts the
resulting logical servo angles to joint radians.

Everything here is written without importing pyserial at module import time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from .kinematics import (
    N_JOINTS,
    SERVO_RANGE_DEG,
    SERVO_REVERSED_IDS,
    clamp_joints,
    joint_deg_to_servo,
    servo_to_joint_deg,
)

__all__ = [
    "SERVO_BAUDRATE",
    "SERVO_TIMEOUT_S",
    "GripperBus",
    "SerialBus",
    "SerialProtocolError",
    "ServoBus",
    "SimBus",
    "build_read_frame",
    "build_single_servo_frame",
    "build_write6_frame",
    "frame_checksum",
    "pwm_to_servo_angle",
    "servo_angle_to_pwm",
]

#: Baud rate of the Arm_Lib servo bus (Arm_Lib.py line 38: serial.Serial(...,115200)).
SERVO_BAUDRATE = 115200
#: Default serial read timeout in seconds (Arm_Lib.py line 38: timeout=.2).
SERVO_TIMEOUT_S = 0.2


@runtime_checkable
class ServoBus(Protocol):
    """Minimal arm command interface shared by sim and real hardware."""

    def write_joints(self, joints_rad, duration_s: float = 0.5) -> None:
        """Command the 5 arm joints (radians) to reach the target in ``duration_s``."""
        ...

    def read_joints(self) -> np.ndarray:
        """Return the current 5 joint angles in radians."""
        ...


@runtime_checkable
class GripperBus(Protocol):
    """Optional gripper extension implemented by both concrete buses."""

    def write_gripper(self, gripper_deg: float, duration_s: float = 0.5) -> None:
        """Command the gripper servo (0 = closed, 180 = open)."""
        ...

    def read_gripper_deg(self) -> float:
        """Return the last known gripper servo angle in degrees."""
        ...


# ---------------------------------------------------------------------------
# Frame construction (pure, testable without any serial port)
# ---------------------------------------------------------------------------


def frame_checksum(frame_without_checksum) -> int:
    """Arm_Lib checksum: ``(5 + sum(frame)) & 0xFF`` (``sum(cmd, 5)``)."""
    if isinstance(frame_without_checksum, (bytes, bytearray)):
        return int((5 + sum(frame_without_checksum)) & 0xFF)
    arr = np.asarray(frame_without_checksum, dtype=np.int64).reshape(-1)
    return int((5 + int(arr.sum())) & 0xFF)


def servo_angle_to_pwm(servo_id: int, angle_deg: float) -> int:
    """Convert a logical Arm_Lib servo angle to the bus PWM/position value.

    IDs 1-4 and 6 map 0-180 deg -> 900-3100; ID 5 maps 0-270 -> 380-3700.
    This matches Arm_Lib.py lines 92, 110, 128, 152-175 exactly.
    """
    sid = int(servo_id)
    if sid < 1 or sid > 6:
        raise ValueError(f"servo id must be 1..6, got {sid}")
    angle = float(angle_deg)
    if sid == 5:
        angle = float(np.clip(angle, 0.0, 270.0))
        return int((3700 - 380) * angle / 270.0 + 380)
    angle = float(np.clip(angle, 0.0, 180.0))
    return int((3100 - 900) * angle / 180.0 + 900)


def pwm_to_servo_angle(servo_id: int, pwm: int) -> float:
    """Inverse of :func:`servo_angle_to_pwm` (raw/logical angle, 0-180 / 0-270).

    PWM 0 (or a value below the documented floor) means the servo did not report
    a position; it must not be turned into a plausible-looking negative angle.
    """
    sid = int(servo_id)
    if int(pwm) < (380 if sid == 5 else 900):
        raise SerialProtocolError(f"servo {sid} reported no position (pwm={int(pwm)})")
    if sid == 5:
        return (270.0 - 0.0) * (float(pwm) - 380.0) / (3700.0 - 380.0)
    return (180.0 - 0.0) * (float(pwm) - 900.0) / (3100.0 - 900.0)


def _position_bytes(servo_id: int, angle_deg: float) -> tuple[int, int]:
    pwm = servo_angle_to_pwm(servo_id, angle_deg)
    return (pwm >> 8) & 0xFF, pwm & 0xFF


def build_write6_frame(servo_deg, time_ms: int) -> bytes:
    """Build the 19-byte ``Arm_serial_servo_write6`` frame.

    ``servo_deg`` holds the 6 *logical* angles (IDs 2/3/4 not yet reversed).
    The function reproduces Arm_Lib.py lines 146-188 byte for byte, including
    the internal ``180 - angle`` reversal for IDs 2/3/4 and the PWM mapping.
    """
    s = np.asarray(servo_deg, dtype=float).reshape(-1)
    if s.shape != (6,):
        raise ValueError(f"expected 6 servo angles, got shape {s.shape}")
    if np.any(s < -1e-9) or np.any(s > np.asarray(SERVO_RANGE_DEG) + 1e-9):
        raise ValueError(f"servo angles outside mechanical range: {s.tolist()}")
    time_ms = int(np.clip(int(time_ms), 0, 0xFFFF))
    values: list[int] = []
    for i, angle in enumerate(s):
        sid = i + 1
        if sid in (2, 3, 4):  # Arm_Lib.py lines 156-169
            angle = 180.0 - angle
        hi, lo = _position_bytes(sid, angle)
        values.extend([hi, lo])
    time_h, time_l = (time_ms >> 8) & 0xFF, time_ms & 0xFF
    cmd = [0xFF, 0xFC, 0x11, 0x1D] + values + [time_h, time_l]  # 18 bytes
    cmd.append(frame_checksum(cmd))
    return bytes(cmd)


def build_single_servo_frame(servo_id: int, angle_deg: float, time_ms: int) -> bytes:
    """Build one ``Arm_serial_servo_write`` frame (9 bytes).

    Mirrors Arm_Lib.py lines 100-101 (and the ID 5 / general branches with the
    same layout).  IDs 2/3/4 are reversed internally by Arm_Lib as well.
    """
    sid = int(servo_id)
    if sid < 1 or sid > 6:
        raise ValueError(f"servo id must be 1..6, got {sid}")
    angle = float(angle_deg)
    if angle < -1e-9 or angle > float(SERVO_RANGE_DEG[sid - 1]) + 1e-9:
        raise ValueError(f"angle {angle} outside servo {sid} range 0-{SERVO_RANGE_DEG[sid - 1]}")
    if sid in (2, 3, 4):
        angle = 180.0 - angle
    hi, lo = _position_bytes(sid, angle)
    time_ms = int(np.clip(int(time_ms), 0, 0xFFFF))
    cmd = [0xFF, 0xFC, 0x07, 0x10 + sid, hi, lo, (time_ms >> 8) & 0xFF, time_ms & 0xFF]
    cmd.append(frame_checksum(cmd))
    return bytes(cmd)


def build_read_frame(servo_id: int) -> bytes:
    """Build one ``Arm_serial_servo_read`` request frame (5 bytes)."""
    sid = int(servo_id)
    if sid < 1 or sid > 6:
        raise ValueError(f"servo id must be 1..6, got {sid}")
    cmd = [0xFF, 0xFC, 0x03, sid + 0x30]
    cmd.append(frame_checksum(cmd))
    return bytes(cmd)


class SerialProtocolError(RuntimeError):
    """Raised when the servo controller returns a malformed/absent reply."""


# ---------------------------------------------------------------------------
# SimBus
# ---------------------------------------------------------------------------


class SimBus:
    """In-memory bus that delegates to a callable joint-target function.

    Parameters
    ----------
    target : callable, optional
        ``target(joints_rad: np.ndarray, duration_s: float) -> None``.  Wire it
        to the simulator, e.g. ``lambda q, d: (data.qpos.__setitem__(slice(0,5), q))``.
        When omitted the bus behaves as an ideal position-controlled arm and
        simply remembers the last command.
    read : callable, optional
        ``read() -> sequence[float]`` returning the *actual* joints (e.g. from
        MuJoCo ``data.qpos``).  Defaults to the last commanded values.
    gripper_target : callable, optional
        ``gripper_target(gripper_deg: float, duration_s) -> None``.
    gripper_read : callable, optional
        ``gripper_read() -> float`` current gripper angle in degrees.
    clamp : bool
        Clamp commanded joints to the URDF limits (default True).
    """

    def __init__(
        self,
        target: Callable[[np.ndarray, float], None] | None = None,
        read: Callable[[], np.ndarray] | None = None,
        gripper_target: Callable[[float, float], None] | None = None,
        gripper_read: Callable[[], float] | None = None,
        clamp: bool = True,
    ):
        self._target = target
        self._read = read
        self._gripper_target = gripper_target
        self._gripper_read = gripper_read
        self._clamp = bool(clamp)
        self._joints = np.zeros(N_JOINTS, dtype=float)
        self._gripper_deg = 90.0
        self.commands: list[tuple[np.ndarray, float]] = []
        self.gripper_commands: list[tuple[float, float]] = []

    def write_joints(self, joints_rad, duration_s: float = 0.5) -> None:
        q = np.asarray(joints_rad, dtype=float).reshape(-1)
        if q.shape != (N_JOINTS,):
            raise ValueError(f"expected {N_JOINTS} joint values, got shape {q.shape}")
        if self._clamp:
            q = clamp_joints(q)
        self._joints = q.copy()
        self.commands.append((q.copy(), float(duration_s)))
        if self._target is not None:
            self._target(q.copy(), float(duration_s))

    def read_joints(self) -> np.ndarray:
        if self._read is not None:
            q = np.asarray(self._read(), dtype=float).reshape(-1)
            if q.shape != (N_JOINTS,):
                raise ValueError(f"read() returned shape {q.shape}, expected ({N_JOINTS},)")
            return q.copy()
        return self._joints.copy()

    def write_gripper(self, gripper_deg: float, duration_s: float = 0.5) -> None:
        angle = float(np.clip(float(gripper_deg), 0.0, 180.0))
        self._gripper_deg = angle
        self.gripper_commands.append((angle, float(duration_s)))
        if self._gripper_target is not None:
            self._gripper_target(angle, float(duration_s))

    def read_gripper_deg(self) -> float:
        if self._gripper_read is not None:
            return float(self._gripper_read())
        return float(self._gripper_deg)

    @property
    def last_joints(self) -> np.ndarray:
        """Last commanded joints (copy)."""
        return self._joints.copy()

    def contact_count(self) -> int:
        """Contact count stub: the simulator workstream may override this."""
        return 0


# ---------------------------------------------------------------------------
# SerialBus
# ---------------------------------------------------------------------------


class SerialBus:
    """Real-hardware bus implementing the Arm_Lib serial frame protocol.

    Parameters
    ----------
    port : str
        Serial device.  Arm_Lib's default is ``/dev/myserial``.
    baudrate : int
        115200 (Arm_Lib default).
    timeout : float
        Serial read timeout in seconds (Arm_Lib default 0.2).
    serial_factory : callable, optional
        ``serial_factory(port, baudrate, timeout) -> serial-like object``.
        Injected for tests; when omitted, pyserial is imported lazily and a
        clear ``RuntimeError`` is raised if it is missing.
    read_retries : int
        How many times to (re)send a read request before failing.

    Notes
    -----
    ``write_joints`` sends one 6-servo frame with the current gripper angle.
    ``read_joints`` polls servos 1..5 one by one (the hardware protocol has no
    bulk-read command) and converts the replies exactly like Arm_Lib does.
    """

    def __init__(
        self,
        port: str = "/dev/myserial",
        baudrate: int = SERVO_BAUDRATE,
        timeout: float = SERVO_TIMEOUT_S,
        serial_factory: Callable[..., object] | None = None,
        read_retries: int = 3,
        boot_settle_s: float = 1.2,
        discard_first_reply: bool = True,
    ):
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.read_retries = int(read_retries)
        # Opening the port toggles DTR, which resets the on-board controller.
        # Give it time to boot, then drop whatever the boot printed.
        self.boot_settle_s = float(boot_settle_s)
        # The real controller answers the first request after a gap with a stale
        # or empty frame; ask twice and keep the second answer.
        self.discard_first_reply = bool(discard_first_reply)
        if serial_factory is None:
            try:  # guarded: pyserial is not required for pure control tests
                import serial  # type: ignore
            except Exception as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "SerialBus requires pyserial (pip install pyserial); import serial failed"
                ) from exc
            self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        else:
            self._serial = serial_factory(self.port, self.baudrate, self.timeout)
        if self.boot_settle_s > 0:
            time.sleep(self.boot_settle_s)
        self._drain_input()
        self._gripper_deg = 90.0
        self._last_joints = np.zeros(N_JOINTS, dtype=float)
        self.write_count = 0

    # -- low level ---------------------------------------------------------
    def _drain_input(self) -> None:
        reset = getattr(self._serial, "reset_input_buffer", None)
        if callable(reset):
            reset()
            return
        try:  # pragma: no cover - older pyserial
            while self._serial.in_waiting:  # type: ignore[attr-defined]
                self._serial.read(1)
        except Exception:
            pass

    def _write(self, data: bytes) -> None:
        self._serial.write(data)
        self._serial.flush()

    def _read_exact(self, n: int) -> bytes:
        raw = self._serial.read(n)
        if raw is None or len(raw) != n:
            raise SerialProtocolError(f"short read: wanted {n} bytes, got {raw!r}")
        return bytes(raw)

    def _read_reply(self) -> tuple[int, list[int]]:
        """Read one ``[0xFF][0xFB][len][type][data...][cs]`` reply."""
        head = self._read_exact(1)[0]
        if head != 0xFF:
            raise SerialProtocolError(f"bad frame head 0x{head:02X}")
        device = self._read_exact(1)[0]
        if device != 0xFB:  # DEVICE_ID - 1
            raise SerialProtocolError(f"bad device id 0x{device:02X}")
        ext_len = self._read_exact(1)[0]
        ext_type = self._read_exact(1)[0]
        data_len = ext_len - 2
        if data_len < 1 or data_len > 32:
            raise SerialProtocolError(f"implausible length byte {ext_len}")
        payload = self._read_exact(data_len)
        computed = (ext_len + ext_type + sum(payload[:-1])) & 0xFF
        if computed != payload[-1]:
            raise SerialProtocolError(
                f"checksum mismatch: computed 0x{computed:02X}, got 0x{payload[-1]:02X}"
            )
        return ext_type, list(payload[:-1])

    # -- ServoBus API ------------------------------------------------------
    def write_joints(self, joints_rad, duration_s: float = 0.5) -> None:
        q = np.asarray(joints_rad, dtype=float).reshape(-1)
        if q.shape != (N_JOINTS,):
            raise ValueError(f"expected {N_JOINTS} joint values, got shape {q.shape}")
        servo_deg = joint_deg_to_servo(np.degrees(q), gripper_deg=self._gripper_deg)
        time_ms = int(np.clip(round(float(duration_s) * 1000.0), 0, 0xFFFF))
        self._write(build_write6_frame(servo_deg, time_ms))
        self._last_joints = q.copy()
        self.write_count += 1

    def write_servo_deg(self, servo_deg, duration_s: float = 0.5) -> None:
        """Send all six logical servo angles directly (degrees)."""
        time_ms = int(np.clip(round(float(duration_s) * 1000.0), 0, 0xFFFF))
        self._write(build_write6_frame(np.asarray(servo_deg, dtype=float), time_ms))
        self.write_count += 1

    def read_joints(self) -> np.ndarray:
        servo_deg = np.empty(6, dtype=float)
        servo_deg[5] = self._gripper_deg
        for sid in range(1, 6):
            frame = build_read_frame(sid)
            last_error: Exception | None = None
            for attempt in range(max(1, self.read_retries)):
                try:
                    # a late reply from the previous request would otherwise be
                    # mistaken for this one (one-frame lag)
                    self._drain_input()
                    self._write(frame)
                    time.sleep(0.005)
                    if attempt == 0 and self.discard_first_reply:
                        # This board's first reply after a gap is unreliable
                        # (0 or a stale value). Discard it and ask once more.
                        self._read_reply()
                        continue
                    ext_type, data = self._read_reply()
                    if ext_type != 0x0A:  # FUNC_UART_SERVO
                        raise SerialProtocolError(f"unexpected reply type 0x{ext_type:02X}")
                    if len(data) < 3:
                        raise SerialProtocolError(f"short servo reply: {data!r}")
                    echoed = data[2]
                    if echoed != (0x30 + sid):
                        # a late reply from the previous request; do not attribute it
                        raise SerialProtocolError(
                            f"reply for servo 0x{echoed:02X} while reading {sid}"
                        )
                    position = (data[0] << 8) | data[1]
                    angle = pwm_to_servo_angle(sid, position)
                    if sid in SERVO_REVERSED_IDS:  # Arm_Lib.py lines 401-402
                        angle = 180.0 - angle
                    if angle < -1.0 or angle > 271.0:
                        raise SerialProtocolError(f"servo {sid} replied out of range: {angle}")
                    servo_deg[sid - 1] = angle
                    last_error = None
                    break
                except SerialProtocolError as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
        joints = np.radians(servo_to_joint_deg(servo_deg))
        self._last_joints = joints.copy()
        return joints

    def write_gripper(self, gripper_deg: float, duration_s: float = 0.5) -> None:
        angle = float(np.clip(float(gripper_deg), 0.0, 180.0))
        time_ms = int(np.clip(round(float(duration_s) * 1000.0), 0, 0xFFFF))
        self._write(build_single_servo_frame(6, angle, time_ms))
        self._gripper_deg = angle

    def read_gripper_deg(self) -> float:
        return float(self._gripper_deg)

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            pass

    def __enter__(self) -> SerialBus:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
