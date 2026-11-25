"""Bus tests: SimBus callable wiring and the exact Arm_Lib serial frame bytes."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dofbot_grasp.bus import (
    SERVO_BAUDRATE,
    SerialBus,
    SerialProtocolError,
    ServoBus,
    SimBus,
    build_read_frame,
    build_single_servo_frame,
    build_write6_frame,
    frame_checksum,
    pwm_to_servo_angle,
    servo_angle_to_pwm,
)


class FakeSerial:
    """Minimal serial.Serial stand-in: records writes, replays queued reads."""

    def __init__(self, replies: bytes = b""):
        self.written = bytearray()
        self.replies = bytearray(replies)
        self.closed = False

    def write(self, data) -> None:
        self.written.extend(bytes(data))

    def flush(self) -> None:
        pass

    def read(self, n: int = 1) -> bytes:
        out = bytes(self.replies[:n])
        del self.replies[:n]
        return out

    def close(self) -> None:
        self.closed = True


def servo_reply(position: int, servo_id: int, ext_type: int = 0x0A) -> bytes:
    """Firmware reply for a servo read: FF FB len type H L (0x30+id) checksum.

    The third data byte echoes the id byte of the read request (``0x30 + id``),
    which is what the real controller sends (verified on hardware).
    """
    high, low = (position >> 8) & 0xFF, position & 0xFF
    ext_len = 6
    payload = [high, low, 0x30 + servo_id, 0]
    payload[-1] = (ext_len + ext_type + sum(payload[:-1])) & 0xFF
    return bytes([0xFF, 0xFB, ext_len, ext_type] + payload)


# ---------------------------------------------------------------------------
# SimBus
# ---------------------------------------------------------------------------


def test_simbus_servo_protocol_and_ideal_echo() -> None:
    bus = SimBus()
    assert isinstance(bus, ServoBus)
    target = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    bus.write_joints(target, 0.25)
    np.testing.assert_allclose(bus.read_joints(), target)
    assert bus.commands[-1][1] == 0.25


def test_simbus_delegates_to_callable_target() -> None:
    calls = []

    def target(joints, duration):
        calls.append((np.asarray(joints).copy(), float(duration)))

    bus = SimBus(target=target)
    bus.write_joints([0, 0.1, 0.2, 0.3, 0.4], 0.5)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], [0, 0.1, 0.2, 0.3, 0.4])
    assert calls[0][1] == 0.5


def test_simbus_read_callable_overrides_command_echo() -> None:
    actual = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    bus = SimBus(read=lambda: actual)
    bus.write_joints([0.3, 0.3, 0.3, 0.3, 0.3])
    np.testing.assert_allclose(bus.read_joints(), actual)


def test_simbus_clamps_and_validates() -> None:
    bus = SimBus()
    bus.write_joints([9.0, -9.0, 0.0, 0.0, 9.0])
    np.testing.assert_allclose(bus.read_joints(), [1.5708, -1.5708, 0.0, 0.0, 3.1416])
    with pytest.raises(ValueError):
        bus.write_joints([0.0, 0.0])


def test_simbus_gripper() -> None:
    calls = []
    bus = SimBus(gripper_target=lambda angle, duration: calls.append((angle, duration)))
    bus.write_gripper(20.0, 0.3)
    assert calls == [(20.0, 0.3)]
    assert bus.read_gripper_deg() == pytest.approx(20.0)
    assert bus.contact_count() == 0


# ---------------------------------------------------------------------------
# Frame construction (byte-exact against Arm_Lib.py)
# ---------------------------------------------------------------------------


def _arm_lib_checksum(prefix: bytes) -> int:
    # Arm_Lib.py: checksum = sum(cmd, 5) & 0xff
    return (5 + sum(prefix)) & 0xFF


def test_frame_checksum_matches_arm_lib_receiver_rule() -> None:
    prefix = bytes(
        [
            0xFF,
            0xFC,
            0x11,
            0x1D,
            0x07,
            0xD0,
            0x07,
            0xD0,
            0x07,
            0xD0,
            0x07,
            0xD0,
            0x07,
            0xF8,
            0x07,
            0xD0,
            0x03,
            0xE8,
        ]
    )
    sender = frame_checksum(prefix)
    receiver = (0x11 + 0x1D + sum(prefix[4:])) & 0xFF
    assert sender == receiver == _arm_lib_checksum(prefix)


def test_pwm_mapping_matches_arm_lib() -> None:
    assert servo_angle_to_pwm(1, 0) == 900
    assert servo_angle_to_pwm(1, 180) == 3100
    assert servo_angle_to_pwm(5, 0) == 380
    assert servo_angle_to_pwm(5, 135) == 2040
    assert servo_angle_to_pwm(5, 270) == 3700
    for sid in (1, 2, 3, 4, 6):
        for angle in (0.0, 45.0, 90.0, 180.0):
            assert pwm_to_servo_angle(sid, servo_angle_to_pwm(sid, angle)) == pytest.approx(
                angle, abs=0.2
            )
    assert pwm_to_servo_angle(5, 3700) == pytest.approx(270.0)


def test_build_write6_frame_byte_exact() -> None:
    frame = build_write6_frame([90, 90, 90, 90, 135, 90], 1000)
    expected_prefix = bytes(
        [
            0xFF,
            0xFC,
            0x11,
            0x1D,
            0x07,
            0xD0,  # s1 2000
            0x07,
            0xD0,  # s2 180-90 -> 2000
            0x07,
            0xD0,  # s3
            0x07,
            0xD0,  # s4
            0x07,
            0xF8,  # s5 2040 (0-270 mapping)
            0x07,
            0xD0,  # s6 2000
            0x03,
            0xE8,  # time 1000 ms
        ]
    )
    assert len(frame) == 19
    assert frame[:-1] == expected_prefix
    assert frame[-1] == _arm_lib_checksum(expected_prefix)


def test_build_write6_frame_reverses_ids_2_3_4() -> None:
    # logical 0 deg for all servos: IDs 2/3/4 are sent as raw 180 -> pwm 3100,
    # while IDs 1/6 stay at pwm 900 and S5 goes to its 0-deg value 380.
    frame = build_write6_frame([0, 0, 0, 0, 0, 0], 0)
    positions = [
        (frame[4] << 8) | frame[5],
        (frame[6] << 8) | frame[7],
        (frame[8] << 8) | frame[9],
        (frame[10] << 8) | frame[11],
        (frame[12] << 8) | frame[13],
        (frame[14] << 8) | frame[15],
    ]
    assert positions[0] == 900  # s1 not reversed
    assert positions[1:4] == [3100, 3100, 3100]  # s2..s4 reversed
    assert positions[4] == 380  # s5 0 deg uses its own 380..3700 range
    assert positions[5] == 900  # s6 not reversed


def test_build_write6_frame_validates_range() -> None:
    with pytest.raises(ValueError):
        build_write6_frame([90, 90, 90, 90, 135], 1000)
    with pytest.raises(ValueError):
        build_write6_frame([90, 200, 90, 90, 135, 90], 1000)


def test_build_single_servo_frame_byte_exact() -> None:
    frame = build_single_servo_frame(2, 90, 1000)
    prefix = bytes([0xFF, 0xFC, 0x07, 0x12, 0x07, 0xD0, 0x03, 0xE8])
    assert frame == prefix + bytes([_arm_lib_checksum(prefix)])
    # ID 5 uses its own PWM mapping and function byte 0x15
    frame5 = build_single_servo_frame(5, 270, 0)
    assert frame5[3] == 0x15
    assert (frame5[4] << 8) | frame5[5] == 3700


def test_build_read_frame_byte_exact() -> None:
    frame = build_read_frame(1)
    prefix = bytes([0xFF, 0xFC, 0x03, 0x31])
    assert frame == prefix + bytes([_arm_lib_checksum(prefix)])
    assert build_read_frame(6)[3] == 0x36


# ---------------------------------------------------------------------------
# SerialBus
# ---------------------------------------------------------------------------


def _make_serial_bus():
    fake = FakeSerial()
    bus = SerialBus(
        port="/dev/fake", serial_factory=lambda p, b, t: fake, discard_first_reply=False
    )
    return bus, fake


def test_serialbus_baudrate_default() -> None:
    assert SERVO_BAUDRATE == 115200


def test_serialbus_write_joints_sends_write6_frame() -> None:
    bus, fake = _make_serial_bus()
    bus.write_joints(np.radians([0, 0, 0, 0, 0]), duration_s=0.5)
    expected = build_write6_frame([90, 90, 90, 90, 135, 90], 500)
    assert bytes(fake.written) == expected
    assert bus.write_count == 1


def test_serialbus_read_joints_parses_replies() -> None:
    # Desired logical angles [90,90,90,90,135].  The controller reports raw
    # positions; IDs 2/3/4 must be un-reversed exactly like Arm_Lib.
    positions = {
        1: servo_angle_to_pwm(1, 90),
        2: servo_angle_to_pwm(2, 180 - 90),
        3: servo_angle_to_pwm(3, 180 - 90),
        4: servo_angle_to_pwm(4, 180 - 90),
        5: servo_angle_to_pwm(5, 135),
    }
    replies = b"".join(servo_reply(positions[sid], sid) for sid in range(1, 6))
    bus, fake = _make_serial_bus()
    fake.replies = bytearray(replies)
    joints = bus.read_joints()
    np.testing.assert_allclose(joints, np.zeros(5), atol=1e-9)
    # read requests are the Arm_Lib read frames, one per servo
    expected_requests = b"".join(build_read_frame(sid) for sid in range(1, 6))
    assert bytes(fake.written) == expected_requests


def test_serialbus_read_joints_s5_full_travel() -> None:
    positions = {sid: servo_angle_to_pwm(sid, 90 if sid < 5 else 270) for sid in range(1, 6)}
    positions[2] = servo_angle_to_pwm(2, 90)
    positions[3] = servo_angle_to_pwm(3, 90)
    positions[4] = servo_angle_to_pwm(4, 90)
    replies = b"".join(servo_reply(positions[sid], sid) for sid in range(1, 6))
    bus, fake = _make_serial_bus()
    fake.replies = bytearray(replies)
    joints = bus.read_joints()
    # servo 5 logical = 270 -> joint5 = 135 deg per ANGLE_MAPPING.md
    assert joints[4] == pytest.approx(np.radians(135.0))


def test_serialbus_gripper_frame() -> None:
    bus, fake = _make_serial_bus()
    bus.write_gripper(20.0, duration_s=0.2)
    expected = build_single_servo_frame(6, 20.0, 200)
    assert bytes(fake.written) == expected
    assert bus.read_gripper_deg() == pytest.approx(20.0)


def test_serialbus_corrupt_reply_raises() -> None:
    bus, fake = _make_serial_bus()
    bad = bytearray(servo_reply(2000, 1))
    bad[-1] ^= 0xFF
    fake.replies = bad * 3
    with pytest.raises(SerialProtocolError):
        bus.read_joints()


def test_serialbus_missing_pyserial_is_reported() -> None:
    if importlib.util.find_spec("serial") is not None:
        pytest.skip("pyserial installed; the guarded-import path is not exercised")
    with pytest.raises(RuntimeError, match="pyserial"):
        SerialBus(port="/dev/definitely-not-a-port")
