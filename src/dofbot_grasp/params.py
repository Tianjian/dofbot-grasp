"""Measured constants for the DOFBOT SE, in one place.

Every number here was measured on this arm or on its cameras; none is a guess from a
datasheet. They are grouped by what they describe so a sim2real run can be audited against
the hardware it will touch.

Sources: docs/hardware-notes.md (bus, servos, grasp), docs/sim2real.md (correspondence),
RESULTS.md (summary table).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BusParams:
    """Serial link and the two timing traps that silently corrupt data."""

    port: str = "/dev/ttyUSB0"
    baud: int = 115200  # 1 000 000 does not work on this CH340
    guard_after_write_s: float = 0.55  # a read frame inside ~0.3 s cancels the move
    settle_tolerance_deg: float = 0.35  # poll until two samples differ by less than this
    settle_timeout_s: float = 10.0
    move_time_s: float = 1.2  # the move time carried by every write6 frame


@dataclass(frozen=True)
class JointParams:
    """Servo angle <-> joint angle, and the travel each joint really has."""

    # q = u - offset for every joint; u = 90 is the vertical / neutral pose.
    offsets_deg: dict[int, float] = field(
        default_factory=lambda: {1: 90.0, 2: 90.0, 3: 90.0, 4: 90.0, 5: 135.0}
    )
    servo_range_deg: dict[int, float] = field(
        default_factory=lambda: {1: 180.0, 2: 180.0, 3: 180.0, 4: 180.0, 5: 270.0, 6: 180.0}
    )
    # Servo 6 is a claw: ~20 deg has the jaws apart, ~140 deg shuts them.
    gripper_open_deg: float = 20.0
    gripper_shut_deg: float = 140.0
    # Measured: the fingertips sit ~56 mm below the TCP point.
    #: palm 40 mm tall, fingers 40 mm long hanging below it, TCP 50 mm above link5
    fingertip_below_tcp_m: float = 0.090


@dataclass(frozen=True)
class MountParams:
    """Wrist camera pose in link4, registered to the camera modelled in link5.STL."""

    # Millimetres in link4's frame. The camera looks along the tool axis, which is
    # link4's -x; the roll is 180 deg about that axis (otherwise the image is upside down
    # and every vertical IBVS correction goes the wrong way).
    t_link4_cam_mm: tuple[float, float, float] = (-58.4, -57.6, -0.7)
    roll_deg: float = 180.0
    # The caliper numbers that first placed it (23 mm back, 50 mm lateral) had the
    # lateral offset on the wrong side of the axis.


@dataclass(frozen=True)
class CameraFiles:
    wrist_intrinsics: str = "calib/intrinsics_wrist.json"
    observer_intrinsics: str = "calib/intrinsics_observer.json"
    wrist_device: str = "/dev/video0"
    observer_device: str = "/dev/video2"


@dataclass(frozen=True)
class GraspParams:
    """The geometry that makes the grasp work, and the servoing limits."""

    # The cube rests on the table; its centre is half its edge above it.
    cube_edge_m: float = 0.030
    # The fingertip should end up at cube-centre height, i.e. 15 mm, and a little below
    # is better than above: at 20-40 mm the jaws only touch the top face and lift nothing.
    fingertip_grasp_z_m: float = 0.010
    # Image-space servoing.
    deadband_px: float = 8.0  # ~6 mm at grasp range; the jaws tolerate ~10 mm
    max_step_deg: float = 2.0
    #: per-joint cost in the centring solve; the wrist is expensive so the shoulder
    #: and elbow do the reaching instead of the arm folding up on a pinned j4
    joint_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 6.0)
    #: per-joint step cap in degrees; capping every joint at 2 deg hid the weighting
    #: entirely because each step saturated
    max_step_per_joint: tuple[float, ...] = (2.0, 2.0, 2.0, 0.5)
    gain: float = 0.5
    # Per-step joint steps used to identify the image Jacobian.
    probe_deg: float = 3.0
    # Re-identifying every iteration matters: the camera sits off the tool axis, so a
    # Jacobian measured elsewhere has the wrong gains and sometimes wrong signs.
    reidentify_every: int = 4
    # Table plane used when back-projecting a pixel to a 3-D point.
    table_z_m: float = 0.015
    #: jaw slide at fully shut in the simulated model
    sim_jaw_travel_m: float = 0.0113  # 25 mm inner face -> 13.7 mm on a 30 mm cube
    #: quarter turn of the gripper base; the fingers then descend either side of the
    #: cube instead of sweeping through its top, which is what the wrist camera sees
    default_jaw_offset_deg: float = 90.0
    #: stop descending once the cube has moved this far in the wrist frame
    cube_shift_abort_px: float = 14.0


@dataclass(frozen=True)
class SafetyParams:
    """Limits for an unattended run. The observer camera is the referee."""

    max_servo_lag_deg: float = 8.0  # beyond this something is blocking the arm
    max_tip_descent_m: float = 0.004  # per commanded descent step
    require_observer_confirmation: bool = True
    abort_on_repeated_loss: int = 3  # consecutive frames without the cube


BUS = BusParams()
JOINTS = JointParams()
MOUNT = MountParams()
CAMERAS = CameraFiles()
GRASP = GraspParams()
SAFETY = SafetyParams()


def servo_to_joint(servo_deg: dict[int, float]) -> list[float]:
    """Servo read-back (ids 1..5, degrees) -> joint angles in radians."""
    import math

    return [math.radians(servo_deg[i] - JOINTS.offsets_deg[i]) for i in range(1, 6)]


def joint_to_servo(q_rad) -> dict[int, float]:
    """Joint angles (radians) -> servo commands (degrees)."""
    import math

    return {i: float(math.degrees(q_rad[i - 1]) + JOINTS.offsets_deg[i]) for i in range(1, 6)}
