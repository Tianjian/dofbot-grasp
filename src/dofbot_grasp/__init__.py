"""DOFBOT SE grasp-control package.

Control stack for the 5-DOF DOFBOT SE desktop arm: URDF kinematics, a servo-bus abstraction
(simulation and real serial), a GPU segmenter that finds the cube, and the measure-once then
grasp-by-IK sequence in :mod:`dofbot_grasp.controller`.

Importing the package only requires NumPy: MuJoCo, OpenCV, pyserial and ultralytics are
imported lazily by the concrete sim/real adapters and the segmenter.

See ``docs/controller-architecture.md`` for the fixed sim/real contract and
``docs/control-api.md`` for the module map.
"""

from . import arm, bus, camera, kinematics, safety, tracker
from .bus import SerialBus, ServoBus, SimBus
from .camera import Intrinsics, SimCamera, UsbCamera, WristCamera, intrinsics_from_fovy
from .kinematics import (
    HOME_JOINTS_RAD,
    JOINT_LIMITS,
    N_JOINTS,
    TCP_OFFSET_LINK5_M,
    forward,
    forward_tcp,
    inverse,
    joint_deg_to_servo,
    servo_to_joint_deg,
)

__all__ = [
    "HOME_JOINTS_RAD",
    "JOINT_LIMITS",
    "N_JOINTS",
    "TCP_OFFSET_LINK5_M",
    "Intrinsics",
    "SerialBus",
    "ServoBus",
    "SimBus",
    "SimCamera",
    "UsbCamera",
    "WristCamera",
    "arm",
    "bus",
    "camera",
    "forward",
    "forward_tcp",
    "intrinsics_from_fovy",
    "inverse",
    "joint_deg_to_servo",
    "kinematics",
    "safety",
    "servo_to_joint_deg",
    "tracker",
]

__version__ = "0.1.0"
