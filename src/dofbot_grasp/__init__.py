"""DOFBOT SE grasp-control package.

Pure control stack for the 5-DOF DOFBOT SE desktop arm: URDF kinematics,
classical cube perception, a servo-bus abstraction (sim + real serial) and the
``home -> search -> align -> approach -> grasp -> lift -> verify`` grasp
pipeline.

Importing the package only requires NumPy: MuJoCo, OpenCV, pyserial and PyYAML
are imported lazily by the concrete sim/real/CLI adapters.

See ``docs/controller-architecture.md`` for the fixed sim/real contract and
``dofbot_grasp/docs/control-api.md`` for the module map and config keys.
"""

from . import bus, camera, kinematics, perception, pipeline
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
from .perception import CubeObservation, DetectConfig, detect_cube
from .pipeline import GraspPipeline, PipelineConfig, PipelineResult, StepRecord

__all__ = [
    "HOME_JOINTS_RAD",
    "JOINT_LIMITS",
    "N_JOINTS",
    "TCP_OFFSET_LINK5_M",
    "CubeObservation",
    "DetectConfig",
    "GraspPipeline",
    "Intrinsics",
    "PipelineConfig",
    "PipelineResult",
    "SerialBus",
    "ServoBus",
    "SimBus",
    "SimCamera",
    "StepRecord",
    "UsbCamera",
    "WristCamera",
    "bus",
    "camera",
    "detect_cube",
    "forward",
    "forward_tcp",
    "intrinsics_from_fovy",
    "inverse",
    "joint_deg_to_servo",
    "kinematics",
    "perception",
    "pipeline",
    "servo_to_joint_deg",
]

__version__ = "0.1.0"
