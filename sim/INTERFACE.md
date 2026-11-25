# Sim/real interface contract (fixed; both sides code against this)

Project root: `RoboHarness/dofbot_grasp/`

## MuJoCo model
- Path: `sim/dofbot_se.xml` (MJCF, meters, radians)
- Joints (revolute, hinge): `joint1` `joint2` `joint3` `joint4` `joint5`
- Gripper: `gripper_left`, `gripper_right` (prismatic, meters; ctrl 0.010 = open, -0.002 = pinch)
- TCP site: `tcp` at `(0,0,0.050)` in link5 (invisible)
- Actuators (position servos): `act_joint1`..`act_joint5`, `act_gripper`
- Camera: `wrist_cam` on `link5`, offset `(0.035, 0, 0.010)`, looking along **+Z of link5**
  (down the tool axis), fovy 75. Decision rule: the cube must be >=3% of the frame at pre-grasp.
- Scene bodies: `base`, `table`, `cube` (30 mm), `wrist_cam`
- qpos layout for the arm: joints 1..5 then gripper; gripper ctrl is a single target
  applied to both fingers (one position actuator per finger driven together).

## Python package `src/dofbot_grasp/`
- `kinematics.py`: `forward(joints_rad) -> 4x4 pose`, `inverse(pose, seed) -> joints_rad`,
  using the same chain as the MJCF.
- `bus.py`: `ServoBus` protocol with `write_joints(joints_rad, duration_s)` and
  `read_joints()`; implementations `SimBus` (MuJoCo) and `SerialBus` (Arm_Lib frames).
- `camera.py`: `WristCamera` protocol: `read() -> BGR ndarray`; `SimCamera`, `UsbCamera`.
- `perception.py`: `detect_cube(bgr) -> CubeObservation | None`
  (bbox, pixel center, estimated distance, bearing), cube edge length is a parameter.
- `pipeline.py`: `GraspPipeline(bus, camera, cfg)` with named steps matching the
  roboharness TaskProtocol phases: `home search align approach grasp lift verify`.
- `cli.py`: `python -m dofbot_grasp.cli sim|real [--config ...]`

## roboharness phases (TaskProtocol name="dofbot_grasp")
`home -> search -> align -> approach -> grasp -> lift -> verify`
Checkpoint metrics must include: `cube_height_mm`, `gripper_cube_distance_mm`,
`contact_count`, `alignment_error_px`.
