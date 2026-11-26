# DOFBOT SE grasp control API

Control stack for the 5-DOF DOFBOT SE desktop arm, implementing the fixed
sim/real contract in [`docs/controller-architecture.md`](controller-architecture.md).

* Pure modules (`kinematics`, `perception`, `camera`, `bus`, `pipeline`) import
  **NumPy + stdlib only**. OpenCV, pyserial, MuJoCo and PyYAML are imported
  lazily by the concrete hardware/sim/CLI adapters, so the unit tests run on a
  bare Python + NumPy environment.
* Target Python: 3.10+ (validated with the repo `.venv`, Python 3.12).
* FK was cross-checked against the actual MuJoCo model: `tcp` site position
  differs by < 4.1e-5 m and rotation by 1.1e-3 rad over random joint vectors;
  the wrist-camera offset from `cam_xpos` matches `(0.035, 0, 0.010)` to 1e-5 m.

```bash
# from RoboHarness/
pytest -q
# 71 passed
```

## Package layout

```
dofbot-grasp/
├── src/dofbot_grasp/
│   ├── __init__.py      re-exports the public API
│   ├── kinematics.py    URDF FK, DLS IK, servo<->joint degree mapping
│   ├── camera.py        WristCamera protocol, UsbCamera, SimCamera, intrinsics
│   ├── perception.py    HSV/adaptive threshold + components + minAreaRect
│   ├── bus.py           ServoBus protocol, SimBus, SerialBus (Arm_Lib frames)
│   ├── pipeline.py      7-step grasp state machine + config/records/metrics
│   └── cli.py           python -m dofbot_grasp.cli sim|real --config x.yaml
├── tests/               pytest suite (70 tests, mocks only)
├── docs/control-api.md  this file
├── sim/                 MuJoCo model, grasp trial, recording
└── harness/             roboharness TaskProtocol wiring (owned by harness workstream)
```

## Module map

### `dofbot_grasp.kinematics`

| API | Description |
| --- | --- |
| `forward(joints_rad) -> (4,4) ndarray` | link5/EE pose in the base frame; chain transcribed from `prior/arm_cam_control/dofbot_info/urdf/dofbot.urdf` and validated against `sim/dofbot_se.xml`. |
| `forward_tcp(joints_rad) -> (4,4)` | grasp-point pose: link5 shifted 50 mm along the tool axis (`TCP_OFFSET_LINK5_M`, matching the MJCF `tcp` site). |
| `forward_all(joints_rad) -> list[(4,4)]` | `[base, link1, …, link5]` frames (for a body/camera mount). |
| `inverse(pose, seed, *, tol=1e-4, rot_tol=1e-3, max_iters=400, damping=1e-3, orientation_weight=0.1) -> joints` | damped least squares (Levenberg–Marquardt) full-pose IK; every iterate is clamped to `JOINT_LIMITS`. |
| `inverse_position(position, seed, *, tol=1e-4, max_iters=300, restarts=12) -> joints` | position-only IK used by the grasp moves; deterministic multi-start when the seed is far. |
| `clamp_joints`, `clamp_servo_deg` | URDF / mechanical range clamping. |
| `servo_to_joint_deg(servo_deg)`, `joint_deg_to_servo(joint_deg, gripper_deg=90)` | logical Arm_Lib servo degrees ↔ joint degrees (IDs 1–4 `s-90`, S5 `s-135`, gripper passthrough). Batch (…, 5/6) arrays supported. |
| `raw_servo_to_joint_deg`, `joint_deg_to_raw_servo` | physical servo direction ↔ joint, showing the ID 2/3/4 `180-angle` reversal. |
| `servo_to_joint_rad`, `joint_rad_to_servo` | radian convenience wrappers. |
| `rpy_to_matrix`, `rot_log` | URDF fixed-axis rpy and rotation-vector helpers. |

FK/IK facts (URDF limits: joints 1–4 ±1.5708 rad, joint5 −1.5708..3.1416 rad):

* home (all joints 0): EE position ≈ **(0.0000009, 0.0011039, 0.3470436) m**,
  rotation ≈ identity — the arm stands straight up.
* `q = [0, −π/2, 0, 0, 0]`: EE ≈ (0, 0.2397, 0.1065) m (arm folded forward).
* `q = [π/2, 0, 0, 0, 0]`: EE ≈ (0.0019, 0.0030, 0.3470) m and the EE frame is
  yawed 90° (EE +X ≈ base +Y).
* IK full-pose round-trip from seeds perturbed by N(0, 0.2 rad):
  max position error **0.098 mm** (mean 0.027 mm), max orientation error
  2.6 × 10⁻⁴ rad over 30 random reachable poses.
* Position-only IK with fully random far seeds: max **0.097 mm** over 30
  random reachable points.

### `dofbot_grasp.camera`

| API | Description |
| --- | --- |
| `WristCamera` | protocol: `read() -> BGR uint8 ndarray`. |
| `UsbCamera(index=0, width=640, height=480)` | `cv2.VideoCapture` wrapper; `cv2` imported lazily (raises `RuntimeError` if missing). `.intrinsics` defaults to a centred pinhole; calibrate for real use. |
| `SimCamera(frame_fn=None, intrinsics=None)` | placeholder the sim workstream wires (`camera.wire(lambda: frame)`); `read()` raises `RuntimeError` until wired. |
| `Intrinsics(fx, fy, cx, cy, width, height)` | pinhole intrinsics (`as_matrix()`, `scaled()`). |
| `wrist_camera_to_ee()`, `ee_to_wrist_camera()` | fixed 4×4 mount per the contract: offset `(0.035, 0, 0.010)` in link5, optical +Z = link5 +Z (down the tool axis), image x = +X of link5, image y = +Y of link5. Cross-checked against MuJoCo `cam_xpos`/`cam_xmat`. |
| `intrinsics_from_fovy(fovy_deg, width, height)` | MJCF fovy → pinhole intrinsics (fovy 75° at 480×640 → fx = fy ≈ 313 px). |
| `project_point`, `unproject_pixel` | pinhole helpers. |

> The mount matches `sim/dofbot_se.xml`
> (`<camera ... pos="0.035 0 0.010" quat="0 1 0 0" fovy="75"/>`). Override with
> `camera_to_ee` in the pipeline config if a real camera is mounted differently.

### `dofbot_grasp.perception`

Classical detection, no deep learning and no OpenCV:

`BGR → HSV mask (or Otsu adaptive fallback) → 3×3 open/close → connected
components (flood fill) → convex hull → rotating-calipers min-area rect`,
then pick the most square-like component.

| API | Description |
| --- | --- |
| `detect_cube(bgr, cube_size_m=0.030, intrinsics=None, cfg=DetectConfig(), **overrides) -> CubeObservation \| None` | returns `None` when no plausible cube. |
| `CubeObservation` | `bbox`, `center_px`, `width_px`, `distance_m`, `bearing_rad`, `angle_deg`, `mask_area_px`, `rect_size_px`, `rectangularity`. |
| `cube_pose_estimate(obs, intrinsics, camera_to_base=None, cube_size_m=0.03)` | 4×4 cube pose: back-projected pixel centre at the pinhole depth, yaw from the rect angle; optional transform to base frame. |
| `cube_position_m(obs, intrinsics, camera_to_base=None)` | cube centre (camera or base frame). |
| `bgr_to_hsv`, `connected_components`, `convex_hull`, `min_area_rect` | reusable NumPy building blocks. |

Geometry: a 30 mm cube at depth `Z` projects to `width_px = f_px*0.030/Z`, so
`Z = f_px*0.030/width_px` (the "face-on" approximation; a strongly rotated cube
reads slightly too close). Bearing = `atan2(u - cx, f_px)`. Synthetic tests at
0.20 / 0.35 / 0.50 m stay within **< 3 %** distance error (requirement < 10 %).

`DetectConfig` keys: `cube_size_m`, `f_px`, `cx`, `cy`, `hsv_lower`,
`hsv_upper` (OpenCV-convention HSV, default orange), `min_area_px`,
`max_area_frac`, `min_rectangularity`, `max_aspect`, `morphology`,
`adaptive_fallback`, `adaptive_prefer` (`"smaller"`/`"brighter"`),
`max_mask_frac`.

### `dofbot_grasp.bus`

| API | Description |
| --- | --- |
| `ServoBus` | protocol: `write_joints(joints_rad, duration_s)`, `read_joints() -> joints_rad`. |
| `GripperBus` | optional extension: `write_gripper(deg)`, `read_gripper_deg()`. |
| `SimBus(target=None, read=None, gripper_target=None, gripper_read=None, clamp=True)` | delegates commands to a callable joint-target function and reads either from a callable or echoes the last command. `contact_count()` returns 0 (override for force/contact data). |
| `SerialBus(port="/dev/myserial", baudrate=115200, timeout=0.2, serial_factory=None)` | Arm_Lib wire protocol; pyserial imported lazily, injectable factory for tests. |
| `build_write6_frame`, `build_single_servo_frame`, `build_read_frame` | pure frame builders (byte-exact, unit-tested). |
| `servo_angle_to_pwm`, `pwm_to_servo_angle` | 900–3100 µs (IDs 1–4/6) / 380–3700 µs (ID 5). |

Wire protocol (source: `prior/arm_cam_control/Arm_Lib.py`, lines 146–191,
363–405; the full citation lives in the `bus.py` module docstring):

```
write 6 servos (19 bytes):
  FF FC 11 1D S1H S1L S2H S2L S3H S3L S4H S4L S5H S5L S6H S6L TH TL CS
  CS = (5 + sum(previous 18 bytes)) & 0xFF        # Arm_Lib: sum(cmd, 5) & 0xff
  positions: 900 + 2200*angle/180  (IDs 1-4,6),  380 + 3320*angle/270  (ID 5)
  IDs 2/3/4 are inverted inside Arm_Lib: angle -> 180 - angle

write one servo (9 bytes):
  FF FC 07 (0x10+id) PH PL TH TL CS

read one servo (5 bytes):   FF FC 03 (id+0x30) CS
reply:                      FF FB len type H L id CS   (type 0x0A)
```

`Arm_serial_servo_read` already un-reverses IDs 2/3/4, so `servo_to_joint_deg`
only subtracts the 90/135 degree centre — applying `90 - s` again is the
documented "double reverse" bug.

### `dofbot_grasp.pipeline`

`GraspPipeline(bus, camera, cfg=PipelineConfig(), detector=detect_cube,
clock=time.monotonic, sleep_fn=time.sleep)` runs the roboharness phases:

```
home → search → align → approach → grasp → lift → verify
```

* `search` sweeps `search_yaw_deg` base-yaw offsets around home until
  `detect_cube` returns an observation; **no cube → abort at `search`**.
* `align` servos the base yaw from the measured bearing until the cube centre
  is within `align_tolerance_px` of the target column.
* `approach` estimates the cube in the base frame
  (`FK(joints) @ camera_to_ee @ T_cam_cube`), solves position-only IK so the
  **TCP** (link5 origin + `tool_offset_m` along the tool axis) ends
  `approach_standoff_m` before the cube, and checks the FK position error.
* `grasp` fine-moves the TCP to `grasp_standoff_m` from the cube centre and
  closes the gripper (if the bus implements `write_gripper`).
* `lift` raises the EE by `lift_height_m`; `verify` checks the rise, optional
  visibility and the TCP-to-cube distance, then fills the checkpoint metrics.

Every step returns a `StepRecord(name, joints, observation, elapsed_s, success,
message, extra)`; an expected mission failure returns
`PipelineResult(success=False, aborted_step=..., reason=...)` instead of
raising. Unexpected programming errors still raise.

Checkpoint metrics (`PipelineResult.metrics`):

| Metric | Meaning |
| --- | --- |
| `cube_height_mm` | estimated cube-centre height above `table_height_m` after the lift (detected if visible, else last known position + lift), in mm. |
| `gripper_cube_distance_mm` | final TCP-to-cube distance (detected if visible, else last known position + lift). |
| `contact_count` | `bus.contact_count()` when available, else 0 (sim may override). |
| `alignment_error_px` | last horizontal pixel error from `align`. |

`PipelineConfig` keys (all optional; angles are logical servo degrees):

| Group | Keys |
| --- | --- |
| home/search | `home_servo_deg`, `search_yaw_deg`, `open_gripper_deg` |
| timing | `move_duration_s`, `settle_s`, `gripper_settle_s` |
| perception | `cube_size_m`, `f_px`, `cx`, `cy`, `detect` (a `DetectConfig` mapping), `camera_to_ee` (4×4) |
| align | `target_u_px`, `align_tolerance_px`, `align_gain`, `align_max_iters`, `align_max_step_rad` |
| approach/grasp | `approach_standoff_m`, `approach_tolerance_m`, `grasp_standoff_m`, `grasp_tolerance_m`, `tool_offset_m`, `grasp_gripper_deg`, `gripper_open_deg` |
| lift/verify | `lift_height_m`, `verify_min_lift_m`, `table_height_m`, `verify_require_cube_visible`, `verify_max_cube_distance_m` |
| IK | `ik_orientation_weight`, `ik_tol_m`, `ik_max_iters` |
| error policy | `strict_camera` |

`PipelineConfig.from_dict(mapping)` validates unknown keys,
`PipelineConfig.from_yaml(path)` reads the `pipeline` section. Default home pose
is `home_servo_deg = (90, 75, 31, 5, 86, 90)` — a look-down pose with the
tool-axis wrist camera over the table, i.e. joint radians
`(0, -0.2618, -1.0297, -1.4835, -0.8552)`, EE position ≈ `(0, 0.1295, 0.1406) m`
and camera at ≈ `(0.0230, 0.1578, 0.1407) m`. With that pose the home camera
ray hits a table-top cube at ≈ `(0.023, 0.206, 0.015) m`. The mock pipeline run
reports `cube_height_mm ≈ 64`, `gripper_cube_distance_mm ≈ 6.6`,
`alignment_error_px ≈ -0.5`, i.e. inside the harness contract
(`cube_height ≥ 20 mm`, `gripper_cube_distance ≤ 15 mm`).

## How to fake a bus and camera

A bus is anything with `write_joints`/`read_joints`. For a pure-logic test:

```python
import numpy as np
from dofbot_grasp.bus import SimBus
from dofbot_grasp.pipeline import GraspPipeline, PipelineConfig

bus = SimBus()                      # ideal position-controlled arm
obs = []                            # optional: inspect commands
cfg = PipelineConfig(search_yaw_deg=(0.0,), move_duration_s=0.0)
```

A camera is anything with `read() -> BGR uint8 frame`. The test suite wires a
scene that renders the cube from the bus's current joints, so the pipeline
closes the loop through real geometry instead of canned observations
(`tests/helpers.py`):

```python
class CubeScene:
    def __init__(self, cube_world, intrinsics, camera_to_ee):
        self.cube_world = cube_world
        ...
    def render(self, joints_rad):
        t_cam = forward(joints_rad) @ self.camera_to_ee
        p_cam = np.linalg.inv(t_cam) @ np.append(self.cube_world, 1.0)
        u = intr.fx * p_cam[0] / p_cam[2] + intr.cx
        v = intr.fy * p_cam[1] / p_cam[2] + intr.cy
        side = intr.fx * 0.030 / p_cam[2]
        ...  # draw a `side`-pixel bright square on a plain background

class SceneCamera:
    intrinsics = ...
    def read(self):
        return scene.render(bus.read_joints())
```

For MuJoCo (`SimBus` is deliberately agnostic to the sim workstream's model):

```python
bus = SimBus(
    target=lambda q, d: data.ctrl.__setitem__(slice(0, 5), q),
    read=lambda: np.asarray(data.qpos[0:5]),
)
camera = SimCamera(frame_fn=lambda: render_wrist_frame(model, data),
                   intrinsics=Intrinsics(fx=..., fy=..., cx=..., cy=...))
```

Or pass a detector stub:
`GraspPipeline(bus, camera, cfg, detector=lambda frame, **kw: fixed_observation)`.

## CLI

```bash
python -m dofbot_grasp.cli sim  --config config.yaml
python -m dofbot_grasp.cli real --config config.yaml --json
```

Exit codes: `0` verified grasp, `1` mission aborted (e.g. missing cube), `2`
configuration / hardware error. Example config:

```yaml
pipeline:
  home_servo_deg: [90, 75, 31, 5, 86, 90]
  search_yaw_deg: [0, -20, 20, -40, 40]
  cube_size_m: 0.030
  f_px: 600
  align_tolerance_px: 25
  approach_standoff_m: 0.06
  tool_offset_m: 0.050
  grasp_gripper_deg: 20
  lift_height_m: 0.05
  table_height_m: 0.0
bus:                       # real mode
  port: /dev/myserial
  baudrate: 115200
camera:                    # real mode
  index: 0
  width: 640
  height: 480
sim:                       # sim mode; module exposing make_bus()/make_camera()
  adapter: my_project.sim_adapter
```

Without a `sim.adapter` the CLI uses an ideal `SimBus` plus an unwired
`SimCamera`; the pipeline aborts cleanly at `search` with a clear message.

## Assumptions and limitations

1. **S5 mapping.** `ANGLE_MAPPING.md` is internally inconsistent: its formula
   `servo/270*270 - 135` and mid-point check (servo 135 → joint 0) imply
   joint5 ∈ [−135°, +135°], while its endpoint examples claim servo 0 → −90°
   and servo 270 → 180°. This package follows the formula and mid-point check,
   so joint5 commands above +135° saturate at servo 270° (the URDF upper limit
   of +180° is not reachable through the servo). IDs 1–4 map exactly
   `joint = servo - 90`.
2. **Camera and TCP mounts.** Both come from the fixed sim contract: camera at
   `(0.035, 0, 0.010)` in link5 looking down +Z, TCP at `(0, 0, 0.050)` in
   link5. `forward()` returns the link5 frame; `forward_tcp()` / the pipeline's
   `tool_offset_m` add the 50 mm tool offset. Override `camera_to_ee` if a real
   camera is mounted differently.
3. **Cube distance.** `Z = f_px * size / width_px` assumes a face-on cube;
   strong rotation biases the estimate by the projected-width ratio.
4. **Verify is kinematic.** There is no force/contact sensor in the sim/real
   contract, so `verify` checks lift height, optional post-lift visibility and
   TCP-to-cube distance; `contact_count` is whatever the bus reports (0 for the
   default `SimBus`). The MuJoCo wedge measures real contacts itself.
5. **IK restarts.** `inverse_position` uses deterministic multi-start seeds
   when the supplied seed fails; the pipeline always seeds from the current
   joints, so its normal path is a single fast DLS solve.
6. **No deep learning.** Perception is a classical colour/adaptive pipeline;
   a cube whose colour matches the table may need `hsv_lower`/`hsv_upper`
   tuning or the adaptive fallback.
7. **Servo 5 endpoints.** Only joint5 ∈ [−135°, +135°] is reachable through the
   servo (see item 1); the URDF's +180° upper limit is not commandable.
