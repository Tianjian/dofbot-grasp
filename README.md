# dofbot-grasp

**Wrist-camera grasp controller for the Yahboom DOFBOT SE — validated in MuJoCo before it touches hardware.**

A 30 mm cube on a table, seen by a USB camera mounted on the wrist, grasped by a
5-DOF arm whose servos are driven over a serial line. The same perception,
kinematics and state machine run in simulation and on the robot; only the camera
and the servo bus are swapped.

[![CI](https://github.com/Tianjian/dofbot-grasp/actions/workflows/ci.yml/badge.svg)](https://github.com/Tianjian/dofbot-grasp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

> **Demo** — [`sim/controller_sim.mp4`](sim/controller_sim.mp4): four cube positions,
> one after another, wrist camera on the left and observer camera on the right. Each
> frame carries the phase and, underneath, the fingertip and cube coordinates with the
> distance between them -- the number that decides whether the jaws straddle the cube or
> shove it. All four grasp, lifting **49-53 mm**.
>
>
> **All measured numbers, real and simulated, are in [`RESULTS.md`](RESULTS.md).**

---

## What it does

1. **Perceive** — find a 30 mm cube in a 640×480 wrist-camera frame. A colour threshold
   seeds SAM2, which returns the whole silhouette including the pale top face that a
   saturation mask misses (15,898 px against the mask's 14,844 on the same frame). The
   tracker costs 134-863 ms per frame on an RTX 2060.
2. **Reach** — solve the 5-DOF arm's inverse kinematics with damped least squares
   against a URDF-derived chain, then map joint radians to servo degrees.
3. **Grasp** — run a seven-phase state machine (`home → search → align → approach →
   grasp → lift → verify`) and emit joint targets to six servos over a
   `115200` baud serial link.

Perception optionally uses a GPU through SAM2; the servoing and grasp logic are
ordinary Python. No cloud call, no ROS.

## Verified in simulation

| Check | Result |
|
## Status

| layer | state |
|---|---|
| Real arm — bus, joint directions, zero pose, limits | verified, read-back stable to 0.16 deg |
| Real arm — grasp | one successful grasp, cube lifted clear and held |
| Cameras — intrinsics | wrist cx 269.8 (50 px off centre), observer RMS 0.65 px |
| Cameras — mount | wrist registered to the camera modelled in `link5.STL`; observer matched to the real cell including the black backdrop |
| Simulation ↔ arm | forward kinematics **0.0 mm** over seven poses; static error 1.53 deg vs ~1.5; settling 1.6 s vs 1.5-2.0 |
| Simulation — grasp | **4 of 4 cube positions grasped**, lifting 49-53 mm |
| CI gate | grasp rate is a **blocking** check at 0.75, currently 1.00 |
| Simulation → real | **not yet run** |

Details: [`RESULTS.md`](RESULTS.md) for the numbers,
[`docs/sim2real.md`](docs/sim2real.md) for the servo/joint correspondence,
[`docs/ibvs.md`](docs/ibvs.md) for the servoing pipeline and the faults found building it.

---|---|
| Reference motion lifts the cube | **107.99 mm** with **8** finger contacts |
| roboharness wedge, normal run | **PASS**, lifted **106.21 mm**, 4 contacts, `roboharness gate` → **exit 0** |
| roboharness wedge, fingers never close | **FAIL** at `lift`, lifted **0.0 mm**, 0 contacts, gate → **exit 1** |
| Controller FK vs MuJoCo TCP | **< 0.05 mm**, 1.1e-3 rad |
| Inverse kinematics round-trip | **≤ 0.098 mm**, 2.6e-4 rad |
| Unit tests | **71 passed** |

Details and exact commands: [`harness/EVIDENCE.md`](harness/EVIDENCE.md).

## Quick start

```bash
git clone https://github.com/Tianjian/dofbot-grasp.git
cd dofbot-grasp
python -m pip install -e ".[dev,sim]"

# unit tests (no simulator, no hardware)
pytest -q

# reference grasp in MuJoCo (headless)

# re-record the demo video
python -m pip install -e ".[demo]"
```

### Verify with roboharness (optional but recommended)

The wedge turns the grasp into a decision your CI can refuse. It needs the
roboharness fork that ships the `gate` command:

```bash
python -m pip install "roboharness @ git+https://github.com/Tianjian/roboharness.git"

MUJOCO_GL=egl python harness/run_wedge.py --out /tmp/wedge_good
MUJOCO_GL=egl python harness/run_wedge.py --out /tmp/wedge_fault --fault

roboharness gate /tmp/wedge_good/approval_report.json    # exit 0
roboharness gate /tmp/wedge_fault/approval_report.json   # exit 1
```

## Architecture

```
                 ┌──────────────── shared by sim and real ────────────────┐
USB camera ─────►│ perception.detect_cube  → 30 mm cube bbox / range      │
 (or MuJoCo)     │            │                                           │
                 │            ▼                                           │
                 │ pipeline.GraspPipeline (7 phases, abort on no target)  │
                 │            │ TCP pose                                  │
                 │            ▼                                           │
                 │ kinematics.inverse (DLS + limits) → servo degrees      │
                 └────────────┼───────────────────────────────────────────┘
                              ▼
                    bus.SerialBus ──► 6 servos (115200, Arm_Lib frames)
```

| Layer | Simulation | Real robot |
|---|---|---|
| camera | MuJoCo `wrist_cam` render | `cv2.VideoCapture` (USB) |
| bus | joint targets into MuJoCo | `pyserial` frames |
| perception / kinematics / pipeline | **identical** | **identical** |

The MuJoCo model is generated from the real URDF chain — joint origins, axes and
limits are copied verbatim, so the forward kinematics agree to under 0.05 mm. That
is what makes the simulator a useful gate rather than a separate toy.

## Layout

```
src/dofbot_grasp/     kinematics, perception, camera, bus, pipeline, cli
tests/                71 unit tests (numpy only; no mujoco/cv2/serial needed)
sim/                  MJCF derived from the URDF, reference grasp, video recorder
harness/              roboharness wedge + the evidence it produced
docs/
  deploy-runbook.md   real-robot bring-up: safety, serial, mapping, camera,
                      calibration, timing, acceptance, abort
  control-api.md      module map, config keys, how to fake a bus/camera
  prior-art.md        line-level analysis of the two earlier repos
  hardware-notes.md   values measured on a real DOFBOT SE (serial, polarity,
                      image->joint gains, gripper, what is still unmeasured)
```

## Hardware notes (read before wiring anything)

- **Serial frame** for six servos is 19 bytes: `FF FC 11 1D v1H v1L … v6H v6L tH tL cs`
  at `115200`; PWM is `900 + 2200·u/180` (servo 5: `380 + 3320·u/270`).
- **Servos 2/3/4 are already reversed inside the vendor driver** (both write and
  read). Flipping them again in your controller silently commands `-q` — this
  happened in one of the earlier repos.
- **Servo 5 has no trustworthy zero.** The old documentation is self-inconsistent
  and a `+44°` offset in three files pushes the range outside the URDF limits.
  Re-derive it on the bench.
- **There is no usable motion feedback.** Treat the arm as open-loop and the camera
  as the only sensor; wait out each move instead of polling.

Full detail, with file/line citations: [`docs/prior-art.md`](docs/prior-art.md).

## Deploying to the robot

[`docs/deploy-runbook.md`](docs/deploy-runbook.md) is the ordered bring-up:
bench safety and a ±10° direction check per joint, serial port discovery, the
servo↔joint mapping and its traps, camera intrinsics and exposure locking, a
9-point hand-eye calibration (target residual < 3 mm), the per-phase timing table,
acceptance tests, and abort/rollback rules.

## Status and limitations

- Simulation is validated; the real arm is now driven over serial and the joint
  polarity, look pose, image→joint gains and gripper direction are measured
  ([`docs/hardware-notes.md`](docs/hardware-notes.md)). **The grasp itself has not
  been executed yet**: the camera→TCP offset is uncalibrated. Parameters marked
  `[待实测]` in the runbook still need a bench measurement.
- The reference motion is scripted, not visually servoed. A closed loop on the
  cube's image offset is the next step for hardware.
- `contact_count` in simulation comes from MuJoCo contacts; on hardware it comes
  from the bus (and there is no force sensing).
- Perception assumes a face-on cube for the pinhole range estimate; a table-plane
  homography is the planned upgrade.

## 中文说明

这是 DOFBOT SE 桌面机械臂的**手腕相机抓取控制器**：USB 相机看 30 mm 木块 → 本机算力做
检测与逆解 → 串口把 6 个舵机角度下发。核心设计是**仿真与真机共用同一套感知/运动学/状态机**，
只替换相机与总线两个实现；MuJoCo 模型直接从真实 URDF 生成，因此仿真里验证过的阈值搬到真机含义不变。

- 仿真结果：抬起 **107.99 mm**、8 个接触；roboharness 验证 **PASS/exit 0**，故障注入 **FAIL/exit 1**
- 演示视频：[`sim/controller_sim.mp4`](sim/controller_sim.mp4)
- 真机部署：[`docs/deploy-runbook.md`](docs/deploy-runbook.md)（含串口帧格式、S5 零位、手眼标定、时序与回退）
- 旧仓库分析：[`docs/prior-art.md`](docs/prior-art.md)

## Credits

The URDF chain, the `Arm_Lib` serial protocol and the hardware knowledge come from
[`Tianjian/arm_cam_control`](https://github.com/Tianjian/arm_cam_control); the camera
service design from [`Tianjian/face_tracker`](https://github.com/Tianjian/face_tracker).
Validation harness: [`Tianjian/roboharness`](https://github.com/Tianjian/roboharness).

## License

MIT — see [LICENSE](LICENSE).
