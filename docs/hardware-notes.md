# Hardware notes: what was measured on a real DOFBOT SE

Everything below was measured on the machine, not copied from the vendor docs.
It supersedes any `[待实测]` marker in `deploy-runbook.md` for these items.

## Serial

- Port `/dev/ttyUSB0`, CH340 (`1a86:7523`), **115200** works; 1 000 000 does not.
- Servos 1–6 all answer. Reply frame: `FF FB 06 0A <posH> <posL> <0x30+id> <cs>`.
- **The first reply after a gap is unreliable** (empty or stale). Ask twice, keep
  the second answer — that is `SerialBus(discard_first_reply=True)`, the default.
- Opening the port toggles DTR and **resets the controller**; wait ~1.2 s
  (`boot_settle_s`) before the first request.
- Position command → measured position has ~1–3° of lag/backlash. Closed-loop
  stepping (command, read, correct) is required for accurate placement.

## Joint polarity: no inversion needed

Servo logical angle `u` maps to joint angle exactly as `kinematics.servo_to_joint_deg`
says (`q = u − 90` for 1–4, `q5 = u − 135`). Evidence, wrist camera used as an
orientation sensor (it looks along the tool axis):

| command | measured | wrist camera sees | FK tool axis |
|---|---|---|---|
| `u4 = 0` (q4 = −90°) | 0.1° | open office (people, plants) | `(0, +1, 0)` → +Y |
| `u4 = 180` (q4 = +90°) | 179.8° | red cubicle partition | `(0, −1, 0)` → −Y |
| `u = [90, 95, 30, 5, 135]` | — | desk mat + cube | `(0, +0.64, −0.77)` |

Observation and FK agree in every case, so the vendor's internal 180° reversal for
S2/S3/S4 (already applied inside `Arm_Lib` and mirrored in `bus.build_write6_frame`)
is correct and **must not be flipped again**.

## Geometry: "tool straight down" barely exists

With the 50 mm tool offset and 83/83/74 mm links, a pose with the tool axis exactly
vertical only exists with the TCP at **z ≈ 5 cm** (the arm has to fold right over).
Enumerating tool-down poses subject to the real limits (|q2..q4| ≤ 90°) gives
TCP heights of 0.05–0.06 m only.

Consequence: a *look* pose is necessarily tilted. A usable one:

```
u = [90, 95, 30, 5, 135]   (q ≈ [0, 5, −60, −85, 0])
TCP ≈ (0.00, 0.13, 0.11) m, tool axis (0, 0.42, −0.91)
```

The wrist camera then sees the desk mat and the cube at ~8 cm.

## Perception

- USB camera `/dev/video0`, 640×480 @ 30 fps (a second camera, `/dev/video2`, is
  used as the external observer).
- The yellow cube has mean HSV ≈ **(39, 151, 176)**; a generous threshold
  `H 18–40, S ≥ 70, V ≥ 70` isolates it cleanly on the patterned desk mat.
- At the look pose the cube measures ~200 px wide, i.e. ≈8 cm from the camera
  under the pinhole model.

## Image → joint gains (measured, one probe each)

| joint | effect on the cube's pixel position |
|---|---|
| `j1` (base yaw) | **+9.4 px/deg in x**, +1.5 px/deg in y |
| `j3` (elbow) | **+22 px/deg in y**, −1.4 px/deg in x |

These are the numbers a pixel-error servo loop needs; they replace guessing.

## Gripper

- Servo 6: **170° = open, 10° = closed**, 46° is the pose it was shipped in.
- No force sensing; the only closing signal is the visual one.

## The wrist camera is NOT on the tool-roll link

The simulation contract puts `wrist_cam` on `link5` (so it rolls with `j5`). On the
real arm it does not: commanding `u5` from 225° to 45° moves the servo (read back
both ways) but the wrist image is **identical** (normalised correlation 0.90 at 0°
rotation, -0.05 at 90°). The camera is therefore mounted on the link below the
roll — it follows `j4` (wrist pitch) but not `j5`.

Consequences, both real:

- The simulation should mount the camera on `link4`, otherwise the model promises a
  view the hardware never produces.
- The camera's optical axis is **not** the tool axis. Anything that reasons "the
  camera points where the gripper points" (including a wrist-roll calibration)
  is wrong on this machine.

This also explains the failed jaw-alignment idea: the camera looks past the fingers
rather than between them, and rolling the wrist does not orbit the view.

## Still unmeasured

- Camera intrinsics (`fx, fy, cx, cy, distortion`) — no calibration target used yet.
- Camera-to-TCP offset on the **real** bracket. The frame is mounted ~35 mm off the
  tool axis, but the exact value and direction have not been measured, so centring
  the cube in the image does **not** put the TCP over the cube. This is the one
  number blocking a reliable grasp: it needs either a hand-eye calibration or an
  alignment criterion that uses the visible jaws instead of the image centre.
- Link compliance, backlash in degrees per joint, maximum payload.
