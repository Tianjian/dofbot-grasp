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

## Contact can be felt through the servos

There is no force sensor, but a stall is visible in the position read-back. Driving
`j2` down in 3° steps and comparing the commanded angle with the measured one:

| phase | commanded vs measured |
|---|---|
| free air | tracks within ±0.4° |
| approaching the cube | lags −1.7°, −3.0°, −3.6°, −5.1°, −6.8°, −9.4° |

The lag grows monotonically once the finger meets something, so a threshold of
≈5–6° is a usable "contact" signal. Contact was flagged when FK put the TCP at
z ≈ 24 mm, against a 30 mm cube — i.e. the TCP height model agrees with reality to
within ~5–10 mm.

Caveat seen in the same run: a blocked servo also stops following large commands
(the gripper was commanded from 12° to 163° and stalled at 95° while pressed
against the cube). Motion routines must treat "measured != commanded" as an event,
not as something to keep retrying.

## The camera cannot watch the approach

Measured while lowering the arm onto the cube at `j1≈93`:

| TCP z | cube in view? | apparent width |
|---|---|---|
| 105.9 mm | yes | 219 px |
| 101.5 mm | yes | 226 px |
| 95.9 mm | **no** | — |
| 92.6 … 53.4 mm | no | — |

The cube leaves the frame after roughly **10 mm of descent**, because the camera
(sitting below the roll joint, looking past the fingers) pans forward as the arm
comes down. So a purely visual approach-and-grasp is not possible on this mount:
the target must be located once, converted into the base frame, and the descent
executed blind from forward kinematics.

That conversion needs camera intrinsics and the camera-to-arm transform — i.e. the
hand-eye calibration that the earlier experiments were trying to avoid.

## Camera intrinsics (measured)

From 18 hand-held views of board A, principal point fixed at the frame centre and
only k1/k2 free; 9 views survived outlier rejection (the rest had 2.4-5.9 px
residuals, most likely hand motion during capture):

```
fx = 1015.3   fy = 1014.9   cx = 320.0   cy = 240.0
k1 = -0.415   k2 = +0.061   RMS = 0.838 px over 9 views
```

An unconstrained solve of the same views produced `fx = 8263` and `k3 = -1.1e6` --
the classic planar-target degeneracy. The model is therefore deliberately small and
the solve is seeded with `f = 0.85 * width`; results outside 0.3-3.0x the image size
are reported as IMPLAUSIBLE rather than written out silently.

## Observer camera intrinsics (measured)

The fixed USB observer (`/dev/video2`) was calibrated from 15 hand-held views of
board B (9x6 squares of 20 mm), principal point free:

```
fx = 639.9   fy = 640.2   cx = 281.9   cy = 290.4
k1 = -0.073  k2 = +0.074  RMS = 0.652 px over 15 views
```

Unlike the wrist camera, this solve is well conditioned: the board fills 50-70% of
the frame and is tilted 30-40 deg in every direction.

## Hand-eye calibration is blocked by the field of view

Eye-in-hand (AX = XB) needs the same board seen from poses with rotational
diversity. Measured reality on this arm:

- the board is fully visible at essentially one pose;
- moving yaw by 20 deg or wrist pitch by 15 deg pushes it out of frame;
- of 10 orbit poses only 1 kept the board, and Tsai on 5 near-identical poses
  returned the identity matrix.

The camera sits 10-15 cm above the desk, so its field of view covers roughly 9-10
cm. A board small enough to stay in frame across poses would be 1.5-2 cm at that
distance -- too small to print and detect reliably.

Consequence: **the wrist camera cannot be calibrated against the base by moving the
arm.** Calibrating the fixed observer camera instead (eye-to-hand: board on the desk,
gripper parked at known TCP positions) is the practical route.

## The simulation now carries the real meshes

`sim/dofbot_se.xml` renders the URDF's STL meshes (`sim/meshes/`, 16 MB) as visual
geoms, while the original primitives stay in place for collision (alpha 0, still
`contype=1`). Physics is unchanged: the reference grasp lifts 104.9 mm with 8
finger contacts, the wedge still prints PASS / gate exit 0, and the fault run FAIL /
exit 1. The primitive-only model is kept as `dofbot_se_primitives.xml`.

With the real geometry the rendered silhouette is now the machine, which is what
the model-based camera fit needs.

## Silhouette fitting needs a segmentation the bench does not provide

Matching the rendered arm to the observer image currently fails, and not because of
the model: in the observer view **94% of the region of interest passes the
"bright and desaturated" test**, i.e. the arm barely differs from the bright ceiling
and wall behind it. A coarse grid over camera positions (x 0.1-0.75, y -0.5..0.5,
z 0.25-0.8 m) reaches IoU 0.158 at best -- no camera pose explains the mask, because
the mask is not the arm.

So model fitting on this view needs either a real segmentation cue (a dark backdrop
behind the arm, or the black gripper as the target) or a marker on the robot.

A black backdrop was mounted behind the arm. The contrast is now usable -- at
V>150 the mask is the arm and nothing else -- but the fit still stops at IoU ~0.19,
for two reasons the bench cannot fix:

- the camera has no working exposure control (V4L2 reports -1 and ignores `set`),
  so the "black" board is auto-exposed to V~120-150 and the threshold that separates
  arm from board is unstable between frames;
- the arm is only ~4.7k pixels at 320x240 and the highlight-biased mask covers a
  fraction of it, capping IoU well below 1 even with a perfect pose.

The deterministic alternative is a target on the robot (board taped to the gripper,
AX=XB): it needs no segmentation and its accuracy is set by chessboard corners, not
by exposure.

## Wrist camera mount (solved from the measured geometry)

Using the three numbers measured with a caliper against the link4 flange --
23 mm back from the flange face, 50 mm lateral, optical axis along the arm -- the
transform is:

```
t_link4_cam = (-50.85, +50.0, 0.0) mm      (the flange sits at x = -73.85 mm)
optical axis = link4 -x                     (i.e. along the arm, away from the base)
```

This is consistent with every qualitative observation on the machine: at q = 0
link4's -x points up, and the wrist camera does look at the ceiling there; rotating
the wrist roll (j5) does not change the view because the camera is on link4.

An enumeration over the remaining discrete choices (lateral axis, optical axis among
+x/-x/+y/-y/+z/-z, and four rolls around it) scores each candidate by how tightly the
same cube back-projects to one point across eight arm poses. The winner is the one
above: 14-24 mm spread, and the recovered cube sits at (1, 128, 15) mm in the base
frame.

It is **not yet validated**: reprojecting that cube into the eight frames has an rms
error of 143 px, and the reason is the input data, not the model -- three nearly
identical arm poses (identical j4) reported the cube at (305,242), (253,132) and
(393,250). A static cube cannot do that, so those detections are different yellow
objects. The mount must be checked against gate-confirmed observations.

## Controller verification: angle, direction, zero

Every servo was commanded +15 deg from a common pose, read back, and photographed
with the observer camera:

| servo | commanded | read back | error | observed motion | FK prediction |
|---|---|---|---|---|---|
| j1 (base yaw) | 105.4 | 105.1 | -0.4 | whole arm swings about vertical | dx -48 mm |
| j2 (shoulder) | 110.7 | 109.0 | -1.7 | upper arm and gripper rise | dz +45 mm |
| j3 (elbow) | 36.6 | 34.9 | -1.7 | forearm and gripper rise | dz +50 mm |
| j4 (wrist pitch) | 64.1 | 63.0 | -1.1 | gripper tilts up | dz +32 mm |
| j5 (tool roll) | 149.7 | 149.3 | -0.4 | only the gripper rotates, arm still | displacement 0 |

No joint moved opposite to the model, and j5 producing no TCP motion is exactly what
the kinematics predicts -- so the servo -> joint mapping, including the vendor's
internal S2/S3/S4 reversal, matches this machine.

Zero pose (`u = [90, 90, 90, 90, 135]`): the arm stands vertically in the observer
view, FK gives TCP `(0, 3, 397) mm` with the tool axis `(0, 0.02, 1)`.

Gripper: full 0-180 deg travel in free air with tracking within 0.4 deg, 170 deg =
open, 10 deg = closed.

## Still unmeasured

- Camera intrinsics (`fx, fy, cx, cy, distortion`) — no calibration target used yet.
- Camera-to-TCP offset on the **real** bracket. The frame is mounted ~35 mm off the
  tool axis, but the exact value and direction have not been measured, so centring
  the cube in the image does **not** put the TCP over the cube. This is the one
  number blocking a reliable grasp: it needs either a hand-eye calibration or an
  alignment criterion that uses the visible jaws instead of the image centre.
- Link compliance, backlash in degrees per joint, maximum payload.
