# Simulation ↔ real robot correspondence

Everything here is measured on the real arm and reproduced in `sim/dofbot_se.xml`.
Where the two disagreed the real measurement wins.

## Joint convention

The bus servos report `u` in degrees: 0-180 for ids 1-4 and 6, 0-270 for id 5. The
vendor library reverses servos 2/3/4 internally on both write and read, and after that

```
q1..q4 = u - 90 deg        (u = 90 is the vertical / neutral pose)
q5     = u - 135 deg       (270 deg of travel)
```

`dofbot_grasp.params` carries `servo_to_joint` and `joint_to_servo`, so a command can
move between the two conventions in one call. The
zero pose `u = [90, 90, 90, 90, 135]` maps to `q = 0` and is the pose the real arm
stands in when parked.

## Limits

| joint | real servo | real q | simulation |
|---|---|---|---|
| j1..j4 | 0-180 | -90..+90 | -90..+90 |
| j5 | 0-270 | **-135..+135** | **-135..+135** (was -90..180 before this pass) |
| j6 (gripper) | 0-180, jaws apart ~20, shut ~140 | -- | hinge 0..9.2 deg |

The gripper is one servo driving both jaws (URDF `mimic multiplier="-1"`), modelled with
a MuJoCo joint equality. Servo 6 around 20 deg is fully open and around 140 deg is shut;
that 120 deg of travel maps onto 9.2 deg of hinge, about a 13:1 reduction.

## Verification

Two independent checks that the joint layer really matches:

**Forward kinematics, 0.0 mm.** For seven poses -- including the parked zero pose and two
poses photographed on the real arm -- the simulation's tool-tip position and the
project's real kinematics module (`dofbot_grasp.kinematics`) agree to **0.0 mm**:

| pose | servo u | real FK TCP (mm) | sim tip (mm) |
|---|---|---|---|
| zero | 90/90/90/90 | (0, 1, 397) | (0, 1, 397) |
| oblique | 105/115/6/13 | (-32, 120, 137) | (-32, 120, 137) |
| wrist-down | 100/87/19/7 | (-23, 131, 97) | (-23, 131, 97) |
| j2 only | 90/60/90/90 | (0, 146, 358) | (0, 146, 358) |
| j3 only | 90/90/60/90 | (0, 105, 369) | (0, 105, 369) |
| j4 only | 90/90/90/60 | (0, 63, 381) | (0, 63, 381) |

**Image comparison is not a valid check here.** `sim/sim2real_pose_check.png` puts the
real observer photo beside the simulated overview for three poses. They look quite
different -- the real arm folded high, the simulated one low -- but that is the two
cameras being in different places, not a kinematic difference: the numbers above are the
same to 0.0 mm. Any visual joint check has to fix the viewpoint first, otherwise it
measures the camera, not the arm.

## Reference images

| file | what it settles |
|---|---|
| `sim/observer_view_match.png` | the observer camera's viewpoint against the real cell |
| `sim/sim2real_pose_check.png` | why image comparison is not a kinematics check: the two cameras sit in different places, so the pictures differ while the numbers agree to 0.0 mm |
| `sim/gripper_jaw_axis.png` | the jaws separate along link5 **y**, not x -- one is edge-on, the other faces the cube as the photographs do |

## Response

Measured on the real arm: a 1.2 s commanded move settles in **1.5-2.0 s** and holds a
static error of about **1.5 deg** under gravity.

The simulation reproduces both, but only after two changes:

* **kp 30 -> 18** on j1..j4. At kp=30 the static error was 0.91 deg, too stiff against
  the real 1.5 deg.
* **timed moves**. Setting `data.ctrl` directly settled in 0.14-0.26 s, eight times
  faster than the hardware, which changes how the controller behaves. `servos()` now
  ramps `ctrl` to the target over `MOVE_TIME_S = 1.2`, mirroring the move time carried
  by every real `write6` frame.

Result: static error **1.53 deg**, settling **1.6 s** -- both inside the real range.

## Bus behaviours the simulation does not model

Kept here because they will bite during deployment:

* a read frame sent within about 0.3 s of a `write6` **cancels the move**;
* a 1.2 s commanded move can still be travelling at 1.6 s, so "sample on a fixed delay"
  reads the arm mid-move;
* the first reply after an idle period is stale, so the driver discards it and asks
  twice.

These are documented in `docs/hardware-notes.md` with the measurements behind them.

## The fingers change length with the jaws

The real gripper is a parallel-linkage mechanism: at the open position its fingers are short,
and they lengthen towards the closed position, where they approach the length modelled in the
simulation. The simulated fingers are a fixed box, so they are full length even when open.

That matters for the approach. A cube 9 mm off the tool axis is straddled by closed-length
fingers but swept aside by open-length ones, which is why three of the four simulated seeds
push the cube 26-37 mm during the descent. On the arm the fingers are short at that moment and
pass beside it. The simulated grasp rate is therefore pessimistic for this failure mode, and
fixing it is a modelling change rather than a control change -- not urgent.

What is worth doing on the controller side is the same in both places: make the verification
judge the grasp from the jaws' travel rather than from the fingertip's height.

