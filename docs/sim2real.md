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

`sim/ibvs_sim.py` carries `servo_to_joint`, `joint_to_servo`, `jaw_from_servo6` and
`servo6_from_jaw` so a command can move between the two conventions in one call. The
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
