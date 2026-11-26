# Sim2real runbook

How to take the simulation pipeline to the arm. Every constant referenced here lives in
`src/dofbot_grasp/params.py`, which records where each measured number came from.

## Before touching the arm

```bash
python scripts/run_ibvs_real.py --dry-run --pixel 320,240 \
       --task-pose '[91.9, 97.5, 15.7, 1.3, 134.8]'
```

`--dry-run` reads the pose, back-projects a pixel onto the table and prints the grasp
target without moving anything. Use it to check that FK, the mount transform and the
intrinsics agree before the arm moves at all.

Sanity numbers from the parked-ish pose above:

| quantity | value |
|---|---|
| TCP | `(-3.9, 118.6, 109.2) mm` |
| tool axis | `(-0.01, 0.41, -0.91)` |
| camera centre | `(-5.6, 143.1, 192.1) mm` (45 mm ahead of, 37 mm below link5) |
| centre pixel hits the table at | `(-8.2, 223.3) mm` |
| TCP target for a grasp there | z = **66 mm**, so the fingertip sits at 10 mm |

## Rules the real bus imposes

These are not style preferences; each one silently produced wrong data before it was
found (see `docs/hardware-notes.md`).

1. **Never read within ~0.3 s of a `write6`.** It cancels the timed move and the arm
   stays put while the code records the previous pose. `Arm.move` sleeps
   `BUS.guard_after_write_s` before the first read.
2. **Poll until settled, never sleep a fixed time.** A 1.2 s commanded move takes 1.5-2.0 s
   to stop; `Arm.wait_settled` waits for two consecutive samples within 0.35 deg.
3. **Stop if a servo lags.** `Arm.move` raises when any joint is more than
   `SAFETY.max_servo_lag_deg` from its command -- that is what contact looks like.
4. **The first reply after idle is stale**; `Arm.read` discards it and asks twice.

## Geometry that the grasp depends on

| fact | value | consequence if wrong |
|---|---|---|
| fingertip sits below the TCP | 56 mm | descending to the wrong height drives the fingers through the desk |
| fingertip grasp height | 10 mm | at 20-40 mm the jaws only touch the cube's top face and lift nothing |
| jaw opening axis | link5 **y** | 90 deg off makes the jaws close on the cube's corners |
| mount lateral offset | -57.6 mm (link4 y) | the mirrored value puts the camera on the wrong side of the gripper |
| optical axis | link5 +z (OpenCV +z) | the opposite sign sends every ray up out of the table |

## Sequence

1. `home` -- move to the parked pose, read back, verify the wedge.
2. `perceive` -- wrist frame, SAM2 or colour fallback, report pixel and area.
3. `centre` -- IBVS on the tool-axis pixel, re-identifying the image Jacobian each
   iteration. Converges to under a pixel in simulation.
4. `align` -- turn j5 so the jaw axis is parallel to a world axis.
5. `fix` -- back-project the cube pixel onto the table plane for a 3-D target.
6. `descend` -- vertical move holding x/y to TCP z = 66 mm.
7. `close` -- servo 6 from ~20 deg to ~140 deg.
8. `lift` -- raise 25 deg of elbow, then confirm with the observer camera.
9. `verify` -- observer frame: is the cube in the jaws and off the table?

## Current state

The sequence lives in `dofbot_grasp.controller` and runs unchanged against either backend
(`docs/controller-architecture.md`). In simulation it grasps **4 of 4** cube positions,
lifting 20-22 mm, from measurements 2-5 mm off the truth (`RESULTS.md`).

The sequence is measure-once-then-IK: the camera looks once from a pose the search has
verified can see the cube, and everything after that is solved kinematically. On the arm the
same order applies, and the observer camera should watch the approach and the lift.

`scripts/run_ibvs_real.py` has the bus wrapper, the geometry and a `--dry-run` planner, but
its motion phases are **not wired to the controller yet** -- the arm has never run this.

## What the simulation already settled

Three behaviours that a first real run would otherwise have to discover:

* the descent must be solved, not stepped. Numerical stepping reported success while leaving
  the fingertip 24-44 mm short, which is how the jaws closed on air; IK puts it within a
  millimetre;
* the descent must stop on **mechanical** contact. Watching the cube's pixel instead gave
  both false stops (fingers entering the frame move the silhouette centroid) and false
  continues (the arm pressed the cube through the table);
* centring the cube in the image is worse than leaving it where it is. Moving it to the
  centre needs tens of degrees of joint motion, and the resulting pose can be 30 mm short of
  reaching the grasp point. Measure from the pre-position instead.

## Range

Grasping works between roughly 100 and 160 mm of the base. Nearer than 100 mm the fingers
occlude the cube in a 90 deg field and centring fails; further out the wrist reaches its
limit before the cube is centred.
