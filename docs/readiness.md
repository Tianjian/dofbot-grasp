# Pre-flight readiness for the first real run

Checked against the hardware on 2026-11-26, arm parked. Nothing here is assumed; each row
is backed by a measurement taken at the time.

## Ready

| check | result |
|---|---|
| serial device | `/dev/ttyUSB0` present (CH340, `1a86:7523`), opens, user in `dialout` |
| bus read | parked pose reads `90.5, 90.2, 89.8, 89.5, 134.9` -- the zero pose, stable |
| bus write | commanded three poses and all settled within **1.2 deg** of target |
| wrist camera | `/dev/video0`, 640x480 @30 fps, opens |
| observer camera | `/dev/video2`, 640x480 @25 fps, opens |
| wrist intrinsics | `calib/intrinsics_wrist.json`, fx 1009.2, cx 269.8 (50 px off centre) |
| observer intrinsics | `calib/intrinsics_observer.json`, fx 639.9 |
| wrist view at the task pose | the **cube** is visible bottom-left and the **fingers** on the right -- the camera does look along the tool axis at the table |
| cube back-projection | the yellow cube resolves to `(-58, 220, 15) mm`, a 227 mm radius -- inside the 239 mm envelope measured earlier |
| forward kinematics | park pose gives a TCP of `(0, 2, 403) mm`, consistent with the 397 mm measured before the fingertip convention was tightened |

## Blocking

**1. (resolved) A second mount definition disagreed, and was retired.**

`calib/wrist_mount.json` put the camera at `(-50.9, +50.0, 0) mm` looking along link4 `+x`,
against `params.MOUNT` at `(-58.4, -57.6, -0.7) mm` along the tool axis. Tested on the same
wrist frame, `params.MOUNT` back-projects the cube onto the table while the other model
produces no intersection at all -- its ray runs parallel to the table. That file and the
hand-eye script that generated it are **deleted**; `params.MOUNT` is the only definition.

**2. The motion phases are not connected.** `scripts/run_ibvs_real.py` has the bus wrapper,
the geometry, the guards and a `--dry-run` planner, but nothing runs `controller.grasp`
against the arm. There is no code path that actually performs a grasp on hardware.

**3. There is no observer safety monitor.** The observer camera opens but no code reads it.
The simulation has no such need; the arm does.

## Worth knowing before the first run

* the wrist camera looks **over the desk edge** at poses above about 170 mm: at
  `(90, 115, 25, 20, 135)` the frame shows the carpet and a chair, not the table. Aiming
  for an overhead view means going lower than feels natural;
* the fingers are in frame at the task pose, which is what moved the model to thin jaws;
* the arm is parked at `(90, 90, 90, 90, 135)` with the jaws open (`~20 deg`).

## Suggested order for the first session

1. `--dry-run` at the task pose, cube in view: confirm the printed cube estimate matches
   where the cube actually is, within a few millimetres. This single step settles the mount
   question;
2. wire the **centre-only** phase and run it with the observer camera watched by hand. The
   arm should stay above 60 mm tip height and simply rotate the cube to the tool-axis pixel;
3. only then wire descent, close and lift, with the observer monitor in place.
