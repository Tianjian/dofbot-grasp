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

**2. (resolved) The motion phases were not connected.** `scripts/run_ibvs_real.py` now builds
`SerialRobot`, checks that the cube is visible, and hands over to
`controller.grasp_measured` -- the same sequence the simulation runs. Its duplicated geometry
is gone: the bus wrapper moved into `dofbot_grasp.arm` and the camera maths now comes from
the controller, so there is one implementation of each.

**3. (resolved) There is now an observer safety monitor.** `dofbot_grasp.safety` runs the
observer camera in a background thread and flags two things the wrist view cannot: unexplained
motion in the room (someone reaching in) and a camera that has stopped delivering. It aborts
the run between stages and writes the frames around the alarm to `logs/observer/`.

### Reviewing the monitor after the cube's appearance was clarified

Three faults, all of them false alarms that would have made the monitor unusable on the first
run rather than dangerous:

* **the cube check counted misses even when it was off.** Segmentation now records whether it
  actually ran (`cube_checked`); with no seed, a missing centre means nothing.
* **a still room looked like a dead camera.** Staleness was tested against pixel change, so a
  quiet scene alarmed after three seconds. It now tests whether frames are arriving.
* **a camera that stopped returning frames was never noticed at all.** That check lived inside
  the per-frame path, which silence never reaches. It is now in the worker loop.

And one that is not a false alarm but a trap: `ObserverVision` seeded SAM2 from a yellow blob,
which can never find this cube from the side, because the sides are white stickers. The seed
now has to come from the caller.

### Simulation and reality differ here more than anywhere else

The simulated observer renders a plain orange cube on a checkered floor with a flat backdrop.
The real observer sees a white-stickered cube among papers, a shelf of colourful packaging and
a chair. Nothing calibrated on the rendered observer view transfers: thresholds, colour ranges
and blob sizes all came from the real camera instead, and the simulated observer is used only
to draw the video.

It is deliberately image-space, not a 3-D envelope: the observer's intrinsics are calibrated
but its extrinsics are not, so it can say "the scene changed" but not "a hand is inside the
workspace". Thresholds: 4 % of a downsampled grey frame changing by more than 25 levels, on
three consecutive frames, after a 20-frame warm-up for auto-exposure. Tested against the real
camera (no false alarms over four seconds) and against synthetic motion (alarms).

## Worth knowing before the first run

* the wrist camera looks **over the desk edge** at poses above about 170 mm: at
  `(90, 115, 25, 20, 135)` the frame shows the carpet and a chair, not the table. Aiming
  for an overhead view means going lower than feels natural;
* the fingers are in frame at the task pose, which is what moved the model to thin jaws;
* the arm is parked at `(90, 90, 90, 90, 135)` with the jaws open (`~20 deg`).

## Every run is filmed

`scripts/run_ibvs_real.py` records the observer camera for the whole run to
`logs/observer/run_<stamp>.mp4` (10 fps, timestamp and stage label burnt in). Recording starts
before anything moves: the most useful review is usually of a run that failed early, and an
earlier version only started the camera at the grasp stage, which is precisely when a failed
run never gets there.

The monitor's capture thread does the recording. A second reader on the same device fights the
first one, and the frames a review wants are the ones the monitor itself judged.

## Modes

```bash
python scripts/run_ibvs_real.py --dry-run        # read the pose, report, move nothing
python scripts/run_ibvs_real.py --measure-only   # sweep for the cube, measure, stop
python scripts/run_ibvs_real.py                  # full grasp, observer monitor running
```

The sweep is four poses that look down at the front of the desk (j2 90, j3 20, j4 20, j1 in
{60,80,100,120}), chosen because the tool axis must point down or the wrist camera looks at
the room.

**Where the cube has to be.** The arm reaches about 175 mm from its base. The object used to
identify the target sat roughly twice that far away -- near the observer camera rather than
the arm -- so no sweep could see it. Estimated from its apparent size (about 100 px across for
a 30 mm cube at the observer's 640 px focal length) it was ~190 mm from the observer, which is
~350 mm from the arm base. It has to be inside the swept area in front of the arm.

**What the target actually looks like.** A 30 mm cube whose top face is plain yellow and whose
four side faces each carry a different white sticker with a black line drawing -- a paper roll,
chopsticks, a cigarette and a peach. The side patterns do not affect the wrist pipeline: the
camera looks straight down and sees only the yellow top, and `cube_fit` matches silhouettes
against the 30 mm prior, so which sticker is facing where is irrelevant to the measurement.
They would matter only to a reader trying to identify the cube from the side, which is why the
observer's cube check takes an explicit seed. The wrist camera looks down, so it sees the yellow top and the colour
detector works normally.

The observers see the opposite. The fixed observer looks at the cube from the side, where
there is only a white sticker, so seeding SAM2 from yellow in that view can never find it.
`ObserverVision.check` therefore takes its seed from the caller and `expect_cube` defaults to
False: the observer's cube check is off unless somebody who knows where the cube is supplies a
point. Its person check and the pixel-motion check are unaffected.

**The observer camera's extrinsics are not right yet.** Its pose was taken from the simulated
mount (0.410, 0.350, 0.025 aiming at the origin at 150 mm). Projecting the fingertip into the
observer frame puts the marker about 140 px from where the gripper actually is.

The fingertip is not a usable target for fixing that, though: the gripper hides it, its edge is
ambiguous, and its position is itself extrapolated 80 mm along the tool axis from a camera
mount that sits 56 mm off that axis -- calibrating against it would fold the error being
measured into the calibration.

`scripts/calibrate_observer_extrinsics.py` uses the table chessboard from
`calib/a4_target.pdf` instead (9x6 squares of 20 mm, 8x5 inner corners). Put it flat on the
desk, confirm it is seen with `--detect-only`, measure its top-left corner relative to the
base plus its rotation, and solve: the corners are exact points on the table plane, so solvePnP
gives the observer's pose and a reprojection error to judge it by. Result lands in
`calib/observer_extrinsics.json`.

**The colour detector false-positives on the real desk.** It fired on a pale patch of the
paper covering the table during a sweep, and on a yellow bag strap in another view. The cube
must be the only yellow object in the swept area, and the fit's IoU gate (0.6) is what
currently rejects the bad ones.

## Suggested order for the first session

1. `--dry-run` at the task pose, cube in view: confirm the printed cube estimate matches
   where the cube actually is, within a few millimetres. This single step settles the mount
   question;
2. wire the **centre-only** phase and run it with the observer camera watched by hand. The
   arm should stay above 60 mm tip height and simply rotate the cube to the tool-axis pixel;
3. only then wire descent, close and lift, with the observer monitor in place.
