# Results

Everything below is measured, not estimated. Real-robot rows come from the arm itself;
simulation rows from `sim/dofbot_se.xml` driven by `sim/grasp_trial.py`.

## Real robot

| item | value | where |
|---|---|---|
| Serial bus | `/dev/ttyUSB0`, CH340, **115200** baud (1 Mbaud fails) | `docs/hardware-notes.md` |
| Servo read-back stability at rest | peak-to-peak **<= 0.16 deg** | measured over 30 samples |
| Joint directions | all verified against FK with the observer camera | `docs/hardware-notes.md` |
| Zero pose | `u = [90, 90, 90, 90, 135]`, arm vertical, TCP `(0, 3, 397) mm` | measured |
| Static tracking error | **~1.5 deg** under gravity | measured |
| Settling time | **1.5-2.0 s** for a 1.2 s commanded move | measured |
| Gripper direction | servo 6 **~20 deg = jaws apart**, **~140 deg = shut** | video verified |
| Fingertip offset | **~56 mm** below the TCP point | contact test |
| Top-down grasp envelope | radius <= 239 mm at 0 mm height, <= 173 mm at 140 mm, none above 150 mm | kinematics + collision sweep |
| First successful grasp | cube held and lifted clear of the desk, servo read-back unchanged after 3 s | `calib/grasp_success.png` |

Two bus behaviours that silently corrupt data, both reproduced:

* a read frame sent within **~0.3 s** of a `write6` cancels the timed move;
* a 1.2 s move can still be travelling at 1.6 s, so sampling on a fixed delay reads the
  arm mid-move.

## Cameras

| camera | intrinsics | notes |
|---|---|---|
| wrist | fx 1009.2, fy 1016.6, **cx 269.8**, cy 245.7, k1 -0.495, k2 +0.509, RMS 1.06 px | principal point is 50 px left of centre; holding it at centre was a 1.7 deg ray bias |
| observer | fx 639.9, fy 640.2, cx 281.9, cy 290.4, k1 -0.073, k2 +0.074, RMS 0.65 px | 15 views |

Wrist camera placement is registered to the camera that is modelled in `link5.STL`:
link4 `(-48.4, -57.6, -0.7) mm`, rolled 180 deg about the optical axis, then nudged 10 mm
along the axis to clear the lens barrel. The roll matters: without it the image is
vertically inverted, which inverts the vertical IBVS error and drives the arm the wrong
way.

Observer is placed at `(660, 280, 240) mm` aimed at `(80, 40, 150) mm`, 41 deg vertical
field, with a black backdrop behind the arm as in the real cell.

## Perception

The wrist camera sees a 30 mm cube whose pale top face is less saturated (S~43) than the
desk mat (S~104), so a saturation threshold only finds the lit front face. Its centroid
is therefore biased towards that face, and a back-projected pixel turns the bias into
aim error.

SAM2 is point-prompted from the colour blob and returns the whole silhouette. Measured on
four simulated wrist frames against the projection of the cube's true centre:

| frame | colour centroid error | SAM2 centroid error |
|---|---|---|
| 0 | 8.2 px | 6.6 px |
| 1 | 9.8 px | 4.0 px |
| 2 | 3.3 px | 3.8 px |
| 3 | 6.3 px | 1.8 px |
| **mean** | **6.9 px** | **4.1 px** |

So the GPU tracker removes about 40 % of the centroid bias, which is a large part of the
11 mm back-projection residual noted below. It costs 134-863 ms per frame on an RTX 2060,
which is why it is used for the 3-D fix while the colour blob keeps providing the prompt.

`dofbot_grasp.tracker` picks the backend at import time and degrades to colour rather than
raising, so a machine without CUDA still runs the pipeline.

## Simulation matches the arm

| check | result |
|---|---|
| forward kinematics, 7 poses incl. the parked pose and two photographed poses | **0.0 mm** vs the project's real kinematics module |
| joint limits | j1-j4 +-90 deg, j5 +-135 deg (0-270 servo), all matching the bus |
| static tracking error | **1.53 deg** (real ~1.5) |
| settling after a 1.2 s move | **1.6 s** (real 1.5-2.0) |

`docs/sim2real.md` carries the full servo <-> joint mapping.

## Grasping in simulation

Wrist-camera IBVS, no camera-calibration guesswork in the loop:

1. **centre** the cube on the tool-axis pixel, re-identifying the image Jacobian every
   iteration (converges to **<1 px**);
2. **align** j5 so the jaw axis is parallel to a world axis;
3. **back-project** the cube's pixel onto the table plane to get a 3-D fix
   (**1-2 mm** typical);
4. **descend vertically** holding x/y to z = 10 mm, **close**, **lift**.

| cube position | radius | lift | cube drift before the clamp |
|---|---|---|---|
| (20, 140) mm | 141 mm | **48.7 mm** | 7 mm |
| (-60, 140) mm | 152 mm | **49.9 mm** | 2 mm |
| (60, 140) mm | 152 mm | **52.2 mm** | 6 mm |
| (40, 100) mm | 108 mm | **52.9 mm** | 58 mm |

**All four seeds grasp** with the simulation's original 90 deg camera field, lifting the
cube 49-53 mm clear. That field turned out to be **3.4x wider than the real camera's
26.6 deg** (`docs/camera-alignment.md`), and with the field corrected the same trial
scores 0 of 4 -- the rate above did not transfer. Retuning is in progress. Drift is how far the
cube travelled before the jaws closed: three of the four are clean (2-7 mm), the close-in
one is shoved along and then gripped, which by the agreed criterion still counts.

The graspable range is narrower than the kinematics allow. Closer than about 100 mm radius
the centring fails -- the fingers occlude the cube in a 90 deg field -- and at 172 mm the
reach leaves the arm in a configuration whose wrist is already on its limit.

The graspable region was mapped by sweeping the kinematics with collision filtering:
grasp points split evenly about y (49 % at y > 0) with the dense core at
x in [-80, 80] mm, y in [-140, -60] mm on the -y side and the mirror on the +y side.

## What is still open

| item | status |
|---|---|
| back-projection residual | 1-3 mm after the tracker rework; the 11 mm case was the colour centroid sitting off the cube on an oblique view |
| near field | centring fails inside ~100 mm radius: the fingers occlude the cube |
| sim-to-real transfer | **not yet run** -- the controller has never driven the arm |

## Controller and the CI gate

`dofbot_grasp.controller` holds the phases once; `dofbot_grasp.hardware` provides a MuJoCo
backend and a serial backend behind five methods (`docs/controller-architecture.md`).

Two faults dominated the debugging, both invisible in a height-only success test:

* **centring settled in a local minimum.** With the wrist pinned on its limit the pixel
  gradient stalled 23 px short while a 1.8 px solution sat ten degrees away on the shoulder
  and wrist. The controller now tries a small joint-space grid when the error stops
  improving, skipping probes that would drive the fingers into the cube.
* **the descent drifted.** Height-only commands let the lateral error grow from 8 mm to over
  15 mm across one descent, past the 10 mm the jaws allow; correcting all three axes each
  step removes it. The descent also stops on mechanical contact rather than vision -- the
  fingers entering the frame move the silhouette centroid, which produced both false stops
  and false continues.

`sim/grasp_trial.py` runs those four positions and exits non-zero below a rate. CI runs it as
a **blocking** job at a threshold of 0.75, so one seed of slack remains before a robustness
regression turns the build red.

## Reproducing

```bash
python -m pytest -q                              # 71 tests
MUJOCO_GL=egl python sim/grasp_trial.py --min-rate 0.75   # the CI gate, four positions
MUJOCO_GL=egl python sim/record_controller.py             # recording -> sim/controller_sim.mp4
```
