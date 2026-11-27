# The task flow: park → watch → centre → measure → grasp → verify → hold

This is the sequence to implement. It replaces "sweep and hope" with a deterministic loop, and
it puts a verification step where there was none. Below is the design, then what the current
code already does, then the differences that have to be built.

## The constraint that shapes everything

The wrist camera is 57 mm off the tool axis, 95 mm above the fingertip, and sees 26.6°. From
an observation pose where the camera sits 149 mm above the desk, its footprint is a circle of
radius 149·tan(13.3°) = **35 mm** — a patch 70 mm across, on a 30 mm cube.

Every design decision below follows from that one number. It is why a fixed pose cannot find a
cube that is not already under it, and why "wait until it enters view" works only when somebody
places it there.

| observation pose | camera height | looks at | footprint |
|---|---|---|---|
| j (90, 95, 0, 5) | 149 mm | (0, 166) mm | **70 mm across** |
| j (90, 80, 10, 0) | 129 mm | (0, 155) mm | 61 mm |
| j (90, 100, 10, 20) | 209 mm | (0, 308) mm | 99 mm — but out of reach |

## The flow

### 0. PARK
All joints at their zero/vertical pose, gripper open. This is the state the arm returns to
whenever anything fails, so it has to be reachable from anywhere and safe to sit in.

### 1. OBSERVE
Move to the observation pose `OBSERVE_POSE = (90, 95, 0, 5)`, gripper open. From here the
camera looks at (0, 166) mm on the desk with a 70 mm footprint.

### 2. WATCH — loop until something yellow is in view
Repeat at ~1 Hz:

1. capture a wrist frame;
2. segment it (SAM2, prompt-free);
3. for each cube-shaped mask, take the **mean hue of the masked pixels**;
4. accept only if the hue sits in the yellow band and the saturation clears the pale-paper
   floor.

The colour test is a *gate on the segmenter's output*, not a detector. That distinction is the
whole reason the old code failed: a colour threshold asked to find the cube could not, because
the cube's own saturation sat one unit under the threshold. Asked only to confirm what the
segmenter already found, colour is reliable and cheap. It is also what rejects the blue box the
segmenter happily segments at the same spot.

If nothing appears within `WATCH_TIMEOUT` (say 30 s), stop and say so. Waiting forever on a
cube nobody placed is not a state machine, it is a hung robot.

### 3. CENTRE
Move the arm so the cube's centroid sits at the image centre (320, 240), in small steps.

**This step has a trap that has already bitten once.** Centring needs tens of degrees of joint
motion for a cube at the frame edge, and the resulting configuration can put the grasp pose
30 mm out of reach. So centring is bounded: at most `MAX_CENTRE_ITERS = 6` steps of at most 3°
each, and after every step the reachability of the grasp pose is re-checked with
`search_poses`. If centring ever makes the grasp unreachable, undo the step and keep the
previous pose — a well-framed cube that cannot be reached is worse than a badly-framed one that
can.

Convergence: within 25 px of the image centre, or stop at the iteration cap.

### 4. MEASURE
Fit the 30 mm cube prior to the mask: position (x, y) and yaw on the table, using the camera
model and FK. This already exists as `controller._fit_once` and returns IoU, which is the
quality signal. Reject the measurement if IoU < 0.6.

### 5. GRASP
1. Square the jaws to the fitted yaw (`align_jaws_to_yaw`).
2. IK to a point 35 mm above the cube, tool axis down.
3. IK straight down to the grasp height.
4. Close the gripper.

Steps 2–4 exist as `_ik_to_tip` + the CLOSE stage of `grasp_measured`.

### 6. VERIFY — new, and the flow needs it
There is currently no check at all: the sequence ends and reports what it did, not what
happened. Two independent signals are available:

* **Observer, size-based** (needs no extrinsics): measure the cube's apparent width before the
  lift and after it. A cube lifted 20 mm toward the observer grows by roughly
  `30/f·(1/d − 1/(d−20))` — at d = 200 mm with f = 640, about 4 px. That is small but
  measurable, and it is the only signal that says "the cube went up" rather than "the arm went
  up".
* **Wrist, presence-based**: after the lift, the cube should no longer be under the jaws. With
  the narrow footprint this is weak — the cube can be out of frame anyway — so it is a veto on
  contradiction, not a confirmation.

Accept the grasp if the observer's apparent width grows by at least 2 px, or if the wrist loses
the cube while the observer still sees it. Anything else is a failure.

### 7. HOLD or PARK
* On success: keep the gripper closed and the joints commanded for 60 s, re-sending the hold
  command every few seconds so the bus keeps the torque on.
* On failure: return to PARK and report the stage that failed.

## Gap between this and the current code

| step | current | designed | gap |
|---|---|---|---|
| 0 PARK | implicit (whatever pose it starts in) | explicit, used as the failure return | add `go_park()` and call it from every failure path |
| 1 OBSERVE | sweeps 9 poses chosen by camera coverage | one fixed pose | remove the sweep; add `OBSERVE_POSE` |
| 2 WATCH | one detection attempt per sweep pose | loop until yellow, with a timeout | **new**: the colour gate, the loop, the timeout |
| 3 CENTRE | removed entirely — it broke reachability | bounded centring with a reachability check each step | **new**, and it must be bounded or it re-breaks what it broke |
| 4 MEASURE | `_fit_once` + IoU gate | same | none — already there |
| 5 GRASP | measure → IK → close | same | none — already there |
| 6 VERIFY | **nothing** | observer size change, plus a wrist veto | **new**; needs a "measure apparent width" call and a hold |
| 7 HOLD / PARK | returns a dict | 60 s hold, or park | **new** |
| detection | SAM2, shape only | SAM2 + yellow gate | colour verification added; this is what stops the blue box |

### What the design deliberately keeps from the current code

* **One controller, one vision stack.** The simulation runs this same sequence; the rule in
  `docs/controller-architecture.md` applies unchanged.
* **The 30 mm prior.** Both the measurement and the verification lean on knowing the cube's
  physical size, which is the only reason a single camera can say anything metric.
* **No colour threshold as a detector.** Colour appears once, as a gate on what segmentation
  found.

### What it deliberately drops

* the pose sweep as the primary search — it is expensive, it needs a GPU pass per pose, and it
  was never covering the desk evenly anyway;
* centring as a free-running servo — bounded now, because it fights reachability.

## Risks worth stating before writing the code

1. **The 70 mm footprint is the whole design.** If the operator places the cube 60 mm off, the
   watch loop never fires and the run times out. Either the placement is precise, or step 1
   needs two or three observation poses rather than one. This is a decision to make explicitly,
   not to discover at run time.
2. **The centring budget may be too small** for a cube at the frame edge: 6 × 3° may not close
   300 px. If it cannot, the fallback is to accept a poorly-centred cube — the measurement's
   IoU gate already tells us when that is not good enough.
3. **Verification is the weakest link.** A 4 px size change is close to noise on a webcam with
   auto-exposure. It may need the observer's exposure pinned, or a larger lift (40 mm instead
   of 20) to make the signal unambiguous.
4. **Holding for 60 s stresses the bus.** The measured settle is 1.5–2 s per command with a
   0.55 s read guard; a 60 s hold should re-send every ~5 s and never read inside the guard.

## Validation order

Per the project rule: simulation first. The sequence is exercised in
`sim/grasp_trial.py` against the same controller, with the detector swapped for the rendered
frame rule only in the CI configuration. The new states (WATCH, VERIFY, HOLD) need cases in the
trial: one where the cube is present, one where it never appears, and one where the grasp slips
so the failure path is exercised.
