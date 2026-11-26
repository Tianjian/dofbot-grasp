# Two-camera alignment against the real arm

Checked with the arm at the task pose `(80.3, 96.5, 2.0, 14.2, 134.8)` -- FK puts the
fingertip at `(21, 123, 82) mm`. Both cameras were captured at that pose and the wrist frame
was round-tripped through the simulation.

## The wrist camera's field of view was 3.4x too wide

| | value |
|---|---|
| real intrinsics | fx **1009.2**, fy 1016.6 -> **26.6 deg** vertical field |
| simulation, before | `fovy="90"` -> 90 deg vertical field |

Nothing had compared them. A 90 deg field keeps the cube in frame from almost anywhere,
which makes centring easy; the real camera loses it quickly. **This alone invalidates the
earlier 4-of-4 grasp rate**, which was measured with the wide field.

After setting `fovy="26.6"` the same four-position trial scores **0 of 4**, all failing in
centring. The simulation is now as hard as the arm, not easier.

## The camera model itself is right

Given the real wrist frame, the yellow cube's centroid sits at pixel `(39, 297)`. Back-
projecting it with `params.MOUNT` and the real intrinsics puts the cube at
`(77, 213, 15) mm`, a 226 mm radius -- plausible for the desk.

Projecting that world point back through the same model, rolling the camera about its
optical axis, the best fit is **roll 180 deg**, which lands the cube at `(30, 298)`: **9 px**
from where the real camera shows it. So the mount position, the optical axis (the tool
axis) and the roll are all consistent with the hardware.

## The observer camera is now placed from measurement

The FOV of the observer was already calibrated (15 views, 0.65 px RMS) but its *position*
was a guess. Measured off the bench:

| quantity | value |
|---|---|
| horizontal offset | **410 mm** in x, **350 mm** in -y, from the base axis projected on the desk |
| height | **25 mm above the desk** |
| aim | tilted **up 13 deg** |
| field | 41.1 deg vertical, from fx 639.9 |

Rendered at that pose beside the real frame, the arm's pose, orientation and proportions
line up (`sim/observer_alignment.png`).

The sign of the x offset is settled by rendering all four combinations
(`sim/observer_camera_signs.png`). The real frame has the gripper on the **right** of the
base; `(-410, -350)` mirrors that and points the fingers left, and `(-410, +350)` does the
same. Only `(+410, -350)` and `(+410, +350)` put the fingers right, and of those
`(+410, -350)` matches the arm's size and angle. What does not line up is size: the arm
appears about 20 % larger in the real frame, which is what a stand under the base would do
by lifting the arm toward the camera.

## The arm stands on a pedestal the model does not have

Zooming the observer frame shows the arm bolted to a four-legged stand whose top sits at
roughly pixel row 340, with the desk at row 395. Scaling by the base cylinder (about 50 mm
across, 65 px in the image) puts the stand at **40-60 mm**.

That is not cosmetic: the simulation's base plane *is* the table plane, so every grasp
height is measured from a plane that is 40-60 mm too high. On the real arm, reaching a
cube on the desk means going below the base plane, not 10 mm above it.

The exact height has to be measured with a rule; the image only bounds it.

## Intrinsics: focal length matches, the rest did not

| term | real | simulation (MuJoCo pinhole) | |
|---|---|---|---|
| wrist focal | 1016.6 px | 1015.3 px | matches, 0.1 % |
| wrist principal point | (269.8, 245.7) | image centre | **50 px off in x** |
| wrist distortion | k1 -0.495, k2 +0.509 | none | edge error ~13 px |
| observer focal | 640.2 px | 640.2 px | matches exactly |
| observer principal point | (281.9, 290.4) | image centre | 38 px / 50 px off |
| observer distortion | k1 -0.073 | none | mild |

MuJoCo cameras are symmetric pinholes: the principal point is the image centre and there is
no lens distortion. The focal lengths now match because the fields of view were derived
from them, but the other two terms did not, and 50 px is 2.8 deg of pointing error -- more
than the grasp tolerance allows.

`sim/real_camera.py` closes that gap: it renders at the real focal length on a wider canvas
and resamples through the measured intrinsics, so the simulated frame has the same geometry
as the real one, distortion included. Verified by construction: the render geometry solves
to f = 1016.6 px for the wrist and 640.2 px for the observer, against 1016.6 and 640.2 real.

## The camera sits 56 mm off the tool axis, and that sets the workspace

A consequence worth stating because it burned several attempts: with the tool axis vertical
and the tip 75 mm above a cube, the cube is **not in frame**. The camera is 56 mm to the side
of the axis, so at 90 mm range the cube sits about 32 deg off a 13 deg half-field.

The camera looks along the tool axis, so it sees whatever the axis points at from 56 mm to
one side -- not what is directly under the tip. Any "move above the cube and look down"
manoeuvre therefore fails, which is why `sim/grasp_trial.py` now searches joint space for a
pose that actually puts the cube inside the frame (tip 50-60 mm up, cube 20-120 px from the
corner) instead of commanding a textbook overhead pose.

That also explains the earlier real frame: at the task pose the tool axis is tilted about
65 deg from vertical, which brings the cube to the bottom-left of the frame rather than
directly ahead.

## The gripper's fingers hung through the palm

The user spotted it in a recording: the finger boxes spanned link5 `-6 .. +40 mm` while the
palm (`link5_geom` 0..24 mm plus `gripper_base` 20..40 mm) occupies `0 .. 40 mm`, so the
fingers passed straight through the palm.

They now continue from its front face: 40 mm long, `40 .. 80 mm` along the tool axis
(`sim/gripper_geometry.png`). Putting them at `-40 .. 0` first was wrong in the other
direction -- that hangs them back towards the arm, which reads as a gripper mounted upside
down, and the user caught it in the recording.

The fingertip is therefore 80 mm out from link5 along the tool axis, 30 mm beyond the TCP
(which sits at 50 mm).

This has a consequence worth stating plainly. With the fingertips 90 mm below the TCP, the
camera -- 56 mm off the tool axis, 25 mm above the fingertips -- only sees the cube when the
arm is high, and the arm is then too high to reach it: a joint sweep finds poses where the
cube is in frame *and* the fingertips clear its top for only two of the four cube positions,
and none for the other two.

## The analytic camera now matches the render

Three convention errors were stacked, and together they made the controller's camera model
disagree with what the simulation draws by **over a thousand pixels**:

* the model rotated about `+z` while MuJoCo cameras look along `-z`;
* the pose search tested `pc[2] > 0` for "in front", which under MuJoCo's convention is
  always false, so every candidate pose was rejected as seeing nothing;
* the roll was a leftover from the OpenCV convention.

With the model expressed in MuJoCo's convention and the roll fitted against the rendered
camera, the two agree to a **25 px mean**. That mattered: the pose search had been choosing
viewpoints that cannot see the cube, which is why seeds kept reporting "cube not visible".

## Still open

**The camera's orientation, in two parts.**

The first part is settled. `sim/camera_orientation_problem.png` shows it: the real wrist
frame puts the cube bottom-left at (39, 297), while the simulation, given the same pose and
the cube at the position the real frame back-projects to, draws it top-right at (559, 188).
Both coordinates flip, which is the signature of a half turn about the optical axis.

Rolling the camera 180 deg takes that error from **533 px to 41 px**, and a 3-degree
refinement of the orientation brings the cube to (77, 296) -- see
`sim/camera_orientation_fixed.png`.

The second part is not settled: in that corrected frame the cube is now on the right side,
but the gripper still is not where the real frame puts it. One point constraint fixes where
the cube lands and leaves the rotation about the camera-to-cube line free, and that free
rotation moves the gripper. A second landmark is required; the cube alone cannot do it.

**Earlier note:** With both the intrinsics and the field of view now right,
a cube placed at the position the real frame back-projects to still renders on the wrong
side of the image -- roughly a half turn about the optical axis. Fitting a single frame
cannot settle this: one point fixes the pointing direction but leaves the roll free (a
one-point fit reaches 0.0 px with three free angles and still looks mirrored). It needs two
or three arm poses with the cube fixed, or a second landmark in the same frame. The dark
gripper bar was the obvious second landmark but a simple darkness threshold did not find it,
so that has to be done properly.

Separately, the quaternion is only axis-aligned approximate. Enumerating the 24 axis-aligned
orientations, the closest to the real pixel is `quat 0.7071 0 0.7071 0`, which lands
`41 px` away -- the true mount is not axis-aligned in link5. The sim needs that rotation
refined to about the degree level, or the two cameras will keep disagreeing by tens of
pixels even though the model is correct.

## What this means for the real arm

The controller has to be retuned for a 26.6 deg field. Everything that depends on the cube
staying visible -- centring, the local-minimum escape, the descent guard -- was tuned
against a field more than three times wider.
