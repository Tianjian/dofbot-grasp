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

Rendered at that pose beside the real frame (`/tmp/obs_final.png` during the session), the
arm's pose, orientation and proportions line up. What does not line up is size: the arm
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

## Still open

The simulation's camera quaternion is only approximate. Enumerating the 24 axis-aligned
orientations, the closest to the real pixel is `quat 0.7071 0 0.7071 0`, which lands
`41 px` away -- the true mount is not axis-aligned in link5. The sim needs that rotation
refined to about the degree level, or the two cameras will keep disagreeing by tens of
pixels even though the model is correct.

## What this means for the real arm

The controller has to be retuned for a 26.6 deg field. Everything that depends on the cube
staying visible -- centring, the local-minimum escape, the descent guard -- was tuned
against a field more than three times wider.
