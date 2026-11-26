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
