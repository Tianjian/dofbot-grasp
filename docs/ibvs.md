# Image-based visual servoing (IBVS)

The grasp sequence needs no camera extrinsic because the wrist camera points down
the **tool axis**:

> cube at the tool-axis pixel  <=>  the tool axis points at the cube

So the controller consumes only the tracker's centroid and area
(`vision/cube_tracker.py`), plus FK for the final descent.

## Phases

1. **CENTRE** - drive the cube's centroid to the *tool-axis pixel*. That pixel is not
   the image centre: the camera sits off the axis (calipers: 23 mm back, 50 mm
   lateral), so the controller projects the tool axis onto the cube's plane
   (z = 15 mm) and aims there. The target is captured **once**; letting it move with
   the arm creates positive feedback (observed: error grew to +265 px).
2. **DESCEND** - lower the tool tip along the tool axis with the direction held
   constant, i.e. a damped-least-squares step on (q2,q3,q4) that moves the tip
   straight down. Rotating one joint instead tilts the view and drops the cube out
   of frame.
3. **GRASP** - close, lift, verify.

## Parameters carried over from the real arm

| parameter | value | why |
|---|---|---|
| deadband | 18 px | real threshold used 25-30 px; sim FOV is wider so the pixel gain is only 5-8 px/deg |
| max step | 2 deg | larger steps overshoot and oscillate (seen on the real arm) |
| damping | scaled to trace(JJ^T) | a fixed 0.02 was 2000x too large against a J of order 1e-3 and froze the motion |
| descent stop | fingertip z <= 12 mm (FK) | area saturates at ~7 % because the camera looks along the tool axis and the cube sinks below the view as the hand closes |
| descent is FK-only | yes | the cube legitimately leaves the frame; vision only aligns, as planned for the real arm |

## Gripper geometry, taken from the mesh

`link5.STL` contains three plates of **21.3 x 3.0 x 47.9 mm** centred at
**(+-31.1, ~0, 64.1) mm**, i.e. the two jaws. The simulation's original gripper was
wrong on three counts and has been rebuilt from these numbers:

| quantity | old sim | from the mesh |
|---|---|---|
| opening axis | y | **x** |
| jaw offset | +-18.5 mm | **+-31.1 mm** |
| jaw size | 18 x 7 x 40 mm | **21.3 x 3 x 47.9 mm** |
| fingertip reach | z = 70 mm | **z = 87.5 mm** |

With a 6.8 mm slide per jaw the inner faces go from +-20.45 mm (open) to +-13.65 mm
(closed), which clamps a 30 mm cube with 1.35 mm of interference -- a real squeeze
rather than the old geometry's zero margin.

## Status

A recorded run is in `sim/ibvs_sim.mp4` (left: wrist view, right: overview). Reviewing
it frame by frame is what exposed the four control faults listed above -- the numbers
alone looked plausible while the arm was climbing away from the cube.

CENTRE converges from all four seeded positions, and with a 6 px deadband the cube
lands within a pixel or two of the tool axis.

DESCEND now works in closed loop: the tip goes 58.5 -> 12.5 mm in six iterations and
stops at a reachable 16 mm. Getting there took four fixes, each found by watching the
video: the descent must constrain **only the tip height** (asking for dx=dy=0 makes the
least-squares step ~100 deg, which the per-joint clip then distorts), the command has
to be **re-anchored to the achieved angles** each iteration (the servos hold ~1.5 deg
of static error under gravity), the stop threshold must be **reachable** (12 mm was
0.5 mm below the arm's limit, so it oscillated forever), and the descent is FK-driven
because the cube legitimately leaves the frame.

The **close still does not retain the cube**: the video shows the jaws arriving at the
cube, but the cube stays on the table during the lift. Jaw interference is 1.35 mm per
side, so the next suspects are contact friction and the moment the jaws meet the cube's
top edge -- the grasp needs the fingers to descend past the cube's mid-height before
closing.
