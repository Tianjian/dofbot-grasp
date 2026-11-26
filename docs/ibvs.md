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

## Status

In simulation, the CENTRE phase converges from all four seeded cube positions
(errors driven inside the deadband in 2-4 iterations). The descent reaches the
fingertip target, but the close/lift step does not yet hold the cube - the finger
geometry in `sim/dofbot_se.xml` (finger slides 7-10 mm, 30 mm cube) leaves little
clamping margin, and the residual centring offset of up to 18 px is a couple of
centimetres at that range. That is the next thing to tune.
