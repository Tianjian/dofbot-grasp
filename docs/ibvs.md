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

## Place the cube inside the graspable band

A kinematic-only sample is not enough: it counts configurations where the arm passes
through the table. Filtering by actual arm-vs-table contacts gives the true region.

```
   x \ y     -220  -180  -139   -99   -59   -19
  -160          .     #     @
   -79         .     *     @     @     @
     0         .     @     @     @     @     *
    80               *     @     @     @
   160                     .     #     @
```

A **full-range** sweep (q2..q4 all across +-90 deg) shows the region is symmetric about
y: 49 % of the collision-free grasp points sit at **y > 0**, with the densest cells at
(20,140), (60,140), (-60,140) and (100,140) mm. That +y side is the one the real arm
works on, so it is the side the simulation should use.

An earlier sweep restricted q3/q4 to a mostly-positive window and therefore reported
only the -y lobe -- which is what made the cube look permanently "in the wrong half".
The remaining dense core on the -y side is x in [-80, 80] mm, y in [-140, -60] mm. Seeds
outside it read as "an ungraspable half of the table" -- and y = -170 mm turned out to
be right on the edge: a fingertip command to (-20, -170, 12) stalls at (-17, -148, 1)
with j4 pinned at its 90 deg limit and the contact count climbing to 7. The original
seeds sat at y = -250 to -285 mm, i.e. outside it, which is why they behaved as
"impossible half of the table" rather than as a controller problem. A pose aimed at the
band's centre (q = [0, -5, 80, 90] deg) keeps a cube anywhere in the band inside the
wrist view.

With seeds inside the band the recorded run centres cleanly (+83 -> -1.3 px) and the
side-entry approach closes to within ~35 mm of the cube.

## The camera belongs on link4, not link5

Mounting the wrist camera on link5 (next to the jaws) was wrong twice over. On the real
arm the camera hangs off the forearm while the jaws sit on the rotating gripper base, so:

* rotating j5 turned the camera with the jaws, making "turn the gripper base" a no-op
  in simulation;
* the view was a straight-down look with the fingers in the middle of it.

The mount is registered to the **camera geometry that is actually in the mesh**. The
STL was split into its connected components and each candidate was rendered with a
marker sphere (`sim/ibvs_camera_candidates.png`); the real camera sits on the 24x24x30
mm part at link4 **(-48.4, -57.6, -0.7) mm**, i.e. **57.6 mm to the -y side** -- the
earlier caliper reading had been applied on the wrong side of the axis, a mirror error.
Its position was then nudged 10 mm further along the optical axis to -58.4 mm so the
virtual camera clears the lens barrel modelled in the mesh (otherwise a ring fills the
centre of the frame).

Earlier note, now superseded by the mesh registration: the caliper values: 23 mm
back from the flange and 50 mm lateral puts the camera at link4 x = -50.85 mm, and the
roll is **flipped 180 deg** about the optical axis. Without the flip the world vertical
maps to the image vertical in the wrong sense -- "the horizon is upside down" -- and
that is not cosmetic: it inverts the vertical error, so every IBVS correction pushed the
arm the wrong way. With the registered orientation the hover stage stops climbing
(89 -> 132 mm before) and settles at z = 44 mm while closing on the cube.
`sim/ibvs_camera_registration.png` puts the registered render beside a real wrist frame
for comparison.

Its height along the tool axis matters too. Rendering a sweep at link4 x = -74, -55,
-35 and -15 mm showed the gripper body swelling into frame as the camera climbed toward
the arm, while -74 mm (right at the link5 origin) put the fingers across the lower
border. **-55 mm**, i.e. 19 mm above the link5 origin, gives the composition the real
camera returns: desk dominant, gripper at the top of frame, fingers only at the bottom
edge. Centring then converges to **-0.1 px**.

Moved to link4, looking along the tool axis (which is link4's **-x**), the view becomes
the side-above view the real camera returns, with the fingers only at the bottom edge.
Centring then converges sub-pixel (160.7 px against a 160.0 px target) because the
target pixel no longer drifts with j5.

A cube resting on the table is best grasped on two opposite faces, so a new stage turns
j5 to put the jaw opening axis parallel to a world-horizontal axis before descending.
With j5 free of the camera, this is now a real degree of freedom.

## Gripper kinematics, matching the official URDF

The DOFBOT-SE URDF drives the jaws as **revolute** joints, with the second one
declared as

```xml
<mimic joint="Rlink1_Joint" multiplier="-1" offset="0"/>
```

so a single servo (j6) moves both. The simulation now mirrors that: two hinge joints
about y with pivots at z = 40 mm, tips at z = 88 mm, and a MuJoCo joint equality
`q_right = -q_left` standing in for the URDF mimic (the same trick the Robotiq 2F-85
and other URDF grippers need, since `mimic` is not portable across simulators).
The model therefore has **6 actuators**, one per real servo, instead of 7.

An 8 deg swing moves each tip 6.7 mm inward: inner faces go from +-20.45 mm open to
+-13.7 mm closed, clamping a 30 mm cube with 1.3 mm of interference.

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

A recorded run is in `sim/ibvs_sim.mp4`, with a keyframe strip in
`sim/ibvs_keyframes.png`. The wrist half overlays the target pixel (green cross) and
the detected cube centroid (red circle), which is what makes the aim error legible:
during the descent the two markers separate even though the reported centring error was
a pixel or two, because the cross is a *fixed pixel* while the arm keeps moving. (left: wrist view, right: overview). Reviewing
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

With the revolute jaws the descent now stops at 17 mm because a **fifth contact**
appears -- the jaws meeting the cube -- so the approach needs a little more lateral
clearance before the final descent.

## The remaining aim error is a parallax, not a tolerance

Tracing the descent with the revolute jaws shows the tip arriving at
**(-28, -221) mm** while the cube sits at **(-30, -250) mm**: a 29 mm shortfall, with
the centring error only 1.1 px. The cause is the "capture the target pixel once"
compromise: that pixel marks where the *original* pose's tool axis projected, and after
the arm travels the current axis points elsewhere. Matching the cube to a stale pixel
is not the same as putting the cube on the live tool axis.

The fix is to recompute the target each iteration **and compensate for its motion**:
the target moves with q, so a naive update loops (that produced the +265 px runaway).
Track the target as a function of q in the same Jacobian, or servo in 3D on the
plane -- the cube's height is known, so the alignment is two-dimensional.

The **close still does not retain the cube**: the video shows the jaws arriving at the
cube, but the cube stays on the table during the lift. Jaw interference is 1.35 mm per
side, so the next suspects are contact friction and the moment the jaws meet the cube's
top edge -- the grasp needs the fingers to descend past the cube's mid-height before
closing.
