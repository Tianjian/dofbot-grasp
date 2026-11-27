# Evidence

Every claim in `RESULTS.md` has a file behind it. This is the index.

## The arm grasping

| file | what it shows |
|---|---|
| [`sim/real_grasp_success.mp4`](../sim/real_grasp_success.mp4) | the observer camera for the whole successful run, 28.7 s at 10 fps, stage and timestamp burnt in |
| [`sim/real_grasp_keyframes.png`](../sim/real_grasp_keyframes.png) | eight frames through it: cube on the table at DESCEND, in the jaws from LIFT, still there 60 s later |
| [`sim/real_run_observer.mp4`](../sim/real_run_observer.mp4) | an earlier run where the jaws closed on air -- same flow, recorded for comparison |

The verdict for the successful run came from the jaws, not the video:

```
observation pose: j2 89.8 (droop compensated)
cube at (-17, 169) mm, iou 0.93, yaw 81 deg
at grasp height: tip (-19, 173, 2) mm
verify: jaws at 137.127 against 139.664 free (holding)
```

## The simulation

| file | what it shows |
|---|---|
| [`sim/controller_sim.mp4`](../sim/controller_sim.mp4) | ten seeds running the same flow, wrist camera left, observer right |
| [`sim/controller_keyframes.png`](../sim/controller_keyframes.png) | the frames where the fingertip is 7-11 mm from the cube (success) or 47-140 mm (failure) |
| [`sim/gpu_detection.png`](../sim/gpu_detection.png) | SAM2's mask on the real cube, at 66 mm |
| [`sim/gripper_geometry.png`](../sim/gripper_geometry.png) | the corrected gripper: palm 0..40 mm, fingers 40..80 mm along the tool axis |

## The cameras

| file | what it shows |
|---|---|
| [`sim/wrist_orientation_versions.png`](../sim/wrist_orientation_versions.png) | the same view with the wrist camera rolled 180°: the flip the user caught |
| [`sim/camera_orientation_fixed.png`](../sim/camera_orientation_fixed.png) | the same fault, older evidence |
| [`sim/observer_alignment.png`](../sim/observer_alignment.png) | the observer camera's viewpoint against the modelled one |
| [`sim/wrist_flip_check.png`](../sim/wrist_flip_check.png) | real frame beside the simulation and both flips, for judging orientation |
| [`sim/cube_placement.png`](../sim/cube_placement.png) | where the target sat relative to the arm, and why no sweep found it |
| [`sim/cube_sticker_face.png`](../sim/cube_sticker_face.png) | the target's sticker face -- which is why a colour gate cannot find it from the side |
| [`sim/visual_inspection.png`](../sim/visual_inspection.png) | observer and wrist view side by side during an inspection |

## Measurements

| file | what it holds |
|---|---|
| `calib/intrinsics_wrist.json` | wrist intrinsics, RMS 1.06 px over the calibration views |
| `calib/intrinsics_observer.json` | observer intrinsics, RMS 0.70 px over 15 views |
| `tests/data/real_cube_wrist.png` | a real frame of the target at 66 mm, used as the detector's regression case |
| `calib/grasp_success.json` | the earlier scripted grasp, kept for comparison |

## Reproducing the numbers

```bash
# simulation gate (ten seeds, same controller as the arm)
MUJOCO_GL=egl python sim/grasp_trial.py --min-rate 0.75

# the recording, with the wrist and observer cameras side by side
MUJOCO_GL=egl python sim/record_controller.py --seeds 0,1,2

# the arm, recording to logs/observer/run_<stamp>.mp4
python scripts/run_ibvs_real.py
```

## What is not in the repository

`logs/` holds the recordings and alarm frames from every run, and is ignored: it is runtime
output, it grows by megabytes a run, and the curated evidence above already preserves what
matters. A fresh run writes there again.

The pack is larger than the working tree because earlier commits tracked that directory before
it was ignored. Removing those objects would mean rewriting published history, which is not
worth doing for a repository whose history is the record of how the controller was arrived at.
