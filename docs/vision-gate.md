# Vision gate: confirm with a model before moving

Colour thresholds are not trustworthy on this bench. The detector reported a cube
in a frame that contained none (a pale corner of the desk mat matches the yellow
hue band), and it also missed a cube that was plainly visible. Both were caught by
simply looking at the picture with a multimodal model.

So the rule for hardware runs here is:

> **No motion on the basis of a number alone.** Capture, have the gate confirm what
> is in the frame, then act.

## Protocol

1. `python scripts/checkpoint.py --tag <stage> --observer /dev/video2`
   writes the raw frames plus an annotated copy showing what the classical
   pre-filter believes. The annotation is **advisory**; it is drawn in red and
   labelled `UNVERIFIED`.
2. The gate (a multimodal model) is given the raw frame paths and answers strict
   JSON: `cube_present`, `bbox_xywh`, `confidence`, `what_you_see`.
3. Act only when the gate agrees with the plan. Log the verdict with the stage.

## Checkpoints used

| stage | frame | question |
|---|---|---|
| `before_search` | wrist | is the cube in view at all? |
| `centred` | wrist | is it centred and fully inside the frame? |
| `at_grasp_height` | wrist + observer | is the gripper around the cube? |
| `after_lift` | observer | did the cube leave the table and stay in the jaws? |
| `after_release` | observer | is the cube back on the table where expected? |

## What the gate has already caught

- A wrist frame with **no cube** in which the detector reported a cube at
  (102, 405), 204 px wide. Root cause: pale mat patch, mean saturation 104 versus
  178 for the real cube. Fixed by a blob-level saturation floor.
- `deepseek-r1:7b` (text-only, served locally) answering "no cube" for a frame
  that clearly contains one — a reminder to check that the judge actually receives
  the image.
- Confirmation that three candidate frames at different base yaws were the same
  cube, not three different yellow objects.
