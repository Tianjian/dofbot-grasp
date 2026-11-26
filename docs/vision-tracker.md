# Cube tracker (SAM2, GPU)

`vision/cube_tracker.py` wraps SAM2 tiny behind a point-prompted interface. The
prompt is seeded from a saturated-yellow centroid and then re-used from the previous
frame, which keeps the mask locked to the same object as the arm moves.

## Why not a colour threshold

The cube's top face is washed out by the camera's auto exposure: its saturation is
**43**, lower than the desk mat's **104**. A saturation mask therefore only captures
the lit front face. Measured on the same frame:

| method | silhouette area |
|---|---|
| colour mask (S>110) | 14,844 |
| **SAM2 (point prompt)** | **15,898** |
| GrabCut (box prompt) | 50,468 (desk included -- wrong) |

SAM2's contour traces the whole visible cube (pale top + yellow front), and it costs
one forward pass. Colour is still used, but only to place the first prompt.

## What it returns

`update(frame)` -> `{centre, area, bbox, clipped, mask}`. Deliberately **no 3D**:
the tracker is extrinsic-free, so it can be consumed by image-based servoing
directly. `clipped` matters because a cube touching the frame edge biases both the
centroid and the area.

## Timing

134-863 ms per 640x480 frame on the RTX 2060 (torch 2.9.1+cu129, ultralytics
8.3.252), measured in the `face_tracker` virtualenv which already carries the CUDA
stack. Weights live in `models/sam2_t.pt` so no run re-downloads them.

## Note on the install path

Do not reach for a fresh CUDA install: this machine's proxy runs at ~236 KB/s and the
CUDA wheels are ~3 GB. A domestic mirror (aliyun pytorch-wheels) measured
**3.6 MB/s**, 15x faster, and the pre-existing `face_tracker` environment already has
torch+CUDA+ultralytics.
