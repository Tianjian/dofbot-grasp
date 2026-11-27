# Running the vision stack locally

The detector is a SAM2 segmenter (`dofbot_grasp.tracker.SamCubeDetector`), so anything that
looks at a camera frame needs `torch` with CUDA and `ultralytics`. There is no colour fallback:
without them the detector raises rather than answering worse.

## Where the GPU stack lives

This machine already has a CUDA build — `torch 2.9.1+cu129`, `ultralytics 8.3.252` — in
`/home/toby/Code/face_tracker/face_tracker_env`. Rather than download a second copy, the
simulation virtualenv is pointed at it with a single `.pth` file:

```bash
SP=/home/toby/Code/face_tracker/face_tracker_env/lib/python3.12/site-packages
echo "$SP" > .venv/lib/python3.12/site-packages/gpu_stack.pth
```

Both environments are CPython 3.12.3, which is what makes this safe: the wheels are built for
the same ABI tag. Verify with

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` means everything below works.

### Why not just pip install

Installing torch from the PyTorch CUDA index fails here:

```
error: ssl-verification-failed
× Failed to establish a secure connection to pypi.nvidia.com
```

The CUDA runtime wheels come from `pypi.nvidia.com`, which the network in this environment
refuses. The `.pth` route avoids the download entirely, and avoids keeping two 2 GB copies of
the same CUDA libraries on disk.

## What runs where

| Task | Needs the GPU stack | Runs in |
|---|---|---|
| `scripts/run_ibvs_real.py` (hardware) | yes | any env with the `.pth` and `pyserial` |
| `sim/grasp_trial.py` (CI gate) | **no** | plain NumPy env — it renders frames and reads their exact colours with `sim/sim_detector.py` |
| `tests/` vision tests | yes, else they skip | `.venv` |
| `tests/` everything else | no | `.venv` |

The CI gate deliberately does not need a GPU: it runs on GitHub runners, and the simulated
frames are unlit renders whose colour is known exactly, so the RGB rule there is not a
weaker version of the detector — it is the right tool for rendered input.
