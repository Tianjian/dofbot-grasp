# Model weights

Not stored in git — the SAM2 tiny checkpoint is 75 MB.

`vision/cube_tracker.py` downloads it on first use (ultralytics resolves
`sam2_t.pt` from its asset release). To place it here explicitly:

```bash
python -c "from ultralytics import SAM; SAM('sam2_t.pt')"   # downloads once
mv sam2_t.pt models/
```

Then construct the tracker with `CubeTracker("models/sam2_t.pt")`.

The download needs roughly 6 minutes on this machine's proxy (~236 KB/s); a domestic
mirror is an order of magnitude faster if the link is the bottleneck.
