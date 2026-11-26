# Controller architecture

One sequence, two backends. The point of the split is that the phases are written once
and the differences between simulation and hardware live behind a small interface.

```
                 ┌───────────────────────────┐
                 │  controller.grasp(io, …)  │   phases, FK, geometry
                 └────────────┬──────────────┘
                              │ RobotIO
              ┌───────────────┴───────────────┐
              │                               │
      ┌───────▼────────┐              ┌───────▼────────┐
      │   SimRobot     │              │  SerialRobot   │
      │ MuJoCo cell    │              │ servo bus + USB│
      └────────────────┘              └────────────────┘
```

## `RobotIO`

Five methods, no policy:

| method | meaning |
|---|---|
| `read_joints()` | servo angles, ids 1..5, degrees |
| `command(target, gripper)` | move and return the settled read-back |
| `capture_wrist()` / `capture_observer()` | RGB frames |
| `project(point)` / `unproject_to_table(u, v, z)` | the camera model |

`project`/`unproject` live on the backend **on purpose**. MuJoCo and OpenCV disagree about
which way the optical axis points (`-z` vs `+z`); encoding that in the controller produced
a roll mismatch that made every target pixel wrong, and the arm chased a point it could
never reach. The backend knows its own convention.

## What each backend does differently

| | `SimRobot` | `SerialRobot` |
|---|---|---|
| command | ramps `ctrl` over the 1.2 s move time, then settles | one `write6` frame, then a guarded poll |
| read | `qpos` | bus read, first reply discarded |
| settle | fixed 2.5 s of stepping | poll until two samples agree within 0.35 deg |
| safety | none needed | raises when a servo lags 8 deg |

The simulation deliberately copies the hardware's *timing*, not just its geometry: a
controller tuned against an instantaneous simulation behaves differently on an arm that
takes 1.5-2 s to stop.

## Parameters

Every measured constant is in `params.py`, grouped by what it describes, with the source
of each number in the module docstring. A real run can be audited against it.

## CI gate

`sim/grasp_trial.py` runs the controller over four cube positions and exits non-zero
below a rate. It is wired into CI but **not armed**: see the comment in
`.github/workflows/ci.yml`.
