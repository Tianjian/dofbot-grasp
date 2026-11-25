# Evidence: simulation validation

All numbers below were produced by running the commands in this repository
(`mujoco 4.0.1`, headless `MUJOCO_GL=egl`). Nothing is hand-written.

## 1. Reference motion (`sim/run_grasp_sim.py`)

Solves IK for pre-grasp / grasp / lift and drives the position servos the way the
real servos are driven.

```
$ python sim/run_grasp_sim.py --out /tmp/refsim
```

| metric | value |
|---|---|
| `lifted_mm` | **107.99** |
| `finger_cube_contacts` | **8** |
| `verdict` | **PASS** |

## 2. roboharness wedge (`harness/run_wedge.py`)

Seven semantic checkpoints (`home → search → align → approach → grasp → lift → verify`),
four hard rules, evaluated through roboharness' assertion engine.

```
$ python harness/run_wedge.py --out /tmp/wedge_good
$ python harness/run_wedge.py --out /tmp/wedge_fault --fault
```

| run | verdict | first failing phase | `lifted_mm` | contacts at grasp | gate exit |
|---|---|---|---|---|---|
| normal | **PASS** | — | **106.21** | **4** | **0** |
| fault (fingers never close) | **FAIL** | `lift` | **0.0** | **0** | **1** |

```
$ roboharness gate /tmp/wedge_good/approval_report.json
Gate: PASS (exit 0)
  reason: no material change surfaced; evidence complete

$ roboharness gate /tmp/wedge_fault/approval_report.json
Gate: FAIL (exit 1)
  reason: a hard contract rule was violated
  surfaced: dofbot_grasp
  action: human review required
```

## 3. The contract

| metric | phase | rule | severity |
|---|---|---|---|
| `cube_height_mm` | lift | ≥ 20 mm | critical |
| `contact_count` | grasp | ≥ 1 | critical |
| `gripper_cube_distance_mm` | grasp | ≤ 15 mm | major |
| `cube_visible` | align | ≥ 1 (cube detected in `wrist_cam`) | major |

Artefacts per run: `contract.json`, `autonomous_report.json`, `alarms.json`,
`phase_manifest.json`, `approval_report.json`.

## 4. Honest note on one metric

`cube_centroid_offset_px` is ~188 px at `align`. That is **not** an error: the camera
is mounted 35 mm off the tool axis, so the cube is never at the image centre. The
scripted reference does no visual servoing, so this value is recorded but **not**
gated; only the real closed loop (see `docs/deploy-runbook.md`) drives it down.
Turning it into a threshold would encode the mount geometry as a defect.

## 5. Not covered here

- No `report.html`: the full roboharness proof surface needs `Harness` capture
  streams; this wedge drives the MJCF directly so it can reuse the reference
  motion, and therefore emits the JSON artefacts only.
- No hardware data: servo/serial/camera parameters come from the prior-art
  analysis; anything marked `[待实测]` in the runbook still needs a bench
  measurement.
