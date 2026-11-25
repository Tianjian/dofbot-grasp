#!/usr/bin/env python3
"""roboharness wedge for the DOFBOT SE wrist-camera grasp.

Runs the same simulation as `sim/run_grasp_sim.py`, but records per-phase metrics
at semantic checkpoints and evaluates them through roboharness' assertion engine,
then writes the approval artefacts the `roboharness gate` command consumes.

Usage (from the workspace root):
    MUJOCO_GL=egl PYTHONPATH=fork/src:dofbot_grasp/src \
        .venv/bin/python dofbot_grasp/harness/run_wedge.py --out /tmp/wedge_good
    ... --fault        # fingers never close -> the FAIL branch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))

import run_grasp_sim as sim  # noqa: E402  (IK + servo-style drive helpers)

try:
    from roboharness.evaluate.assertions import AssertionEngine, MetricAssertion
    from roboharness.evaluate.result import Operator, Severity
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "roboharness is required for the harness wedge. Install the fork that "
        "ships the approval gate:\n"
        '  pip install "roboharness @ git+https://github.com/Tianjian/roboharness.git"'
    ) from exc

PHASES = ["home", "search", "align", "approach", "grasp", "lift", "verify"]

# Hard contract: metric -> (operator, threshold, severity, phase)
CONTRACT = {
    "cube_height_mm": (Operator.GE, 20.0, Severity.CRITICAL, "lift"),
    "contact_count": (Operator.GE, 1.0, Severity.CRITICAL, "grasp"),
    "gripper_cube_distance_mm": (Operator.LE, 15.0, Severity.MAJOR, "grasp"),
    # Note: the camera mount is offset 35 mm from the tool axis, so the cube is
    # never at the image centre. A scripted plan has no servo error to gate; what
    # the wedge checks is that perception actually sees the cube at align time.
    "cube_visible": (Operator.GE, 1.0, Severity.MAJOR, "align"),
}


def measure(model, data, renderer, cube_id, tcp_id) -> dict[str, float]:
    """Metrics for one checkpoint."""
    cube = data.xpos[cube_id].copy()
    tcp = data.site_xpos[tcp_id].copy()
    contacts = 0
    for c in data.contact[: data.ncon]:
        names = {model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)}
        if "cube" in names and any("finger" in n for n in names):
            contacts += 1

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="wrist_cam")
    seg = renderer.render()[:, :, 0]
    renderer.disable_segmentation_rendering()
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    ys, xs = np.where(seg == cube_geom)
    if len(xs):
        h, w = seg.shape
        err = float(np.hypot(xs.mean() - w / 2, ys.mean() - h / 2))
        visible = 1.0
    else:
        err = float("inf")  # cube not visible
        visible = 0.0

    return {
        "cube_height_mm": float(cube[2] * 1000.0),
        "gripper_cube_distance_mm": float(np.linalg.norm(tcp - cube) * 1000.0),
        "contact_count": float(contacts),
        "cube_visible": visible,
        "cube_centroid_offset_px": err,
        "step": float(data.time / model.opt.timestep),
        "sim_time_s": float(data.time),
    }


def run(fault: bool) -> dict:
    model = mujoco.MjModel.from_xml_path(str(sim.XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 480, 640)

    cube_id = model.body("cube").id
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    cube0 = data.xpos[cube_id].copy()
    z = np.array([0.0, 0.0, -1.0])

    seed = np.array([-0.058, 0.909, 0.987, 1.091, 1.127])
    home = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    search = sim.ik_tcp(model, data, cube0 + [0, 0, 0.12], z, seed)
    align = sim.ik_tcp(model, data, cube0 + [0, 0, 0.08], z, search)
    approach = sim.ik_tcp(model, data, cube0 + [0, 0, 0.004], z, align)
    grasp_q = sim.ik_tcp(model, data, cube0 + [0, 0, 0.004], z, approach)
    lift_q = sim.ik_tcp(model, data, cube0 + [0, 0, 0.12], z, grasp_q)

    # ctrl > 0 opens the jaws, ctrl < 0 pinches. The fault keeps them open.
    open_g, closed_g = 0.010, (0.010 if fault else -0.002)
    metrics: dict[str, dict[str, float]] = {}

    def checkpoint(name: str, q, grip: float, steps: int) -> None:
        sim.drive(model, data, q, grip, steps)
        metrics[name] = measure(model, data, renderer, cube_id, tcp_id)

    checkpoint("home", home, open_g, 400)
    checkpoint("search", search, open_g, 1400)
    checkpoint("align", align, open_g, 700)
    checkpoint("approach", approach, open_g, 700)
    checkpoint("grasp", grasp_q, closed_g, 600)
    checkpoint("lift", lift_q, closed_g, 900)
    checkpoint("verify", lift_q, closed_g, 300)

    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fault", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics = run(args.fault)

    assertions = [
        MetricAssertion(metric=m, operator=op, threshold=th, severity=sev, phase=ph)
        for m, (op, th, sev, ph) in CONTRACT.items()
    ]
    report = {
        "schema_version": 1,
        "task": "dofbot_grasp",
        "phase_order": PHASES,
        "snapshot_metrics": metrics,
        "summary_metrics": {
            "lifted_mm": metrics["lift"]["cube_height_mm"] - metrics["home"]["cube_height_mm"],
            "final_contact_count": metrics["verify"]["contact_count"],
        },
    }
    (out / "autonomous_report.json").write_text(json.dumps(report, indent=2))

    result = AssertionEngine(assertions).evaluate(
        report, report_path=str(out / "autonomous_report.json")
    )
    verdict = {v.value if hasattr(v, "value") else str(v): None for v in [result.verdict]}
    raw = next(iter(verdict))
    overall = {"pass": "PASS", "fail": "FAIL", "degraded": "AMBIGUOUS"}[raw]

    alarms = []
    first_bad = None
    for r in result.results:
        ok = bool(r.passed)
        alarms.append(
            {
                "code": f"{r.phase}:{r.metric}",
                "metric": r.metric,
                "phase": r.phase,
                "severity": r.severity.value if hasattr(r.severity, "value") else str(r.severity),
                "status": "ok" if ok else "alarm",
                "actual_value": r.actual_value,
                "expected": r.threshold,
                "message": r.message,
            }
        )
        if not ok and first_bad is None:
            first_bad = r.phase
    (out / "alarms.json").write_text(
        json.dumps({"task": "dofbot_grasp", "alarms": alarms}, indent=2)
    )

    failed = [a for a in alarms if a["status"] == "alarm"]
    manifest = {
        "task": "dofbot_grasp",
        "verdict": overall.lower() if overall != "AMBIGUOUS" else "degraded",
        "failed_phase": first_bad,
        "failed_alarms": [a["code"] for a in failed],
        "rerun_hint": "not_required" if not failed else f"restore:{first_bad}",
        "primary_views": ["wrist_cam"],
    }
    (out / "phase_manifest.json").write_text(json.dumps(manifest, indent=2))

    approval = {
        "schema_version": "roboharness_report/v1",
        "contract_id": "dofbot-grasp-v1",
        "mode": "regression",
        "overall_verdict": overall,
        "stop_reason": "all_rules_satisfied" if not failed else "hard_metric_failed",
        "baseline_authority": "No blessed baseline yet; rules are absolute thresholds.",
        "summary": {
            "cases_total": 1,
            "cases_surfaced": 0 if not failed else 1,
            "cases_suppressed": 1 if not failed else 0,
        },
        "surfaced_cases": [{"case_id": "dofbot_grasp", "status": "FAILED"}] if failed else [],
        "user_action": {
            "needs_review": bool(failed),
            "needs_baseline_blessing": False,
            "review_case_ids": ["dofbot_grasp"] if failed else [],
        },
    }
    (out / "approval_report.json").write_text(json.dumps(approval, indent=2))

    contract = {
        "schema_version": "roboharness_contract/v1",
        "contract_id": "dofbot-grasp-v1",
        "mode": "regression",
        "rules": [
            {
                "metric": m,
                "operator": op.name.lower(),
                "threshold": th,
                "severity": sev.value,
                "phase": ph,
            }
            for m, (op, th, sev, ph) in CONTRACT.items()
        ],
    }
    (out / "contract.json").write_text(json.dumps(contract, indent=2))

    print(
        json.dumps(
            {
                "verdict": overall,
                "failed_phase": first_bad,
                "lifted_mm": report["summary_metrics"]["lifted_mm"],
                "contacts_at_grasp": metrics["grasp"]["contact_count"],
                "cube_visible_at_align": metrics["align"]["cube_visible"],
                "cube_centroid_offset_px": metrics["align"]["cube_centroid_offset_px"],
                "artifacts": sorted(p.name for p in out.iterdir()),
            },
            indent=2,
        )
    )
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
