"""Command line entry point.

Usage::

    python -m dofbot_grasp.cli sim  --config path/to/config.yaml
    python -m dofbot_grasp.cli real --config path/to/config.yaml
    python -m dofbot_grasp.cli --help

The YAML file has optional top-level sections ``pipeline``, ``bus``, ``camera``
and ``sim``::

    pipeline:
      home_servo_deg: [90, 75, 31, 5, 86, 90]
      search_yaw_deg: [0, -20, 20]
      align_tolerance_px: 25
      cube_size_m: 0.03
      f_px: 600

    bus:            # real mode only
      port: /dev/myserial
      baudrate: 115200
    camera:         # real mode only
      index: 0
      width: 640
      height: 480

    sim:            # sim mode only; module with make_bus()/make_camera()
      adapter: my_project.sim_adapter

Exit code is 0 when the pipeline verifies the grasp, 1 when it aborts (e.g.
missing cube), 2 for configuration/argument errors.  ``--json`` prints the
machine-readable result (steps + checkpoint metrics).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from typing import Any

from .bus import SerialBus, ServoBus, SimBus
from .camera import SimCamera, UsbCamera, WristCamera
from .pipeline import GraspPipeline, PipelineConfig, PipelineResult

__all__ = ["build_arg_parser", "build_system", "load_config", "main"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dofbot_grasp.cli",
        description="Run the DOFBOT SE cube-grasp pipeline in sim or on real hardware.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, help_text in (
        ("sim", "run against the simulator (needs a sim adapter or SimCamera wiring)"),
        ("real", "run against the physical arm over the Arm_Lib serial bus"),
    ):
        p = sub.add_parser(mode, help=help_text)
        p.add_argument("--config", default=None, help="YAML configuration file")
        p.add_argument("--json", action="store_true", help="print the full result as JSON")
        p.add_argument(
            "--seed",
            default=None,
            help="comma-separated seed joint radians (default: read from the bus)",
        )
    return parser


def load_config(path: str | None) -> dict[str, Any]:
    """Load a YAML config file (``{}`` when no path is given)."""
    if not path:
        return {}
    try:
        import yaml  # guarded: only needed for --config
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError("--config requires PyYAML") from exc
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return dict(data)


def _pipeline_config(data: Mapping[str, Any]) -> PipelineConfig:
    section = data.get("pipeline", data)
    if not isinstance(section, Mapping):
        raise ValueError("'pipeline' section must be a mapping")
    return PipelineConfig.from_dict(section)


def build_system(mode: str, data: Mapping[str, Any]) -> tuple[ServoBus, WristCamera]:
    """Construct ``(bus, camera)`` for the requested mode."""
    bus: ServoBus
    camera: WristCamera
    if mode == "real":
        bus_cfg = data.get("bus") or {}
        cam_cfg = data.get("camera") or {}
        if not isinstance(bus_cfg, Mapping) or not isinstance(cam_cfg, Mapping):
            raise ValueError("'bus' and 'camera' sections must be mappings")
        bus = SerialBus(
            port=str(bus_cfg.get("port", "/dev/myserial")),
            baudrate=int(bus_cfg.get("baudrate", 115200)),
            timeout=float(bus_cfg.get("timeout", 0.2)),
        )
        camera = UsbCamera(
            index=int(cam_cfg.get("index", 0)),
            width=int(cam_cfg.get("width", 640)),
            height=int(cam_cfg.get("height", 480)),
        )
        return bus, camera

    if mode == "sim":
        sim_cfg = data.get("sim") or {}
        if not isinstance(sim_cfg, Mapping):
            raise ValueError("'sim' section must be a mapping")
        adapter_name = sim_cfg.get("adapter")
        if adapter_name:
            module = importlib.import_module(str(adapter_name))
            bus = module.make_bus()
            camera = module.make_camera()
            return bus, camera
        # No adapter: ideal in-memory bus + unwired placeholder camera.  The
        # pipeline will abort at search with a clear camera error message.
        camera = SimCamera()
        return SimBus(), camera

    raise ValueError(f"unknown mode {mode!r}")


def _print_human(result: PipelineResult) -> None:
    status = "SUCCESS" if result.success else "ABORTED"
    print(f"pipeline: {status}")
    if not result.success:
        print(f"  aborted at step: {result.aborted_step}")
        print(f"  reason: {result.reason}")
    for step in result.steps:
        mark = "ok " if step.success else "FAIL"
        extra = f"  ({step.message})" if step.message else ""
        print(f"  [{mark}] {step.name:9s} {step.elapsed_s * 1000:8.1f} ms{extra}")
    print("checkpoint metrics:")
    for key in (
        "cube_height_mm",
        "gripper_cube_distance_mm",
        "contact_count",
        "alignment_error_px",
    ):
        value = result.metrics.get(key)
        if isinstance(value, float):
            value = round(value, 3)
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        data = load_config(args.config)
        cfg = _pipeline_config(data)
        bus, camera = build_system(args.mode, data)
    except Exception as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    seed = None
    if args.seed:
        try:
            seed = [float(v) for v in args.seed.split(",")]
        except ValueError:
            print("--seed must be comma-separated floats", file=sys.stderr)
            return 2

    pipeline = GraspPipeline(bus, camera, cfg)
    try:
        result = pipeline.run(seed_joints=seed)
    except Exception as exc:  # unexpected runtime failure (camera/serial)
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2
    finally:
        for resource in (bus, camera):
            release = getattr(resource, "release", None) or getattr(resource, "close", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        _print_human(result)
    return 0 if result.success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
