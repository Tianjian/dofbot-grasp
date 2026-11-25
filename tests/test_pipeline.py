"""Pipeline tests: happy path against mock bus+camera, missing-cube abort."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import _DEFAULT_CUBE, default_scene_cube, make_system

from dofbot_grasp.kinematics import forward
from dofbot_grasp.pipeline import (
    STEP_NAMES,
    GraspPipeline,
    PipelineConfig,
    PipelineResult,
    StepRecord,
)


def _fast_cfg(**overrides) -> PipelineConfig:
    values = dict(
        search_yaw_deg=(0.0,),
        move_duration_s=0.0,
        settle_s=0.0,
        gripper_settle_s=0.0,
    )
    values.update(overrides)
    return PipelineConfig(**values)


def _run(cfg: PipelineConfig, cube_world=_DEFAULT_CUBE):
    kwargs = {} if cube_world is _DEFAULT_CUBE else {"cube_world": cube_world}
    scene, bus, camera, cfg = make_system(cfg=cfg, **kwargs)
    pipeline = GraspPipeline(bus, camera, cfg, sleep_fn=lambda _s: None)
    return pipeline.run(), scene, bus, camera


def test_pipeline_happy_path() -> None:
    cfg = _fast_cfg()
    result, scene, bus, _camera = _run(cfg)

    assert isinstance(result, PipelineResult)
    assert result.success, f"pipeline aborted at {result.aborted_step}: {result.reason}"
    assert result.aborted_step is None
    names = [step.name for step in result.steps]
    assert names == list(STEP_NAMES)
    assert all(step.success for step in result.steps)
    assert all(isinstance(step, StepRecord) for step in result.steps)
    assert all(step.joints is not None for step in result.steps)
    assert all(step.elapsed_s >= 0.0 for step in result.steps)

    # checkpoint metrics required by the sim/real interface
    for key in (
        "cube_height_mm",
        "gripper_cube_distance_mm",
        "contact_count",
        "alignment_error_px",
    ):
        assert key in result.metrics
    assert result.metrics["cube_height_mm"] >= 20.0  # harness CONTRACT: >= 20 mm
    assert result.metrics["gripper_cube_distance_mm"] is not None
    assert result.metrics["gripper_cube_distance_mm"] <= 15.0  # CONTRACT: <= 15 mm
    assert result.metrics["alignment_error_px"] is not None
    assert abs(result.metrics["alignment_error_px"]) <= cfg.align_tolerance_px
    assert isinstance(result.metrics["contact_count"], int)

    # the gripper was actually opened then closed
    assert [angle for angle, _ in bus.gripper_commands[0:1]] == [cfg.open_gripper_deg]
    assert any(angle == cfg.grasp_gripper_deg for angle, _ in bus.gripper_commands)

    # the verified lift raised the EE
    lift = result.step("lift")
    verify = result.step("verify")
    assert lift.extra["lifted_m"] >= cfg.verify_min_lift_m
    assert verify.extra["cube_visible"] in (True, False)

    # pipeline read the wrist camera throughout
    assert scene.frames_rendered >= 5


def test_pipeline_aborts_on_missing_cube() -> None:
    cfg = _fast_cfg()
    scene, bus, camera, cfg = make_system(cube_world=None, cfg=cfg)
    result = GraspPipeline(bus, camera, cfg, sleep_fn=lambda _s: None).run()

    assert not result.success
    assert result.aborted_step == "search"
    assert "not found" in result.reason
    assert [step.name for step in result.steps] == ["home", "search"]
    assert result.steps[-1].success is False
    assert "search_yaw_deg_tried" in result.steps[-1].extra
    # no motion was commanded beyond home + the search sweep
    assert len(bus.commands) <= 1 + len(cfg.search_yaw_deg)


def test_pipeline_missing_camera_aborts_cleanly() -> None:
    from dofbot_grasp.bus import SimBus
    from dofbot_grasp.camera import SimCamera

    cfg = _fast_cfg()
    result = GraspPipeline(SimBus(), SimCamera(), cfg, sleep_fn=lambda _s: None).run()
    assert not result.success
    assert result.aborted_step == "search"


def test_align_centres_an_offset_cube() -> None:
    cfg = _fast_cfg()
    home = cfg.home_joints_rad()
    t_home = forward(home)
    # offset the cube sideways in camera/image-x (camera x axis = +X of home EE)
    cube = default_scene_cube(cfg) + 0.03 * t_home[:3, 0]
    scene, bus, camera, cfg = make_system(cube_world=cube, cfg=cfg)
    result = GraspPipeline(bus, camera, cfg, sleep_fn=lambda _s: None).run()

    align = result.step("align")
    assert align is not None
    assert align.success
    assert abs(align.extra["alignment_error_px"]) <= cfg.align_tolerance_px
    assert align.extra["align_iterations"] >= 1
    assert result.success, result.reason


def test_pipeline_result_as_dict_is_json_friendly() -> None:
    cfg = _fast_cfg()
    result, _scene, _bus, _camera = _run(cfg)
    payload = result.as_dict()
    assert payload["success"] is True
    assert [step["name"] for step in payload["steps"]] == list(STEP_NAMES)
    assert isinstance(payload["metrics"], dict)
    import json

    json.dumps(payload)  # must not raise


def test_pipeline_config_from_dict() -> None:
    cfg = PipelineConfig.from_dict(
        {
            "home_servo_deg": [90, 90, 150, 120, 83, 90],
            "search_yaw_deg": [0, -10, 10],
            "align_tolerance_px": 12.0,
            "detect": {"min_area_px": 200},
            "cube_size_m": 0.03,
        }
    )
    assert cfg.search_yaw_deg == (0.0, -10.0, 10.0)
    assert cfg.align_tolerance_px == 12.0
    assert cfg.detect.min_area_px == 200
    assert cfg.cube_size_m == 0.03


def test_pipeline_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown pipeline config keys"):
        PipelineConfig.from_dict({"not_a_real_key": 1})


def test_home_pose_servo_conversion() -> None:
    cfg = PipelineConfig()
    home = cfg.home_joints_rad()
    np.testing.assert_allclose(home, np.radians([0.0, -15.0, -59.0, -85.0, -49.0]), atol=1e-9)
