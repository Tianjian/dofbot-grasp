"""Grasp pipeline state machine for the DOFBOT SE arm.

Named steps match the roboharness ``TaskProtocol`` phases for task
``"dofbot_grasp"`` (see ``docs/controller-architecture.md``)::

    home -> search -> align -> approach -> grasp -> lift -> verify

The pipeline is deliberately dependency-light: it only needs objects satisfying
:class:`~dofbot_grasp.bus.ServoBus` and
:class:`~dofbot_grasp.camera.WristCamera`.  Tests wire a mock bus plus a mock
camera that renders the cube from the current joint state; the MuJoCo
workstream wires ``SimBus``/``SimCamera``.  Every step produces a
:class:`StepRecord` with name, joints, observation, elapsed and success, and a
failed step aborts the run (e.g. a missing cube aborts at ``search``).

Checkpoint metrics (required by the contract) are collected in
``PipelineResult.metrics``: ``cube_height_mm``, ``gripper_cube_distance_mm``,
``contact_count``, ``alignment_error_px``.

The pipeline never raises for expected mission failures; it returns
``PipelineResult(success=False, ...)`` with the failing step recorded.  Actual
programming errors (bad shapes/config) still raise.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .bus import ServoBus
from .camera import Intrinsics, WristCamera, wrist_camera_to_ee
from .kinematics import (
    JOINT_LIMITS,
    clamp_joints,
    forward,
    inverse_position,
    servo_to_joint_deg,
)
from .perception import (
    CubeObservation,
    DetectConfig,
    cube_position_m,
    detect_cube,
)

__all__ = [
    "STEP_NAMES",
    "GraspPipeline",
    "PipelineConfig",
    "PipelineResult",
    "StepRecord",
]

STEP_NAMES = ("home", "search", "align", "approach", "grasp", "lift", "verify")


def _deg6(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape != (6,):
        raise ValueError(f"expected 6 servo angles, got shape {arr.shape}")
    return arr


@dataclass
class PipelineConfig:
    """Tunable parameters for :class:`GraspPipeline`.

    All angles in the config are *logical servo degrees* (what
    ``Arm_serial_servo_write6`` takes: IDs 1-4/6 0-180, ID 5 0-270), which is
    what a DOFBOT operator reads off the Arm_Lib API.  Internally the pipeline
    converts them to joint radians.
    """

    # -- home / search -----------------------------------------------------
    #: 6 servo angles for the home pose; default is a look-down pose that puts
    #: the wrist camera (tool-axis mount) over the table in front of the arm.
    home_servo_deg: tuple[float, ...] = (90.0, 75.0, 31.0, 5.0, 86.0, 90.0)
    #: Base-yaw offsets (degrees) tried during search, relative to home.
    search_yaw_deg: tuple[float, ...] = (0.0, -20.0, 20.0, -40.0, 40.0)
    #: Gripper angle while searching (open).
    open_gripper_deg: float = 90.0

    # -- timing ------------------------------------------------------------
    #: Duration of each joint move passed to the bus (seconds).
    move_duration_s: float = 1.0
    #: Post-move settle time before a camera read / next command (seconds).
    settle_s: float = 0.0
    #: Extra settle time after opening/closing the gripper (seconds).
    gripper_settle_s: float = 0.0

    # -- perception --------------------------------------------------------
    cube_size_m: float = 0.030
    #: Pinhole focal length used when the camera exposes no intrinsics.
    f_px: float = 600.0
    #: Principal point override; ``None`` = image centre.
    cx: float | None = None
    cy: float | None = None
    #: Detection thresholds (HSV ranges etc.).
    detect: DetectConfig = field(default_factory=DetectConfig)
    #: Fixed camera->EE transform when it differs from the default wrist mount.
    camera_to_ee: tuple | None = None

    # -- align -------------------------------------------------------------
    #: Desired cube column in pixels; ``None`` = principal point.
    target_u_px: float | None = None
    #: Horizontal pixel error considered aligned.
    align_tolerance_px: float = 25.0
    #: Fraction of the measured bearing corrected per iteration (1.0 = full).
    align_gain: float = 1.0
    align_max_iters: int = 6
    #: Base-joint movement limit per align iteration (rad).
    align_max_step_rad: float = 0.35

    # -- approach / grasp --------------------------------------------------
    #: Distance from the TCP to the cube centre at the pre-grasp pose, along
    #: the tool axis (+Z of link5).
    approach_standoff_m: float = 0.06
    #: Allowed FK position error after the approach move.
    approach_tolerance_m: float = 0.02
    #: Distance from the TCP to the cube centre at the fine-grasp pose.
    grasp_standoff_m: float = 0.005
    #: Allowed FK position error after the fine grasp move.
    grasp_tolerance_m: float = 0.015
    #: Tool-centre-point offset along the link5 +Z tool axis (metres).  The sim
    #: contract puts the ``tcp`` site 50 mm down the tool axis; set 0 if the
    #: arm's grasp point really is the link5 origin.
    tool_offset_m: float = 0.050
    #: Gripper servo angle when closed.
    grasp_gripper_deg: float = 20.0
    #: Gripper servo angle when open (used for the initial open command).
    gripper_open_deg: float = 90.0

    # -- lift / verify -----------------------------------------------------
    lift_height_m: float = 0.05
    #: Minimum vertical EE rise required to call the lift successful.
    verify_min_lift_m: float = 0.03
    #: Table height for the ``cube_height_mm`` metric (metres).
    table_height_m: float = 0.0
    #: Require the cube to still be visible after lifting.
    verify_require_cube_visible: bool = False
    #: Max allowed TCP-to-cube distance at verify when the cube is visible (m).
    verify_max_cube_distance_m: float = 0.05

    # -- IK ----------------------------------------------------------------
    ik_orientation_weight: float = 0.01
    ik_tol_m: float = 1e-4
    ik_max_iters: int = 400

    #: Raise on unexpected camera errors instead of aborting cleanly.
    strict_camera: bool = False

    # -- derived -----------------------------------------------------------
    def home_joints_rad(self) -> np.ndarray:
        """Home pose as joint radians (clamped to the URDF limits)."""
        deg = servo_to_joint_deg(_deg6(self.home_servo_deg))
        return clamp_joints(np.radians(deg))

    def resolved_detect(self, frame_shape: tuple[int, int] | None = None) -> DetectConfig:
        """Return a :class:`DetectConfig` with pipeline-level overrides applied."""
        cfg = replace(self.detect)
        cfg.cube_size_m = float(self.cube_size_m)
        if cfg.f_px == DetectConfig.f_px:  # untouched by the caller
            cfg.f_px = float(self.f_px)
        if cfg.cx is None and self.cx is not None:
            cfg.cx = float(self.cx)
        if cfg.cy is None and self.cy is not None:
            cfg.cy = float(self.cy)
        return cfg

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineConfig:
        """Build from a mapping (e.g. a YAML ``pipeline`` section).

        Unknown top-level sections used by the CLI (``bus``, ``camera``,
        ``sim``, ``run``) are ignored here; unknown *pipeline* keys raise so
        typos are caught.
        """
        if data is None:
            return cls()
        allowed_extra = {"bus", "camera", "sim", "run", "pipeline"}
        unknown = set(data) - set(cls.__dataclass_fields__) - allowed_extra
        if unknown:
            raise ValueError(f"unknown pipeline config keys: {sorted(unknown)}")
        values: dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key not in data:
                continue
            value = data[key]
            if key == "detect":
                value = DetectConfig(**value) if isinstance(value, Mapping) else value
            elif key in ("home_servo_deg", "search_yaw_deg"):
                value = tuple(float(v) for v in value)
            elif key == "camera_to_ee" and value is not None:
                value = tuple(tuple(float(v) for v in row) for row in value)
            values[key] = value
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        """Load the ``pipeline`` section of a YAML file (or the whole file)."""
        try:
            import yaml  # guarded: only the CLI/config path needs PyYAML
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError("loading a YAML config requires PyYAML") from exc
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        section = data.get("pipeline", data)
        return cls.from_dict(section)


@dataclass
class StepRecord:
    """One executed pipeline step."""

    name: str
    joints: np.ndarray | None
    observation: CubeObservation | None
    elapsed_s: float
    success: bool
    message: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self, include_observation: bool = True) -> dict:
        out: dict[str, Any] = {
            "name": self.name,
            "joints": None if self.joints is None else [float(v) for v in self.joints],
            "elapsed_s": float(self.elapsed_s),
            "success": bool(self.success),
            "message": self.message,
        }
        if include_observation:
            out["observation"] = None if self.observation is None else self.observation.as_dict()
        out.update(self.extra)
        return out


@dataclass
class PipelineResult:
    """Outcome of :meth:`GraspPipeline.run`."""

    success: bool
    steps: list[StepRecord]
    metrics: dict
    aborted_step: str | None = None
    reason: str = ""

    def step(self, name: str) -> StepRecord | None:
        for record in self.steps:
            if record.name == name:
                return record
        return None

    def as_dict(self, include_observation: bool = True) -> dict:
        return {
            "success": bool(self.success),
            "aborted_step": self.aborted_step,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "steps": [s.as_dict(include_observation) for s in self.steps],
        }


class GraspPipeline:
    """Run the 7-step grasp mission against any ServoBus + WristCamera pair.

    Parameters
    ----------
    bus : ServoBus
        Joint command/read interface (``SimBus``, ``SerialBus`` or a mock).
    camera : WristCamera
        Object with ``read() -> BGR ndarray``.
    cfg : PipelineConfig | mapping, optional
        Configuration; mappings are passed through
        :meth:`PipelineConfig.from_dict`.
    detector : callable, optional
        ``detector(frame, cube_size_m=..., intrinsics=..., cfg=...) ->
        CubeObservation | None``; defaults to
        :func:`dofbot_grasp.perception.detect_cube` (injectable for tests).
    clock, sleep_fn : callable, optional
        Time sources, injectable so tests need not sleep.
    """

    def __init__(
        self,
        bus: ServoBus,
        camera: WristCamera,
        cfg: PipelineConfig | Mapping[str, Any] | None = None,
        detector: Callable[..., CubeObservation | None] = detect_cube,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if isinstance(cfg, Mapping):
            cfg = PipelineConfig.from_dict(cfg)
        self.bus = bus
        self.camera = camera
        self.cfg = cfg or PipelineConfig()
        self.detector = detector
        self.clock = clock
        self.sleep_fn = sleep_fn
        self._last_frame: np.ndarray | None = None

    # -- helpers -----------------------------------------------------------
    @property
    def _gripper_bus(self):
        return self.bus if hasattr(self.bus, "write_gripper") else None

    def _camera_to_ee(self) -> np.ndarray:
        if self.cfg.camera_to_ee is not None:
            t = np.asarray(self.cfg.camera_to_ee, dtype=float)
            if t.shape != (4, 4):
                raise ValueError(f"camera_to_ee must be 4x4, got {t.shape}")
            return t
        return wrist_camera_to_ee()

    def _intrinsics(self, frame: np.ndarray | None = None) -> Intrinsics | None:
        intr = getattr(self.camera, "intrinsics", None)
        if isinstance(intr, Intrinsics):
            return intr
        if frame is not None and frame.ndim == 3:
            h, w = int(frame.shape[0]), int(frame.shape[1])
            cx = float(self.cfg.cx) if self.cfg.cx is not None else w / 2.0
            cy = float(self.cfg.cy) if self.cfg.cy is not None else h / 2.0
            return Intrinsics(
                fx=float(self.cfg.f_px),
                fy=float(self.cfg.f_px),
                cx=cx,
                cy=cy,
                width=w,
                height=h,
            )
        return None

    def _read_frame(self) -> np.ndarray | None:
        try:
            frame = self.camera.read()
        except Exception:
            if self.cfg.strict_camera:
                raise
            return None
        return None if frame is None else np.asarray(frame)

    def _detect(self, frame: np.ndarray | None = None) -> CubeObservation | None:
        if frame is None:
            frame = self._read_frame()
        if frame is None or frame.size == 0:
            return None
        self._last_frame = frame
        intr = self._intrinsics(frame)
        cfg = self.cfg.resolved_detect(frame.shape[:2])
        try:
            return self.detector(
                frame,
                cube_size_m=self.cfg.cube_size_m,
                intrinsics=intr,
                cfg=cfg,
            )
        except Exception:
            if self.cfg.strict_camera:
                raise
            return None

    def _move(self, joints_rad, duration_s: float | None = None) -> np.ndarray:
        q = clamp_joints(np.asarray(joints_rad, dtype=float))
        self.bus.write_joints(q, self.cfg.move_duration_s if duration_s is None else duration_s)
        if self.cfg.settle_s > 0:
            self.sleep_fn(self.cfg.settle_s)
        return np.asarray(self.bus.read_joints(), dtype=float)

    def _set_gripper(self, angle_deg: float) -> None:
        bus = self._gripper_bus
        if bus is None:
            return
        bus.write_gripper(float(angle_deg), self.cfg.move_duration_s)
        if self.cfg.gripper_settle_s > 0:
            self.sleep_fn(self.cfg.gripper_settle_s)

    def _ik_position(self, position_m: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Position-only IK used for the grasp moves.

        The 5-DOF arm cannot hold an arbitrary full pose while reaching a
        point, so the pipeline only constrains the EE position for
        approach/grasp/lift and lets the wrist orientation follow.
        """
        return inverse_position(
            position_m,
            seed,
            tol=self.cfg.ik_tol_m,
            max_iters=self.cfg.ik_max_iters,
        )

    def _finish(
        self,
        records: list[StepRecord],
        metrics: dict,
        aborted_step: str | None,
        reason: str,
    ) -> PipelineResult:
        success = aborted_step is None and all(r.success for r in records)
        return PipelineResult(
            success=bool(success),
            steps=records,
            metrics=metrics,
            aborted_step=aborted_step,
            reason=reason,
        )

    def _fail(
        self,
        records: list[StepRecord],
        metrics: dict,
        name: str,
        message: str,
        start_time: float,
        joints=None,
        observation: CubeObservation | None = None,
        extra: dict | None = None,
    ) -> PipelineResult:
        records.append(
            StepRecord(
                name=name,
                joints=None if joints is None else np.asarray(joints, dtype=float),
                observation=observation,
                elapsed_s=max(0.0, self.clock() - start_time),
                success=False,
                message=message,
                extra=extra or {},
            )
        )
        metrics.setdefault("alignment_error_px", None)
        return self._finish(records, metrics, name, message)

    # -- main --------------------------------------------------------------
    def run(self, seed_joints: Sequence[float] | None = None) -> PipelineResult:
        """Execute all seven steps; returns a :class:`PipelineResult`."""
        cfg = self.cfg
        records: list[StepRecord] = []
        metrics: dict = {
            "cube_height_mm": 0.0,
            "gripper_cube_distance_mm": None,
            "contact_count": 0,
            "alignment_error_px": None,
        }
        home = cfg.home_joints_rad()
        if seed_joints is not None:
            try:
                current = clamp_joints(seed_joints)
            except ValueError:
                current = home
        else:
            try:
                current = clamp_joints(np.asarray(self.bus.read_joints(), dtype=float))
            except Exception:
                current = home

        # -- 1. home ------------------------------------------------------
        t0 = self.clock()
        try:
            current = self._move(home)
            self._set_gripper(cfg.open_gripper_deg)
            ok = True
            message = "moved to home pose and opened gripper"
        except Exception as exc:  # command failure -> abort cleanly
            return self._fail(records, metrics, "home", f"home failed: {exc}", t0, current)
        records.append(
            StepRecord("home", current.copy(), None, max(0.0, self.clock() - t0), ok, message)
        )

        # -- 2. search ----------------------------------------------------
        t0 = self.clock()
        found: CubeObservation | None = None
        searched: list[float] = []
        search_joints = current.copy()
        search_error = ""
        for yaw_deg in cfg.search_yaw_deg:
            q = current.copy()
            q[0] = float(np.clip(home[0] + np.radians(float(yaw_deg)), *JOINT_LIMITS[0]))
            try:
                q = self._move(q)
            except Exception as exc:
                search_error = f"search move failed: {exc}"
                break
            searched.append(float(yaw_deg))
            obs = self._detect()
            if obs is not None:
                found = obs
                search_joints = q
                break
        if found is None:
            return self._fail(
                records,
                metrics,
                "search",
                search_error or "cube not found (no detection at any search yaw)",
                t0,
                search_joints,
                extra={"search_yaw_deg_tried": searched},
            )
        records.append(
            StepRecord(
                "search",
                search_joints.copy(),
                found,
                max(0.0, self.clock() - t0),
                True,
                f"cube found after yaw {searched[-1]:.1f} deg",
                {"search_yaw_deg_tried": searched},
            )
        )

        # -- 3. align -----------------------------------------------------
        t0 = self.clock()
        target_u = float(cfg.target_u_px) if cfg.target_u_px is not None else None
        obs = found
        aligned = False
        align_iters = 0
        align_error_px = None
        try:
            current = np.asarray(self.bus.read_joints(), dtype=float)
        except Exception:
            current = search_joints
        for _ in range(max(1, cfg.align_max_iters)):
            align_iters += 1
            obs = self._detect()
            if obs is None:
                break
            intr = self._intrinsics(self._last_frame)
            cx = (
                intr.cx
                if intr is not None
                else (float(cfg.cx) if cfg.cx is not None else float(obs.center_px[0]))
            )
            if target_u is None:
                target_u = float(cx)
            align_error_px = float(obs.center_px[0] - target_u)
            if abs(align_error_px) <= cfg.align_tolerance_px:
                aligned = True
                break
            delta = -cfg.align_gain * float(obs.bearing_rad)
            delta = float(np.clip(delta, -cfg.align_max_step_rad, cfg.align_max_step_rad))
            q = current.copy()
            q[0] = float(np.clip(q[0] + delta, *JOINT_LIMITS[0]))
            current = self._move(q)
        if not aligned:
            return self._fail(
                records,
                metrics,
                "align",
                "could not centre the cube within the pixel tolerance"
                if obs is not None
                else "cube lost during align",
                t0,
                current,
                observation=obs,
                extra={
                    "alignment_error_px": align_error_px,
                    "align_iterations": align_iters,
                },
            )
        metrics["alignment_error_px"] = align_error_px
        records.append(
            StepRecord(
                "align",
                current.copy(),
                obs,
                max(0.0, self.clock() - t0),
                True,
                f"aligned at {align_error_px:+.1f} px",
                {"alignment_error_px": align_error_px, "align_iterations": align_iters},
            )
        )

        # -- 4. approach --------------------------------------------------
        t0 = self.clock()
        frame = self._read_frame()
        obs = self._detect(frame)
        if obs is None:
            return self._fail(
                records, metrics, "approach", "cube not visible for approach", t0, current
            )
        intr = self._intrinsics(frame)
        if intr is None:
            return self._fail(
                records, metrics, "approach", "camera intrinsics unavailable", t0, current, obs
            )
        t_base_ee = forward(current)
        t_base_cam = t_base_ee @ self._camera_to_ee()
        cube_base = cube_position_m(obs, intr, t_base_cam)
        approach_axis = t_base_ee[:3, 2].copy()  # +Z of link5 = tool/grasp axis
        link5_standoff = cfg.approach_standoff_m + cfg.tool_offset_m
        pre_grasp = cube_base - link5_standoff * approach_axis
        try:
            q_approach = self._ik_position(pre_grasp, current)
            current = self._move(q_approach)
        except Exception as exc:
            return self._fail(
                records, metrics, "approach", f"approach move failed: {exc}", t0, current, obs
            )
        approach_error = float(np.linalg.norm(forward(current)[:3, 3] - pre_grasp))
        if approach_error > cfg.approach_tolerance_m:
            return self._fail(
                records,
                metrics,
                "approach",
                f"pre-grasp position error {approach_error * 1000:.1f} mm exceeds "
                f"{cfg.approach_tolerance_m * 1000:.1f} mm",
                t0,
                current,
                obs,
                {
                    "position_error_m": approach_error,
                    "cube_position_base_m": cube_base.tolist(),
                },
            )
        records.append(
            StepRecord(
                "approach",
                current.copy(),
                obs,
                max(0.0, self.clock() - t0),
                True,
                f"pre-grasp reached ({approach_error * 1000:.1f} mm error)",
                {
                    "position_error_m": approach_error,
                    "cube_position_base_m": cube_base.tolist(),
                    "standoff_m": cfg.approach_standoff_m,
                    "tool_offset_m": cfg.tool_offset_m,
                },
            )
        )
        last_cube_base = cube_base

        # -- 5. grasp -----------------------------------------------------
        t0 = self.clock()
        obs_grasp = self._detect()
        if obs_grasp is not None:
            intr_g = self._intrinsics(self._last_frame)
            if intr_g is not None:
                try:
                    last_cube_base = cube_position_m(
                        obs_grasp, intr_g, forward(current) @ self._camera_to_ee()
                    )
                except Exception:
                    pass
        axis = forward(current)[:3, 2].copy()  # tool axis
        grasp_target = last_cube_base - (cfg.grasp_standoff_m + cfg.tool_offset_m) * axis
        try:
            q_grasp = self._ik_position(grasp_target, current)
            current = self._move(q_grasp)
            self._set_gripper(cfg.grasp_gripper_deg)
        except Exception as exc:
            return self._fail(
                records, metrics, "grasp", f"grasp move failed: {exc}", t0, current, obs_grasp
            )
        grasp_error = float(np.linalg.norm(forward(current)[:3, 3] - grasp_target))
        if grasp_error > cfg.grasp_tolerance_m:
            return self._fail(
                records,
                metrics,
                "grasp",
                f"fine grasp position error {grasp_error * 1000:.1f} mm exceeds "
                f"{cfg.grasp_tolerance_m * 1000:.1f} mm",
                t0,
                current,
                obs_grasp,
                {"position_error_m": grasp_error},
            )
        grasp_ee_z = float(forward(current)[2, 3])
        records.append(
            StepRecord(
                "grasp",
                current.copy(),
                obs_grasp,
                max(0.0, self.clock() - t0),
                True,
                f"fingers closed at {cfg.grasp_gripper_deg:.0f} deg "
                f"({grasp_error * 1000:.1f} mm error)",
                {"position_error_m": grasp_error, "gripper_deg": cfg.grasp_gripper_deg},
            )
        )

        # -- 6. lift ------------------------------------------------------
        t0 = self.clock()
        lift_target = forward(current)[:3, 3] + np.array([0.0, 0.0, cfg.lift_height_m])
        try:
            q_lift = self._ik_position(lift_target, current)
            current = self._move(q_lift)
        except Exception as exc:
            return self._fail(
                records, metrics, "lift", f"lift move failed: {exc}", t0, current, obs_grasp
            )
        lifted_m = float(forward(current)[2, 3] - grasp_ee_z)
        lift_ok = lifted_m >= cfg.verify_min_lift_m
        if not lift_ok:
            return self._fail(
                records,
                metrics,
                "lift",
                f"lifted only {lifted_m * 1000:.1f} mm (< {cfg.verify_min_lift_m * 1000:.1f} mm)",
                t0,
                current,
                obs_grasp,
                {"lifted_m": lifted_m},
            )
        records.append(
            StepRecord(
                "lift",
                current.copy(),
                obs_grasp,
                max(0.0, self.clock() - t0),
                True,
                f"lifted {lifted_m * 1000:.1f} mm",
                {"lifted_m": lifted_m},
            )
        )

        # -- 7. verify ----------------------------------------------------
        t0 = self.clock()
        ee = forward(current)
        tcp = ee[:3, 3] + ee[:3, :3] @ np.array([0.0, 0.0, cfg.tool_offset_m])
        obs_verify = self._detect()
        intr_v = self._intrinsics(self._last_frame)
        if obs_verify is not None and intr_v is not None:
            cube_visible = True
            cube_est = cube_position_m(obs_verify, intr_v, ee @ self._camera_to_ee())
        else:
            cube_visible = False
            cube_est = last_cube_base + np.array([0.0, 0.0, max(0.0, lifted_m)])
        gripper_cube_distance_m = float(np.linalg.norm(cube_est - tcp))
        contact_count = 0
        getter = getattr(self.bus, "contact_count", None)
        if callable(getter):
            try:
                contact_count = int(getter())
            except Exception:
                contact_count = 0
        elif isinstance(getter, (int, float)):
            contact_count = int(getter)
        cube_height_mm = 1000.0 * (float(cube_est[2]) - float(cfg.table_height_m))
        metrics.update(
            {
                "cube_height_mm": float(cube_height_mm),
                "gripper_cube_distance_mm": float(gripper_cube_distance_m * 1000.0),
                "contact_count": int(contact_count),
                "alignment_error_px": metrics.get("alignment_error_px"),
            }
        )
        verified = True
        reasons: list[str] = []
        if lifted_m < cfg.verify_min_lift_m:
            verified = False
            reasons.append("insufficient lift")
        if cfg.verify_require_cube_visible and not cube_visible:
            verified = False
            reasons.append("cube not visible after lift")
        if cube_visible and gripper_cube_distance_m > cfg.verify_max_cube_distance_m:
            verified = False
            reasons.append(f"cube drifted {gripper_cube_distance_m * 1000:.0f} mm from the EE")
        observation_used = obs_verify if cube_visible else obs_grasp
        if not verified:
            return self._fail(
                records,
                metrics,
                "verify",
                "verification failed: " + ", ".join(reasons),
                t0,
                current,
                observation_used,
                {"gripper_cube_distance_mm": metrics["gripper_cube_distance_mm"]},
            )
        records.append(
            StepRecord(
                "verify",
                current.copy(),
                observation_used,
                max(0.0, self.clock() - t0),
                True,
                "grasp verified (lift + hold + distance)",
                {
                    "gripper_cube_distance_mm": metrics["gripper_cube_distance_mm"],
                    "cube_visible": cube_visible,
                },
            )
        )
        return self._finish(records, metrics, None, "")
