"""CLI tests: argument handling, clean aborts and config errors (no hardware)."""

from __future__ import annotations

import importlib.util
import json

import pytest

from dofbot_grasp import cli


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_arg_parser().parse_args(["--help"])
    assert excinfo.value.code == 0


def test_requires_a_mode() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_arg_parser().parse_args([])
    assert excinfo.value.code != 0


def test_sim_without_adapter_aborts_cleanly(tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "pipeline:\n  search_yaw_deg: [0]\n  move_duration_s: 0.0\n  settle_s: 0.0\n",
        encoding="utf-8",
    )
    code = cli.main(["sim", "--config", str(config)])
    out = capsys.readouterr().out
    assert code == 1
    assert "ABORTED" in out
    assert "aborted at step: search" in out


def test_sim_json_output(tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("pipeline:\n  search_yaw_deg: [0]\n", encoding="utf-8")
    code = cli.main(["sim", "--config", str(config), "--json"])
    stdout = capsys.readouterr().out
    assert code == 1
    payload = json.loads(stdout)
    assert payload["success"] is False
    assert payload["aborted_step"] == "search"


def test_bad_config_returns_2(tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("pipeline:\n  typo_key: 1\n", encoding="utf-8")
    code = cli.main(["sim", "--config", str(config)])
    assert code == 2
    assert "configuration error" in capsys.readouterr().err


def test_real_mode_reports_missing_pyserial(capsys) -> None:
    if importlib.util.find_spec("serial") is not None:
        pytest.skip("pyserial installed; hardware would be attempted")
    code = cli.main(["real"])
    assert code == 2
    assert "pyserial" in capsys.readouterr().err


def test_build_system_sim_defaults() -> None:
    bus, camera = cli.build_system("sim", {})
    from dofbot_grasp.bus import SimBus
    from dofbot_grasp.camera import SimCamera

    assert isinstance(bus, SimBus)
    assert isinstance(camera, SimCamera)
