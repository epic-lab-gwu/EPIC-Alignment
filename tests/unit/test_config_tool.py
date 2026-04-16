import json
from pathlib import Path

from epa import config_tool


def test_config_tool_set_show_unset_reset(tmp_path: Path) -> None:
    cfg_path = tmp_path / "settings.json"
    parser = config_tool.build_parser()

    args = parser.parse_args(
        [
            "--config-path",
            str(cfg_path),
            "set",
            "plot",
            "false",
            "rpe_delta",
            "3",
        ]
    )
    assert config_tool.run(args) == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["plot"] is False
    assert data["rpe_delta"] == 3

    args = parser.parse_args(["--config-path", str(cfg_path), "unset", "plot"])
    assert config_tool.run(args) == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "plot" not in data
    assert data["rpe_delta"] == 3

    args = parser.parse_args(["--config-path", str(cfg_path), "reset", "-y"])
    assert config_tool.run(args) == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "plot" in data


def test_config_tool_generate_template(tmp_path: Path) -> None:
    out = tmp_path / "template.json"
    parser = config_tool.build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--tool",
            "epa_rpe",
            "--out",
            str(out),
        ]
    )
    assert config_tool.run(args) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pose_relation"] == "trans_part"
    assert data["delta"] == 1.0
    assert data["subcommand"] == "tum"


def test_config_tool_show_diff_and_merge_soft(tmp_path: Path) -> None:
    cfg_path = tmp_path / "settings.json"
    merge_path = tmp_path / "merge.json"
    merge_path.write_text(json.dumps({"plot": True, "rpe_delta": 5}), encoding="utf-8")
    parser = config_tool.build_parser()

    args = parser.parse_args(
        [
            "--config-path",
            str(cfg_path),
            "set",
            "-m",
            str(merge_path),
            "--soft",
            "plot",
            "false",
        ]
    )
    assert config_tool.run(args) == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["plot"] is False
    assert data["rpe_delta"] == 5

    args = parser.parse_args(["--config-path", str(cfg_path), "show", "--brief", "--diff"])
    assert config_tool.run(args) == 0


def test_config_tool_set_and_reset_tool_section(tmp_path: Path) -> None:
    cfg_path = tmp_path / "settings.json"
    parser = config_tool.build_parser()

    args = parser.parse_args(
        [
            "--config-path",
            str(cfg_path),
            "set",
            "--tool",
            "epa_ape",
            "plot_mode",
            "\"xy\"",
            "t_max_diff",
            "0.05",
        ]
    )
    assert config_tool.run(args) == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["epa_ape"]["plot_mode"] == "xy"
    assert data["epa_ape"]["t_max_diff"] == 0.05

    args = parser.parse_args(
        [
            "--config-path",
            str(cfg_path),
            "reset",
            "--tool",
            "epa_ape",
        ]
    )
    assert config_tool.run(args) == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["epa_ape"]["subcommand"] == "tum"


def test_config_tool_unset_tool_section_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "settings.json"
    cfg_path.write_text(
        json.dumps({"epa_rpe": {"delta": 3, "plot_mode": "xy"}}, indent=2),
        encoding="utf-8",
    )
    parser = config_tool.build_parser()
    args = parser.parse_args(
        [
            "--config-path",
            str(cfg_path),
            "unset",
            "--tool",
            "epa_rpe",
            "plot_mode",
        ]
    )
    assert config_tool.run(args) == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "plot_mode" not in data["epa_rpe"]
    assert data["epa_rpe"]["delta"] == 3


def test_config_tool_set_on_empty_existing_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "settings.json"
    cfg_path.write_text("", encoding="utf-8")
    parser = config_tool.build_parser()
    args = parser.parse_args(
        [
            "--config-path",
            str(cfg_path),
            "set",
            "--tool",
            "epa_ape",
            "plot_mode",
            "xy",
        ]
    )
    assert config_tool.run(args) == 0
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["epa_ape"]["plot_mode"] == "xy"
