import argparse
import json

import pytest

from epa.config_cli import GLOBAL_CONFIG_ENV, parse_args_with_config, resolve_scoped_config


def test_parse_args_with_config_supports_required_fields(tmp_path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "metrics_json": ["a.json", "b.json"],
                "mode": "aggregate",
            }
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--metrics-json", nargs="+", required=True, dest="metrics_json")
    parser.add_argument("--mode", choices=["auto", "single", "aggregate"], default="auto")

    args = parse_args_with_config(parser, argv=["--config", str(cfg)], config_dest="config")
    assert args.metrics_json == ["a.json", "b.json"]
    assert args.mode == "aggregate"


def test_parse_args_with_config_overrides_cli_values(tmp_path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"count": 7}), encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--count", type=int, default=0)

    args = parse_args_with_config(
        parser,
        argv=["--count", "1", "--config", str(cfg)],
        config_dest="config",
    )
    assert args.count == 7


def test_parse_args_with_config_handles_boolean_actions(tmp_path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"plot": False}), encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.set_defaults(plot=True)

    args = parse_args_with_config(
        parser,
        argv=["--plot", "--config", str(cfg)],
        config_dest="config",
    )
    assert args.plot is False


def test_parse_args_with_config_rejects_unknown_key(tmp_path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"unknown_param": 1}), encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--count", type=int, default=0)

    with pytest.raises(ValueError, match="Unknown config key"):
        parse_args_with_config(parser, argv=["--config", str(cfg)], config_dest="config")


def test_parse_args_with_global_config_defaults(tmp_path, monkeypatch) -> None:
    gcfg = tmp_path / "global.json"
    gcfg.write_text(json.dumps({"count": 9}), encoding="utf-8")
    monkeypatch.setenv(GLOBAL_CONFIG_ENV, str(gcfg))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--count", type=int, default=0)

    args = parse_args_with_config(parser, argv=[], config_dest="config")
    assert args.count == 9


def test_cli_overrides_global_config(tmp_path, monkeypatch) -> None:
    gcfg = tmp_path / "global.json"
    gcfg.write_text(json.dumps({"count": 9}), encoding="utf-8")
    monkeypatch.setenv(GLOBAL_CONFIG_ENV, str(gcfg))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--count", type=int, default=0)

    args = parse_args_with_config(parser, argv=["--count", "3"], config_dest="config")
    assert args.count == 3


def test_config_file_overrides_cli_and_global(tmp_path, monkeypatch) -> None:
    gcfg = tmp_path / "global.json"
    gcfg.write_text(json.dumps({"count": 9}), encoding="utf-8")
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"count": 7}), encoding="utf-8")
    monkeypatch.setenv(GLOBAL_CONFIG_ENV, str(gcfg))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--count", type=int, default=0)

    args = parse_args_with_config(
        parser,
        argv=["--count", "3", "--config", str(cfg)],
        config_dest="config",
    )
    assert args.count == 7


def test_global_unknown_key_is_ignored(tmp_path, monkeypatch) -> None:
    gcfg = tmp_path / "global.json"
    gcfg.write_text(json.dumps({"unknown_param": 1, "count": 4}), encoding="utf-8")
    monkeypatch.setenv(GLOBAL_CONFIG_ENV, str(gcfg))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--count", type=int, default=0)

    args = parse_args_with_config(parser, argv=[], config_dest="config")
    assert args.count == 4


def test_single_bool_flag_allows_default_value_from_global_config(tmp_path, monkeypatch) -> None:
    gcfg = tmp_path / "global.json"
    gcfg.write_text(json.dumps({"plot": False}), encoding="utf-8")
    monkeypatch.setenv(GLOBAL_CONFIG_ENV, str(gcfg))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--plot", action="store_true")

    args = parse_args_with_config(parser, argv=[], config_dest="config")
    assert args.plot is False


def test_resolve_scoped_config_merges_global_and_tool_sections() -> None:
    cfg = {
        "count": 1,
        "_global": {"count": 2, "plot": False},
        "epa_ape": {"count": 3, "pose_relation": "rot_part"},
    }
    out = resolve_scoped_config(cfg, tool_name="epa_ape")
    assert out["count"] == 3
    assert out["plot"] is False
    assert out["pose_relation"] == "rot_part"


def test_parse_args_with_config_reads_tool_section_from_global(tmp_path, monkeypatch) -> None:
    gcfg = tmp_path / "global.json"
    gcfg.write_text(
        json.dumps(
            {
                "_global": {"count": 2},
                "epa_ape": {"count": 5},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(GLOBAL_CONFIG_ENV, str(gcfg))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--count", type=int, default=0)
    args = parse_args_with_config(parser, argv=[], config_dest="config", tool_name="epa_ape")
    assert args.count == 5


def test_parse_args_with_config_reads_tools_container_section(tmp_path, monkeypatch) -> None:
    gcfg = tmp_path / "global.json"
    gcfg.write_text(
        json.dumps(
            {
                "_global": {"count": 2},
                "tools": {
                    "epa_traj": {"count": 8},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(GLOBAL_CONFIG_ENV, str(gcfg))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--count", type=int, default=0)
    args = parse_args_with_config(parser, argv=[], config_dest="config", tool_name="epa_traj")
    assert args.count == 8

