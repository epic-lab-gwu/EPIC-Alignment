import argparse
import json

import pytest

from epa.config_cli import GLOBAL_CONFIG_ENV, parse_args_with_config, resolve_scoped_config
from epa.cli import _normalize_inputs, _normalize_mode_args, build_parser
from epa.compat.ov_eval.parsers import _build_error_singlerun_parser


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


def test_epa_cli_debug_flag_defaults_off() -> None:
    parser = build_parser()

    default_args = parser.parse_args([])
    debug_args = parser.parse_args(["--debug"])

    assert default_args.debug is False
    assert debug_args.debug is True


@pytest.mark.parametrize(
    ("option", "destination"),
    [
        ("--disable-time-offset-calibration", "disable_time_offset_calibration"),
        ("--disable-extrinsic-calibration", "disable_extrinsic_calibration"),
        ("--disable-calibration", "disable_calibration"),
        ("--disable-identity-safeguard", "disable_identity_safeguard"),
    ],
)
def test_epa_cli_exposes_calibration_disable_options(option: str, destination: str) -> None:
    args = build_parser().parse_args([option])

    assert getattr(args, destination) is True


@pytest.mark.parametrize(
    ("option", "destination"),
    [
        ("--epa-disable-time-offset", "epa_disable_time_offset_calibration"),
        ("--epa-disable-extrinsic", "epa_disable_extrinsic_calibration"),
        ("--epa-disable-calibration", "epa_disable_calibration"),
        ("--epa-disable-identity-safeguard", "epa_disable_identity_safeguard"),
    ],
)
def test_ov_cli_exposes_calibration_disable_options(option: str, destination: str) -> None:
    parser = _build_error_singlerun_parser()
    args = parser.parse_args(["se3", "gt.txt", "est.txt", option])

    assert getattr(args, destination) is True


def test_epa_cli_exposes_public_mode_and_maps_to_eval_align() -> None:
    parser = build_parser()
    args = _normalize_mode_args(parser, parser.parse_args(["--mode", "sim3"]))

    assert args.mode == "sim3"
    assert args.eval_align == "sim3"


def test_identity_safeguard_defaults_on_in_main_cli():
    assert build_parser().parse_args([]).disable_identity_safeguard is False


def test_rotation_information_ratio_threshold_is_configurable_in_main_cli():
    parser = build_parser()
    assert parser.parse_args([]).min_rotation_information_ratio == 0.03
    args = parser.parse_args(["--min-rotation-information-ratio", "0.01"])
    assert args.min_rotation_information_ratio == 0.01


@pytest.mark.parametrize("command", ["singlerun", "dataset", "comparison"])
def test_identity_safeguard_option_in_all_ov_commands(command):
    from epa.compat.ov_eval import parsers
    from epa.compat.ov_eval.evaluate import _epa_eval_kwargs
    parser = getattr(parsers, f"_build_error_{command}_parser")()
    argv = ["se3", "gt", "est"]
    default = parser.parse_args(argv)
    disabled = parser.parse_args(argv + ["--epa-disable-identity-safeguard"])
    assert _epa_eval_kwargs(default)["epa_disable_identity_safeguard"] is False
    assert _epa_eval_kwargs(disabled)["epa_disable_identity_safeguard"] is True


@pytest.mark.parametrize("mode", ["se3", "se3r", "se3-original", "se3-orginal"])
def test_epa_cli_accepts_new_and_original_se3_names(mode: str) -> None:
    parser = build_parser()
    args = _normalize_mode_args(parser, parser.parse_args(["--mode", mode]))

    assert args.mode == mode
    assert args.eval_align == mode


def test_epa_cli_keeps_hidden_eval_align_for_compatibility() -> None:
    parser = build_parser()
    args = _normalize_mode_args(parser, parser.parse_args(["--eval-align", "epica_sim3_stable"]))

    assert args.mode == ""
    assert args.eval_align == "epica_sim3_stable"


def test_epa_cli_keeps_single_input_legacy_entry() -> None:
    parser = build_parser()
    args = _normalize_inputs(parser, parser.parse_args(["gt.txt"]))

    assert args.gt_csv == "gt.txt"
    assert args.est_path == "gt.txt"


def test_epa_cli_two_positional_inputs_are_gt_and_est() -> None:
    parser = build_parser()
    args = _normalize_inputs(parser, parser.parse_args(["gt.txt", "est.txt"]))

    assert args.gt_csv == "gt.txt"
    assert args.est_path == "est.txt"


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
