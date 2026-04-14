from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from vicon_ws.config_cli import get_global_config_path, resolve_scoped_config


_TOOL_CHOICES = [
    "vicon_ws",
    "vicon_ws_traj",
    "vicon_ws_res",
    "vicon_ws_metric_res",
    "vicon_ws_ape",
    "vicon_ws_rpe",
]


def _load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not str(text).strip():
        return {}
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Settings file must be a JSON object: {path}")
    return raw


def _save_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_value(text: str):
    val = str(text).strip()
    if val == "":
        return ""
    try:
        return json.loads(val)
    except Exception:
        return text


def _normalize_key(text: str) -> str:
    return str(text).strip().replace("-", "_")


def _parser_defaults(parser: argparse.ArgumentParser) -> dict:
    out: dict[str, object] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.dest in {"help", "config"}:
            continue
        if action.default is argparse.SUPPRESS:
            continue
        if action.default is None:
            continue
        if isinstance(action.default, (list, tuple)) and len(action.default) == 0:
            continue
        if isinstance(action.default, str) and action.default == "":
            continue
        out[action.dest] = action.default
    return out


def _generate_template(tool_name: str) -> dict:
    def _load_builder(target: str):
        mod_name, fn_name = target.rsplit(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)

    builders = {
        "vicon_ws": "pipeline:build_parser",
        "vicon_ws_traj": "vicon_ws.traj_tool:build_parser",
        "vicon_ws_res": "vicon_ws.benchmark.res:build_parser",
        "vicon_ws_metric_res": "vicon_ws.benchmark.metrics_res:build_parser",
        "vicon_ws_ape": "vicon_ws.ape_tool:_build_parser",
        "vicon_ws_rpe": "vicon_ws.rpe_tool:_build_parser",
    }
    if tool_name not in builders:
        supported = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unsupported tool '{tool_name}'. Supported: {supported}")
    parser = _load_builder(builders[tool_name])()
    cfg = _parser_defaults(parser)
    if tool_name in {"vicon_ws_ape", "vicon_ws_rpe"}:
        cfg["subcommand"] = "tum"
    return cfg


def _default_settings() -> dict:
    try:
        return _generate_template("vicon_ws")
    except Exception:
        return {}


def _ensure_tool_section(data: dict, tool: str) -> dict:
    raw = data.get(tool, None)
    if not isinstance(raw, dict):
        raw = {}
    data[tool] = raw
    return raw


def _resolve_cfg_path(args: argparse.Namespace) -> Path:
    cmd_cfg = str(getattr(args, "config", "") or "").strip()
    top_cfg = str(getattr(args, "config_path", "") or "").strip()
    if cmd_cfg:
        return Path(cmd_cfg).expanduser().resolve()
    if top_cfg:
        return Path(top_cfg).expanduser().resolve()
    return get_global_config_path()


def _subset(data: dict, keys: list[str]) -> dict:
    if not keys:
        return dict(data)
    out: dict = {}
    for key in keys:
        nk = _normalize_key(key)
        if nk in data:
            out[nk] = data[nk]
    return out


def _diff_entries(current: dict, defaults: dict) -> dict:
    out: dict = {}
    for key, value in current.items():
        if key not in defaults or defaults[key] != value:
            out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Global config manager for vicon_ws tools.")
    p.add_argument(
        "--config-path",
        default="",
        help="Custom settings path. Default: ~/.config/vicon_ws/settings.json",
    )
    sub = p.add_subparsers(dest="cmd")
    sub.required = True

    show = sub.add_parser("show", help="Show global settings.")
    show.add_argument("-c", "--config", default="", help="Optional settings file path.")
    show.add_argument(
        "--tool",
        choices=_TOOL_CHOICES,
        default="",
        help="Show effective scoped config for one tool.",
    )
    show.add_argument("--brief", action="store_true", help="Print only JSON payload.")
    show.add_argument("--diff", action="store_true", help="Show only keys differing from defaults.")
    show.add_argument("--json", action="store_true", help="Alias of --brief.")
    show.add_argument("params", nargs="*", help="Optional parameter names to show.")

    setp = sub.add_parser("set", help="Set one or more key-value pairs.")
    setp.add_argument("-c", "--config", default="", help="Optional settings file path.")
    setp.add_argument(
        "--tool",
        choices=_TOOL_CHOICES,
        default="",
        help="Write keys into a tool-specific section instead of root.",
    )
    setp.add_argument("-m", "--merge", default="", help="Merge another JSON config file first.")
    setp.add_argument("--soft", action="store_true", help="Soft merge: do not overwrite existing keys.")
    setp.add_argument("params", nargs=argparse.REMAINDER, help="Pairs: key value [key value ...]")

    unset = sub.add_parser("unset", help="Unset one or more keys.")
    unset.add_argument("-c", "--config", default="", help="Optional settings file path.")
    unset.add_argument(
        "--tool",
        choices=_TOOL_CHOICES,
        default="",
        help="Unset keys from a tool-specific section.",
    )
    unset.add_argument("keys", nargs="+", help="Keys to remove.")

    reset = sub.add_parser("reset", help="Reset settings to defaults.")
    reset.add_argument("-c", "--config", default="", help="Optional settings file path.")
    reset.add_argument(
        "--tool",
        choices=_TOOL_CHOICES,
        default="",
        help="Reset one tool section instead of whole file.",
    )
    reset.add_argument("-y", action="store_true", help="Acknowledge full reset.")
    reset.add_argument("params", nargs="*", help="Optional keys to reset/remove.")

    gen = sub.add_parser("generate", help="Generate template config for a tool.")
    gen.add_argument(
        "--tool",
        default="vicon_ws",
        choices=[
            *(_TOOL_CHOICES),
        ],
        help="Tool name to generate defaults for.",
    )
    gen.add_argument(
        "--out",
        default="",
        help="Optional path to write generated template JSON.",
    )
    return p


def run(args: argparse.Namespace) -> int:
    cfg_path = _resolve_cfg_path(args)
    tool = str(getattr(args, "tool", "") or "").strip()

    if args.cmd == "show":
        raw_data = _load_settings(cfg_path)
        data = resolve_scoped_config(raw_data, tool_name=tool if tool else None)
        if bool(getattr(args, "diff", False)):
            defaults = _generate_template(tool) if tool else _default_settings()
            data = _diff_entries(data, defaults)
        data = _subset(data, [str(x) for x in getattr(args, "params", [])])
        if bool(getattr(args, "brief", False) or getattr(args, "json", False)):
            print(json.dumps(data, ensure_ascii=False))
        else:
            print(f"Path: {cfg_path}")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "set":
        data = _load_settings(cfg_path)
        target = _ensure_tool_section(data, tool) if tool else data
        merge_path = str(getattr(args, "merge", "") or "").strip()
        if merge_path:
            merged_raw = _load_settings(Path(merge_path).expanduser().resolve())
            merged = resolve_scoped_config(merged_raw, tool_name=tool if tool else None)
            if bool(getattr(args, "soft", False)):
                for key, value in merged.items():
                    target.setdefault(_normalize_key(key), value)
            else:
                for key, value in merged.items():
                    target[_normalize_key(key)] = value
        params = [str(x) for x in getattr(args, "params", [])]
        if len(params) % 2 != 0:
            raise ValueError("`set` expects even number of args: key value [key value ...]")
        for i in range(0, len(params), 2):
            target[_normalize_key(params[i])] = _parse_value(params[i + 1])
        _save_settings(cfg_path, data)
        print(f"Saved: {cfg_path}")
        return 0

    if args.cmd == "unset":
        data = _load_settings(cfg_path)
        target = _ensure_tool_section(data, tool) if tool else data
        for key in getattr(args, "keys", []):
            target.pop(_normalize_key(key), None)
        if tool and isinstance(data.get(tool, None), dict) and len(data[tool]) == 0:
            data.pop(tool, None)
        _save_settings(cfg_path, data)
        print(f"Saved: {cfg_path}")
        return 0

    if args.cmd == "reset":
        defaults = _generate_template(tool) if tool else _default_settings()
        params = [str(x) for x in getattr(args, "params", [])]
        if tool and not params:
            data = _load_settings(cfg_path)
            data[tool] = dict(defaults)
            _save_settings(cfg_path, data)
            print(f"Saved: {cfg_path}")
            return 0
        if params:
            data = _load_settings(cfg_path)
            target = _ensure_tool_section(data, tool) if tool else data
            for key in params:
                nk = _normalize_key(key)
                if nk in defaults:
                    target[nk] = defaults[nk]
                else:
                    target.pop(nk, None)
            if tool and isinstance(data.get(tool, None), dict) and len(data[tool]) == 0:
                data.pop(tool, None)
            _save_settings(cfg_path, data)
            print(f"Saved: {cfg_path}")
            return 0
        if not bool(getattr(args, "y", False)):
            raise ValueError("Full reset requires -y acknowledgement.")
        if defaults:
            _save_settings(cfg_path, defaults)
            print(f"Saved defaults: {cfg_path}")
        else:
            if cfg_path.exists():
                cfg_path.unlink()
                print(f"Removed: {cfg_path}")
            else:
                print(f"No settings file: {cfg_path}")
        return 0

    if args.cmd == "generate":
        data = _generate_template(str(args.tool))
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if str(getattr(args, "out", "")).strip():
            out = Path(str(args.out)).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
            print(f"Saved template: {out}")
        else:
            print(text)
        return 0

    raise ValueError(f"Unknown command: {args.cmd}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
