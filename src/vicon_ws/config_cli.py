from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GLOBAL_CONFIG_ENV = "VICON_WS_GLOBAL_CONFIG"


def _normalize_key(key: str) -> str:
    text = str(key).strip()
    while text.startswith("-"):
        text = text[1:]
    return text.replace("-", "_")


def _preferred_option(action) -> str:
    long_opts = [o for o in action.option_strings if o.startswith("--")]
    if long_opts:
        return max(long_opts, key=len)
    return action.option_strings[0]


def _load_json_config(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must be an object (key-value mapping).")
    return cfg


def get_global_config_path() -> Path:
    override = os.getenv(GLOBAL_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".config" / "vicon_ws" / "settings.json"


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config JSON must be an object: {path}")
    return cfg


def config_to_argv(
    parser,
    config: dict,
    reserved_keys: set[str] | None = None,
    ignore_unknown: bool = False,
) -> list[str]:
    reserved = set(reserved_keys or set())
    actions_by_dest: dict[str, list] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        actions_by_dest.setdefault(action.dest, []).append(action)

    argv: list[str] = []
    for raw_key, value in config.items():
        key = _normalize_key(raw_key)
        if key in reserved or key == "help":
            continue
        candidates = actions_by_dest.get(key, [])
        if not candidates:
            if ignore_unknown:
                continue
            raise ValueError(f"Unknown config key: {raw_key}")

        bool_actions = [
            a for a in candidates
            if getattr(a, "nargs", None) == 0 and hasattr(a, "const")
        ]
        if bool_actions:
            if not isinstance(value, bool):
                raise ValueError(f"Config key '{raw_key}' expects boolean value.")
            chosen = next((a for a in bool_actions if getattr(a, "const", None) is value), None)
            if chosen is None:
                default_value = candidates[0].default if candidates else None
                # Single boolean flag case (e.g. only --plot/store_true):
                # when config value equals parser default, no argv token is needed.
                if isinstance(default_value, bool) and default_value is value:
                    continue
                raise ValueError(f"Config key '{raw_key}' cannot represent value: {value}")
            argv.append(_preferred_option(chosen))
            continue

        action = candidates[0]
        opt = _preferred_option(action)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            argv.append(opt)
            argv.extend([str(v) for v in value])
        else:
            argv.extend([opt, str(value)])
    return argv


def parse_args_with_config(parser, argv=None, config_dest: str = "config"):
    raw = list(sys.argv[1:] if argv is None else argv)
    global_cfg = _load_json_if_exists(get_global_config_path())
    global_argv = config_to_argv(
        parser,
        global_cfg,
        reserved_keys={config_dest},
        ignore_unknown=True,
    )

    cfg_path = ""
    opt_name = f"--{config_dest.replace('_', '-')}"
    for idx, tok in enumerate(raw):
        if tok == opt_name and idx + 1 < len(raw):
            cfg_path = raw[idx + 1]
            break
        if tok.startswith(opt_name + "="):
            cfg_path = tok.split("=", 1)[1]
            break
    if not cfg_path:
        return parser.parse_args(global_argv + raw)

    cfg = _load_json_config(cfg_path)
    cfg_argv = config_to_argv(parser, cfg, reserved_keys={config_dest})
    merged = global_argv + raw + cfg_argv
    return parser.parse_args(merged)
