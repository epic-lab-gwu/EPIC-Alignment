from __future__ import annotations

import os
import sys
from typing import Sequence


def _arg_value(argv: Sequence[str], names: Sequence[str]) -> str:
    for i, tok in enumerate(argv):
        for name in names:
            if tok == name and i + 1 < len(argv):
                return str(argv[i + 1]).strip()
            prefix = f"{name}="
            if tok.startswith(prefix):
                return str(tok[len(prefix):]).strip()
    return ""


def bootstrap_matplotlib_backend(matplotlib_mod, argv: Sequence[str] | None = None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    env_backend = str(os.getenv("EPA_MPL_BACKEND", "") or "").strip()
    cli_backend = _arg_value(args, ["--plot-backend", "--plot_backend"])
    requested = env_backend or cli_backend
    explicit_interactive = ("--plot-interactive" in args) or ("--plot_interactive" in args)
    auto_plot_interactive = (
        ("--plot" in args or "-p" in args)
        and bool(sys.stdin.isatty() and sys.stdout.isatty())
    )
    interactive = bool(explicit_interactive or auto_plot_interactive)

    if interactive:
        candidates: list[str] = []
        if requested:
            candidates.append(requested)
        for name in ("qtagg", "qt5agg", "tkagg"):
            if name not in candidates:
                candidates.append(name)
        for name in candidates:
            try:
                matplotlib_mod.use(name, force=True)
                return name
            except Exception:
                continue
        matplotlib_mod.use("Agg", force=True)
        return "Agg"

    if requested:
        try:
            matplotlib_mod.use(requested, force=True)
            return requested
        except Exception:
            pass
    matplotlib_mod.use("Agg", force=True)
    return "Agg"
