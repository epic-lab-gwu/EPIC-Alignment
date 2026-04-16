from __future__ import annotations

import argparse
from pathlib import Path


def build_user_namespace() -> dict[str, object]:
    import matplotlib.pyplot as plt
    import numpy as np

    from epa import ape_tool, rpe_tool, traj_tool
    from epa.benchmark import metrics_res, plot_summary, res
    from epa.core import calibration, evaluation, io_utils, math_utils, pipeline_modular, time_alignment

    return {
        "np": np,
        "plt": plt,
        "Path": Path,
        "calibration": calibration,
        "evaluation": evaluation,
        "io_utils": io_utils,
        "math_utils": math_utils,
        "time_alignment": time_alignment,
        "pipeline_modular": pipeline_modular,
        "ape_tool": ape_tool,
        "rpe_tool": rpe_tool,
        "traj_tool": traj_tool,
        "res_tool": res,
        "metrics_res_tool": metrics_res,
        "plot_summary_tool": plot_summary,
    }


def launch_ipython(ipython_args: list[str], user_ns: dict[str, object], banner: str) -> int:
    try:
        from IPython import start_ipython
    except Exception as exc:
        raise ImportError(
            "IPython is required for epa_ipython. Install with: pip install ipython"
        ) from exc

    print(banner)
    start_ipython(argv=ipython_args, user_ns=user_ns)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch IPython with pre-loaded epa modules."
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List pre-loaded symbols and exit.",
    )
    return p


def run(args: argparse.Namespace, ipython_args: list[str]) -> int:
    ns = build_user_namespace()
    if bool(getattr(args, "list", False)):
        for name in sorted(ns.keys()):
            print(name)
        return 0

    banner = (
        "Welcome to epa IPython\n"
        "Pre-loaded: np, plt, Path, core modules, and tool modules."
    )
    return launch_ipython(ipython_args, ns, banner)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, other = parser.parse_known_args(argv)
    return run(args, ipython_args=other)


if __name__ == "__main__":
    raise SystemExit(main())

