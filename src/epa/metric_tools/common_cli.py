from __future__ import annotations

import argparse


def default_or_suppressed(value, *, suppress_defaults: bool):
    return argparse.SUPPRESS if suppress_defaults else value


def add_usability_args(p: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    dflt = lambda value: default_or_suppressed(value, suppress_defaults=suppress_defaults)
    usability = p.add_argument_group("usability options")
    usability.add_argument(
        "--no_warnings", action="store_true", default=dflt(False), help="reserved for compatibility"
    )
    usability.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=dflt(False),
        help="reserved for compatibility",
    )
    usability.add_argument(
        "--silent", action="store_true", default=dflt(False), help="reserved for compatibility"
    )
    usability.add_argument(
        "--debug", action="store_true", default=dflt(False), help="reserved for compatibility"
    )
    usability.add_argument("--logfile", default=dflt(None), help="reserved for compatibility")


def add_plot_runtime_args(p: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    dflt = lambda value: default_or_suppressed(value, suppress_defaults=suppress_defaults)
    p.add_argument(
        "--plot_interactive",
        "--plot-interactive",
        action="store_true",
        default=dflt(False),
        help="Show interactive matplotlib window after generating plots.",
    )
    p.add_argument(
        "--plot_backend",
        "--plot-backend",
        default=dflt(""),
        help="Optional matplotlib backend override (e.g. qtagg, tkagg).",
    )
    p.add_argument(
        "--serialize_plot",
        "--serialize-plot",
        default=dflt(""),
        help="path to save serialized plot bundle JSON for later re-rendering",
    )
    p.add_argument(
        "--rerun", action="store_true", default=dflt(False), help="Log visualization data to rerun."
    )
    p.add_argument(
        "--rerun_rec_id",
        "--rerun-rec-id",
        default=dflt(None),
        help="Use a specific recording ID for rerun.",
    )
