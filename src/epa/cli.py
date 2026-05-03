import argparse
import sys

from .config_cli import parse_args_with_config
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="epa CLI wrapper (modular engine)."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="INPUT",
        help="Optional positional inputs: <gt_file> <est_file>.",
    )
    parser.add_argument(
        "--engine",
        choices=["modular"],
        default="modular",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to JSON config file. If set, config values override CLI flags.",
    )
    parser.add_argument(
        "--gt",
        dest="gt_alias",
        default="",
        help="Ground-truth trajectory path. Alias for --gt-csv.",
    )
    parser.add_argument("--gt-csv", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--gt-format",
        choices=["auto", "csv", "euroc", "tum", "kitti", "bag", "bag2", "mcap"],
        default="auto",
        help="Ground-truth trajectory format",
    )
    parser.add_argument(
        "--gt-topic",
        default="",
        help="GT topic for bag inputs (e.g. /vicon/pose)",
    )
    parser.add_argument(
        "--est",
        dest="est_alias",
        default="",
        help="Estimated trajectory path. Alias for --est-path.",
    )
    parser.add_argument("--est-path", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--est-format",
        choices=["auto", "csv", "euroc", "tum", "kitti", "bag", "bag2", "mcap"],
        default="auto",
        help="Estimation trajectory format",
    )
    parser.add_argument(
        "--est-topic",
        default="",
        help="Estimation topic for bag inputs (e.g. /odom)",
    )
    parser.add_argument(
        "--dt-resample",
        type=float,
        default=0.001,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--offset-search-window-s",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--offset-min-match-ratio",
        type=float,
        default=0.3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-segment-duration-s",
        type=float,
        default=10.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-segment-overlap-ratio",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-min-segment-samples",
        type=int,
        default=80,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-good-rmse-m",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-partial-rmse-m",
        type=float,
        default=8.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-good-segment-cv",
        type=float,
        default=0.4,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-good-heading-p90-deg",
        type=float,
        default=60.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quality-partial-min-improve-pct",
        type=float,
        default=20.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rigid-check-max-path-ratio",
        type=float,
        default=3.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rigid-check-max-bbox-ratio",
        type=float,
        default=3.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rigid-check-max-global-local-ratio",
        type=float,
        default=6.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rigid-check-max-sim3-gain-ratio",
        type=float,
        default=0.3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quat-interp",
        choices=["linear", "slerp"],
        default="linear",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rpe-delta",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rpe-delta-unit",
        choices=["f", "m", "d", "r"],
        default="f",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rpe-delta-tol",
        type=float,
        default=0.1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rpe-all-pairs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rpe-pairs-from-reference",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--t-max-diff",
        type=float,
        default=0.02,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--t-offset",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval-align",
        choices=["none", "se3", "sim3", "scale", "origin"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval-n-to-align",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval-project-to-plane",
        choices=["none", "xy", "xz", "yz"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ape-pose-relation",
        choices=["all", "full", "trans_part", "rot_part", "angle_deg", "angle_rad", "point_distance"],
        default="trans_part",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rpe-pose-relation",
        choices=[
            "all",
            "full",
            "trans_part",
            "rot_part",
            "angle_deg",
            "angle_rad",
            "point_distance",
            "point_distance_error_ratio",
        ],
        default="trans_part",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plot-x-dimension",
        dest="plot_x_dimension",
        choices=["index", "seconds", "distances"],
        default="seconds",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plot-ape-relation",
        dest="plot_ape_relation",
        choices=["full_transformation", "translation_part", "rotation_part", "rotation_angle_rad", "rotation_angle_deg", "point_distance"],
        default="translation_part",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plot-rpe-relation",
        dest="plot_rpe_relation",
        choices=[
            "full_transformation",
            "translation_part",
            "rotation_part",
            "rotation_angle_rad",
            "rotation_angle_deg",
            "point_distance",
            "point_distance_error_ratio",
        ],
        default="translation_part",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable metric plot generation.",
    )
    parser.set_defaults(plot=True)
    parser.add_argument(
        "--save-results",
        default="",
        help="Optional path to save a bundled result zip (e.g. outputs/results/run1.zip).",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-full-metrics",
        action="store_true",
        help="Keep full per-sample APE/RPE arrays in metrics.json. Default writes compact metrics.",
    )
    parser.add_argument(
        "--downsample-hz",
        type=float,
        default=100.0,
        help="Cap Step-2/3 and metric evaluation to this sampling rate. Step-1 time alignment still uses full input.",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Disable the default post-time-alignment downsampling.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Enable optional Rerun trajectory visualization/logging.",
    )
    parser.add_argument(
        "--rerun-no-spawn",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rerun-stride",
        type=int,
        default=20,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rerun-motion-stride",
        type=int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print modular command preview without executing.",
    )
    return parser


def _normalize_inputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    inputs = list(getattr(args, "inputs", []) or [])
    if len(inputs) > 2:
        parser.error("Expected at most two positional inputs: <gt_file> <est_file>.")

    gt_flags = [value for value in [getattr(args, "gt_alias", ""), getattr(args, "gt_csv", "")] if value]
    est_flags = [value for value in [getattr(args, "est_alias", ""), getattr(args, "est_path", "")] if value]
    raw_argv = list(getattr(args, "_raw_argv", []) or [])
    explicit_gt = any(
        tok == "--gt" or tok.startswith("--gt=") or tok == "--gt-csv" or tok.startswith("--gt-csv=")
        for tok in raw_argv
    )
    explicit_est = any(
        tok == "--est" or tok.startswith("--est=") or tok == "--est-path" or tok.startswith("--est-path=")
        for tok in raw_argv
    )
    if len(gt_flags) > 1:
        parser.error("Use only one of --gt or --gt-csv.")
    if len(est_flags) > 1:
        parser.error("Use only one of --est or --est-path.")
    if inputs and (explicit_gt or explicit_est):
        parser.error("Use either positional inputs or --gt/--est flags, not both.")

    if inputs:
        if len(inputs) != 2:
            parser.error("Positional mode requires exactly two inputs: <gt_file> <est_file>.")
        args.gt_csv, args.est_path = inputs
    else:
        args.gt_csv = gt_flags[0] if gt_flags else ""
        args.est_path = est_flags[0] if est_flags else ""

    if not bool(getattr(args, "synthetic", False)):
        if not args.gt_csv or not args.est_path:
            parser.error("Provide <gt_file> <est_file>, or use --gt/--est.")

    return args


def main(argv=None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args_with_config(parser, argv=argv, config_dest="config", tool_name="epa")
    args._raw_argv = raw_argv
    args = _normalize_inputs(parser, args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
