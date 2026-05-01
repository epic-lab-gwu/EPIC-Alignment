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
        help="Execution engine (modular only).",
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
    parser.add_argument("--gt-csv", default="", help="Path to GT trajectory file/bag")
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
    parser.add_argument("--est-path", default="", help="Path to estimation trajectory")
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
        help="Resampling step for step-1 correlation (seconds)",
    )
    parser.add_argument(
        "--offset-search-window-s",
        type=float,
        default=0.0,
        help=(
            "Step-1 correlation offset search window in seconds (uses +/- window). "
            "Default 0 means full lag range."
        ),
    )
    parser.add_argument(
        "--offset-min-match-ratio",
        type=float,
        default=0.3,
        help=(
            "Minimum timestamp match ratio required for the selected offset. "
            "If not met, fallback near-zero offset is tried; otherwise the run stops."
        ),
    )
    parser.add_argument(
        "--quality-segment-duration-s",
        type=float,
        default=10.0,
        help="Window duration (seconds) for segment-wise alignment quality checks.",
    )
    parser.add_argument(
        "--quality-segment-overlap-ratio",
        type=float,
        default=0.5,
        help="Overlap ratio for quality segments in [0, 1).",
    )
    parser.add_argument(
        "--quality-min-segment-samples",
        type=int,
        default=80,
        help="Minimum samples per segment for quality metrics.",
    )
    parser.add_argument(
        "--quality-good-rmse-m",
        type=float,
        default=0.5,
        help="Step3 RMSE threshold for good_align.",
    )
    parser.add_argument(
        "--quality-partial-rmse-m",
        type=float,
        default=8.0,
        help="Step3 RMSE threshold for partial_align fallback.",
    )
    parser.add_argument(
        "--quality-good-segment-cv",
        type=float,
        default=0.4,
        help="Segment RMSE coefficient-of-variation threshold for good_align.",
    )
    parser.add_argument(
        "--quality-good-heading-p90-deg",
        type=float,
        default=60.0,
        help="Heading-angle p90 (deg) threshold for good_align.",
    )
    parser.add_argument(
        "--quality-partial-min-improve-pct",
        type=float,
        default=20.0,
        help="Minimum raw->step3 RMSE improvement (%%) for partial_align fallback.",
    )
    parser.add_argument(
        "--rigid-check-max-path-ratio",
        type=float,
        default=3.0,
        help="Max symmetric path-length ratio allowed for rigidly_alignable.",
    )
    parser.add_argument(
        "--rigid-check-max-bbox-ratio",
        type=float,
        default=3.0,
        help="Max symmetric bounding-box diagonal ratio allowed for rigidly_alignable.",
    )
    parser.add_argument(
        "--rigid-check-max-global-local-ratio",
        type=float,
        default=6.0,
        help="Max median global/local segment RMSE ratio allowed for rigidly_alignable.",
    )
    parser.add_argument(
        "--rigid-check-max-sim3-gain-ratio",
        type=float,
        default=0.3,
        help="Max relative RMSE gain of Sim3 over SE3 allowed for rigidly_alignable.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic offset/injected transforms",
    )
    parser.add_argument(
        "--quat-interp",
        choices=["linear", "slerp"],
        default="linear",
        help="Quaternion interpolation method after time alignment",
    )
    parser.add_argument(
        "--rpe-delta",
        type=float,
        default=1.0,
        help="RPE delta (same meaning as evo: frame/path/angle increment).",
    )
    parser.add_argument(
        "--rpe-delta-unit",
        choices=["f", "m", "d", "r"],
        default="f",
        help="RPE delta unit: f=frames, m=meters, d=degrees, r=radians.",
    )
    parser.add_argument(
        "--rpe-delta-tol",
        type=float,
        default=0.1,
        help="Relative RPE delta tolerance (used in all-pairs mode for m/d/r).",
    )
    parser.add_argument(
        "--rpe-all-pairs",
        action="store_true",
        help="Use all candidate pairs for RPE.",
    )
    parser.add_argument(
        "--rpe-pairs-from-reference",
        action="store_true",
        help="Build RPE pairs from reference trajectory instead of estimate.",
    )
    parser.add_argument(
        "--t-max-diff",
        type=float,
        default=0.02,
        help="Maximum timestamp difference for trajectory association.",
    )
    parser.add_argument(
        "--t-offset",
        type=float,
        default=0.0,
        help="Constant timestamp offset applied to estimation timestamps before sync.",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="Only keep trajectory samples with t >= t_start (seconds from trajectory start).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Only keep trajectory samples with t <= t_end (seconds from trajectory start).",
    )
    parser.add_argument(
        "--eval-align",
        choices=["none", "se3", "sim3", "scale", "origin"],
        default="none",
        help="Apply optional alignment before APE/RPE evaluation.",
    )
    parser.add_argument(
        "--eval-n-to-align",
        type=int,
        default=-1,
        help="Number of leading poses used for eval alignment (-1 means all).",
    )
    parser.add_argument(
        "--eval-project-to-plane",
        choices=["none", "xy", "xz", "yz"],
        default="none",
        help="Project trajectories to a plane before metric evaluation.",
    )
    parser.add_argument(
        "--ape-pose-relation",
        choices=["all", "full", "trans_part", "rot_part", "angle_deg", "angle_rad", "point_distance"],
        default="trans_part",
        help="APE relation used for summaries/plots.",
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
        help="RPE relation used for summaries/plots.",
    )
    parser.add_argument(
        "--plot-x-dimension",
        dest="plot_x_dimension",
        choices=["index", "seconds", "distances"],
        default="seconds",
        help="X-axis dimension for metric plots.",
    )
    parser.add_argument(
        "--plot-ape-relation",
        dest="plot_ape_relation",
        choices=["full_transformation", "translation_part", "rotation_part", "rotation_angle_rad", "rotation_angle_deg", "point_distance"],
        default="translation_part",
        help="APE relation to visualize in metric plots.",
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
        help="RPE relation to visualize in metric plots.",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="Generate metric plots from the current run.",
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
        "--rerun",
        action="store_true",
        help="Enable optional Rerun trajectory visualization/logging.",
    )
    parser.add_argument(
        "--rerun-no-spawn",
        action="store_true",
        help="Do not auto-spawn the Rerun viewer window.",
    )
    parser.add_argument(
        "--rerun-stride",
        type=int,
        default=20,
        help="Downsample stride for static trajectory lines sent to Rerun.",
    )
    parser.add_argument(
        "--rerun-motion-stride",
        type=int,
        default=5,
        help="Downsample stride for time replay points in Rerun.",
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
