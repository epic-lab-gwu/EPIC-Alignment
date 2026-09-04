from __future__ import annotations

import argparse
import sys

from .commands import (
    run_error_comparison,
    run_error_dataset,
    run_error_singlerun,
    run_format_converter,
    run_plot_trajectories,
)
from .constants import (
    _DEFAULT_ASSOC_MAX_DIFF,
    _DEFAULT_EPA_DOWNSAMPLE_HZ,
    _DEFAULT_EPA_DT_RESAMPLE,
    _DEFAULT_EPA_OFFSET_MIN_MATCH_RATIO,
    _DEFAULT_EPA_QUAT_INTERP,
)


def _build_format_converter_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible format_converter in EPA.")
    p.add_argument("path", help="CSV file or folder")
    return p


def _add_epa_advanced_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--epa-dt-resample",
        type=float,
        default=_DEFAULT_EPA_DT_RESAMPLE,
        help="EPA Step1 resampling interval used by SE3 compatibility mode.",
    )
    p.add_argument(
        "--epa-offset-min-match-ratio",
        type=float,
        default=_DEFAULT_EPA_OFFSET_MIN_MATCH_RATIO,
        help="Minimum overlap-aware timestamp match ratio required by EPA Step1.",
    )
    p.add_argument(
        "--epa-downsample-hz",
        type=float,
        default=_DEFAULT_EPA_DOWNSAMPLE_HZ,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--epa-quat-interp",
        choices=["linear", "slerp"],
        default=_DEFAULT_EPA_QUAT_INTERP,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--epa-no-fallback",
        action="store_true",
        help="Disable EPA resampling fallbacks and fail when strict timestamp association is insufficient.",
    )
    p.add_argument(
        "--epa-verbose-fallback",
        action="store_true",
        help="Compatibility flag retained for older scripts.",
    )
    p.add_argument(
        "--epa-success-threshold-m",
        type=float,
        default=10.0,
        help="APE translation threshold used for valid-segment success rate and valid-only metrics.",
    )
    p.add_argument(
        "--epa-success-threshold-mode",
        choices=["fixed", "adaptive_knee"],
        default="adaptive_knee",
        help="How to choose the valid-segment threshold per case.",
    )
    p.add_argument(
        "--epa-success-threshold-min-m",
        type=float,
        default=5.0,
        help="Minimum threshold used by adaptive_knee success-threshold mode.",
    )
    p.add_argument(
        "--epa-success-threshold-max-m",
        type=float,
        default=30.0,
        help="Maximum threshold used by adaptive_knee success-threshold mode.",
    )
    p.add_argument(
        "--epa-success-threshold-trim-percentile",
        type=float,
        default=95.0,
        help="Upper percentile retained before adaptive knee threshold estimation.",
    )
    p.add_argument(
        "--epa-success-global-gate-mode",
        choices=["fixed", "scale_aware"],
        default="fixed",
        help="How to choose the global accept gate per case.",
    )
    p.add_argument(
        "--epa-success-global-gate-m",
        type=float,
        default=30.0,
        help="Global accept gate in meters; cases with low-percentile APE above this are globally failed.",
    )
    p.add_argument(
        "--epa-success-global-gate-path-ratio",
        type=float,
        default=0.05,
        help="GT path-length ratio used by scale-aware global gate mode.",
    )
    p.add_argument(
        "--epa-success-global-gate-min-m",
        type=float,
        default=2.0,
        help="Minimum gate used by scale-aware global gate mode.",
    )
    p.add_argument(
        "--epa-success-global-gate-max-m",
        type=float,
        default=100.0,
        help="Maximum gate used by scale-aware global gate mode.",
    )
    p.add_argument(
        "--epa-success-global-gate-percentile",
        type=float,
        default=5.0,
        help="APE percentile used by the global accept gate.",
    )
    p.add_argument(
        "--epa-success-drift-threshold-mode",
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="How to resolve local drift thresholds for EPA success-rate metrics.",
    )
    p.add_argument(
        "--epa-success-drift-rpe-1s-m",
        type=float,
        default=2.0,
        help="1s RPE translation threshold used to mark local drift segments.",
    )
    p.add_argument(
        "--epa-success-drift-ape-slope-mps",
        type=float,
        default=1.0,
        help="Positive APE growth-rate threshold used to mark local drift segments.",
    )
    p.add_argument(
        "--epa-success-drift-ape-jump-m",
        type=float,
        default=5.0,
        help="Minimum APE jump paired with APE growth-rate for local drift detection.",
    )


def _add_batch_failure_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--fail-on-skipped",
        action="store_true",
        help="Return a non-zero exit code if any run is skipped or no valid runs are found.",
    )


def _build_error_singlerun_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible error_singlerun in EPA.")
    p.add_argument(
        "align_mode",
        help=(
            "Public modes: se3|se3-original|posyaw|sim3. "
            "Compatibility aliases such as epa_step3/epa_se3/epa_se3_eval and old Sim3 solver names are accepted."
        ),
    )
    p.add_argument("file_gt", help="groundtruth trajectory")
    p.add_argument("file_est", help="estimated trajectory")
    p.add_argument(
        "--max-diff",
        type=float,
        default=_DEFAULT_ASSOC_MAX_DIFF,
        help="timestamp association threshold",
    )
    p.add_argument("--plot", action="store_true", help="reserved in compatibility mode")
    _add_epa_advanced_args(p)
    return p


def _build_error_dataset_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible error_dataset in EPA.")
    p.add_argument(
        "align_mode",
        help=(
            "Public modes: se3|se3-original|posyaw|sim3. "
            "Compatibility aliases such as epa_step3/epa_se3/epa_se3_eval and old Sim3 solver names are accepted."
        ),
    )
    p.add_argument("file_gt", help="groundtruth trajectory")
    p.add_argument("folder_algorithms", help="algorithm root folder")
    p.add_argument(
        "--max-diff",
        type=float,
        default=_DEFAULT_ASSOC_MAX_DIFF,
        help="timestamp association threshold",
    )
    p.add_argument("--plot", action="store_true", help="reserved in compatibility mode")
    _add_epa_advanced_args(p)
    _add_batch_failure_args(p)
    return p


def _build_error_comparison_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible error_comparison in EPA.")
    p.add_argument(
        "align_mode",
        help=(
            "Public modes: se3|se3-original|posyaw|sim3. "
            "Compatibility aliases such as epa_step3/epa_se3/epa_se3_eval and old Sim3 solver names are accepted."
        ),
    )
    p.add_argument("folder_groundtruth", help="groundtruth root folder")
    p.add_argument("folder_algorithms", help="algorithm root folder")
    p.add_argument(
        "--max-diff",
        type=float,
        default=_DEFAULT_ASSOC_MAX_DIFF,
        help="timestamp association threshold",
    )
    _add_epa_advanced_args(p)
    _add_batch_failure_args(p)
    return p


def _build_plot_trajectories_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible plot_trajectories in EPA.")
    p.add_argument(
        "align_mode",
        help=(
            "Public modes: se3|se3-original|posyaw|sim3. "
            "Compatibility aliases such as epa_step3/epa_se3/epa_se3_eval and old Sim3 solver names are accepted."
        ),
    )
    p.add_argument("file_gt", help="groundtruth trajectory")
    p.add_argument("est_files", nargs="+", help="estimated trajectories")
    p.add_argument(
        "--max-diff",
        type=float,
        default=_DEFAULT_ASSOC_MAX_DIFF,
        help="timestamp association threshold",
    )
    return p


def main_format_converter(argv: list[str] | None = None) -> int:
    return int(run_format_converter(_build_format_converter_parser().parse_args(argv)))


def main_error_singlerun(argv: list[str] | None = None) -> int:
    return int(run_error_singlerun(_build_error_singlerun_parser().parse_args(argv)))


def main_error_dataset(argv: list[str] | None = None) -> int:
    return int(run_error_dataset(_build_error_dataset_parser().parse_args(argv)))


def main_error_comparison(argv: list[str] | None = None) -> int:
    return int(run_error_comparison(_build_error_comparison_parser().parse_args(argv)))


def main_plot_trajectories(argv: list[str] | None = None) -> int:
    return int(run_plot_trajectories(_build_plot_trajectories_parser().parse_args(argv)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenVINS ov_eval compatibility layer for EPA.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("format_converter", help="Convert EuRoC CSV to ov_eval txt format")
    sub.add_parser("error_singlerun", help="Single-run ATE/RPE summary")
    sub.add_parser("error_dataset", help="Single-dataset multi-algorithm summary")
    sub.add_parser("error_comparison", help="Multi-dataset multi-algorithm summary")
    sub.add_parser("plot_trajectories", help="Trajectory overlay plotting")

    return p


_COMMAND_PARSERS = {
    "format_converter": _build_format_converter_parser,
    "error_singlerun": _build_error_singlerun_parser,
    "error_dataset": _build_error_dataset_parser,
    "error_comparison": _build_error_comparison_parser,
    "plot_trajectories": _build_plot_trajectories_parser,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _COMMAND_PARSERS:
        command = argv[0]
        rest = argv[1:]
    else:
        parser = build_parser()
        args = parser.parse_args(argv)
        command = str(args.command)
        rest = []

    if command == "format_converter":
        return int(main_format_converter(rest))
    if command == "error_singlerun":
        return int(main_error_singlerun(rest))
    if command == "error_dataset":
        return int(main_error_dataset(rest))
    if command == "error_comparison":
        return int(main_error_comparison(rest))
    if command == "plot_trajectories":
        return int(main_plot_trajectories(rest))

    raise ValueError(f"Unsupported command: {command}")
