#!/usr/bin/env python3
"""Compatibility shim for the legacy pipeline entrypoint.

This module keeps the historical ``python pipeline.py`` interface while
forwarding execution to the modular implementation in ``src/vicon_ws``.
Core algorithm code lives only in ``vicon_ws.core``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from vicon_ws.core.calibration import (
    build_translation_system,
    solve_extrinsic_rotation,
    solve_extrinsic_translation,
    solve_world_alignment,
)
from vicon_ws.core.evaluation import (
    build_rpe_pairs,
    compute_ape_evo_style,
    compute_rpe_evo_style,
    print_metric_block,
    rpe_pairs_by_angle,
    rpe_pairs_by_index,
    rpe_pairs_by_path,
    summarize_abs_errors,
)
from vicon_ws.core.io_utils import (
    _analyze_metric_value,
    _as_float,
    find_col,
    load_bag_trajectory,
    load_csv_numeric_columns,
    load_estimation_csv,
    load_estimation_kitti,
    load_estimation_trajectory,
    load_estimation_tum,
    load_reference_trajectory,
    load_vicon_csv,
    make_output_dir,
    save_metrics,
    to_builtin,
    write_metrics_zh_report,
)
from vicon_ws.core.math_utils import (
    compute_error_statistics,
    normalize_quat_array,
    normalize_time_to_seconds,
    poses_se3_from_traj,
    relative_se3,
    rmse,
)
from vicon_ws.core.time_alignment import (
    compute_psr,
    get_angular_velocity_norm,
    interpolate_quat_linear,
    interpolate_quat_slerp,
)


__all__ = [
    "_analyze_metric_value",
    "_as_float",
    "build_rpe_pairs",
    "build_translation_system",
    "compute_ape_evo_style",
    "compute_error_statistics",
    "compute_psr",
    "compute_rpe_evo_style",
    "find_col",
    "get_angular_velocity_norm",
    "load_bag_trajectory",
    "interpolate_quat_linear",
    "interpolate_quat_slerp",
    "load_csv_numeric_columns",
    "load_estimation_csv",
    "load_estimation_kitti",
    "load_estimation_trajectory",
    "load_estimation_tum",
    "load_reference_trajectory",
    "load_vicon_csv",
    "main",
    "make_output_dir",
    "normalize_quat_array",
    "normalize_time_to_seconds",
    "parse_args",
    "poses_se3_from_traj",
    "print_metric_block",
    "relative_se3",
    "rmse",
    "rpe_pairs_by_angle",
    "rpe_pairs_by_index",
    "rpe_pairs_by_path",
    "run_pipeline",
    "save_metrics",
    "solve_extrinsic_rotation",
    "solve_extrinsic_translation",
    "solve_world_alignment",
    "summarize_abs_errors",
    "to_builtin",
    "write_metrics_zh_report",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3-step trajectory alignment and extrinsic calibration pipeline"
    )
    parser.add_argument("--gt-csv", default="gt.csv", help="Path to GT trajectory file/bag")
    parser.add_argument(
        "--gt-format",
        choices=["auto", "csv", "euroc", "tum", "kitti", "bag", "bag2", "mcap"],
        default="csv",
        help="Ground-truth trajectory format",
    )
    parser.add_argument(
        "--gt-topic",
        default="",
        help="GT topic for bag inputs (e.g. /vicon/pose)",
    )
    parser.add_argument("--est-path", default=None, help="Path to estimation trajectory")
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
        "--synthetic",
        action="store_true",
        help="Use synthetic offset/injected transforms instead of real estimation",
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
        help="Use all candidate pairs for RPE (evo-style).",
    )
    parser.add_argument(
        "--rpe-pairs-from-reference",
        action="store_true",
        help="Build RPE pairs from reference trajectory instead of estimate.",
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
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _load_modular_runner():
    from vicon_ws.core.pipeline_modular import run_pipeline_modular

    return run_pipeline_modular


def run_pipeline(args: argparse.Namespace) -> None:
    runner = _load_modular_runner()
    runner(args=args, script_dir=_REPO_ROOT)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
