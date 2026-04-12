import argparse
import sys

from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="vicon_ws CLI wrapper (legacy/modular engines)."
    )
    parser.add_argument(
        "--engine",
        choices=["modular", "legacy"],
        default="modular",
        help="Execution engine. modular=src core modules, legacy=pipeline.py",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print legacy pipeline command without executing.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
