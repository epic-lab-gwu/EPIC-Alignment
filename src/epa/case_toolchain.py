from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from epa.alignment.modes import resolve_metric_eval_align_mode


def _default_out_root(cwd: Path) -> Path:
    root = cwd / "outputs" / "epa_all"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = root / stamp
    idx = 1
    while out.exists():
        out = root / f"{stamp}_{idx:02d}"
        idx += 1
    return out


def _run(cmd: list[str], cwd: Path) -> float:
    print("+ " + " ".join(str(x) for x in cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    elapsed = float(time.perf_counter() - t0)
    print(f"[timing] {elapsed:.3f}s")
    return elapsed


def _metric_eval_align(align_mode: str) -> str:
    return resolve_metric_eval_align_mode(
        align_mode,
        default="se3",
        collapse_sim3_aliases=True,
        legacy_origin=False,
        step3_as_se3=True,
        strict=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the EPA single-case full toolchain in one command."
    )
    p.add_argument("--gt", default="", help="Ground-truth trajectory file.")
    p.add_argument("--est", default="", help="Estimated trajectory file.")
    p.add_argument(
        "--case-dir",
        default="",
        help="Optional OpenVINS-style case directory containing stamped_groundtruth.txt and stamped_traj_estimate.txt.",
    )
    p.add_argument(
        "--format",
        choices=["tum", "kitti", "euroc"],
        default="tum",
        help="Trajectory format for --gt/--est inputs.",
    )
    p.add_argument(
        "--align-mode",
        default="se3",
        metavar="{se3,se3r,posyaw,sim3}",
        help=(
            "Public alignment mode: se3 (position-only Umeyama), se3r "
            "(rotation-first), posyaw, or sim3. Compatibility aliases are accepted."
        ),
    )
    p.add_argument(
        "--run-openvins-compat",
        action="store_true",
        help="Also run the legacy OpenVINS-style compatibility wrapper as an extra baseline.",
    )
    p.add_argument(
        "--t-max-diff",
        type=float,
        default=0.02,
        help="Timestamp matching threshold.",
    )
    p.add_argument(
        "--out-root",
        default="",
        help="Output root for all generated artifacts.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plots for all supported tools.",
    )
    return p


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, str, Path | None]:
    case_dir = None
    gt = str(args.gt or "").strip()
    est = str(args.est or "").strip()
    fmt = str(args.format)

    if str(args.case_dir or "").strip():
        case_dir = Path(args.case_dir).expanduser().resolve()
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")
        if not gt:
            gt = str(case_dir / "stamped_groundtruth.txt")
        if not est:
            est = str(case_dir / "stamped_traj_estimate.txt")
        fmt = "tum"

    if not gt or not est:
        raise ValueError("Provide --gt and --est, or provide --case-dir.")

    gt_path = Path(gt).expanduser().resolve()
    est_path = Path(est).expanduser().resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")
    if not est_path.exists():
        raise FileNotFoundError(f"EST file not found: {est_path}")
    return gt_path, est_path, fmt, case_dir


def run(args: argparse.Namespace) -> int:
    cwd = Path.cwd().resolve()
    gt_path, est_path, fmt, case_dir = _resolve_inputs(args)
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if str(args.out_root or "").strip()
        else _default_out_root(cwd)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    main_plot_flag = "--no-plot" if bool(args.no_plot) else "--plot"
    metric_plot_flag: list[str] = [] if bool(args.no_plot) else ["--plot"]
    traj_plot_flag: list[str] = [] if bool(args.no_plot) else ["--plot"]
    openvins_plot_flag: list[str] = ["--no-plot"] if bool(args.no_plot) else []

    print("== EPA ALL ==")
    print(f"GT:         {gt_path}")
    print(f"EST:        {est_path}")
    print(f"FORMAT:     {fmt}")
    print(f"OUT_ROOT:   {out_root}")
    print(f"CASE_DIR:   {case_dir if case_dir is not None else '(none)'}")
    print(
        "OPENVINS_COMPAT: "
        + ("enabled" if bool(args.run_openvins_compat) else "skipped")
    )
    print()
    timings: dict[str, float] = {}

    main_workspace = out_root / "main_workspace"
    main_workspace.mkdir(parents=True, exist_ok=True)
    timings["main_epa_cli_s"] = _run(
        [
            sys.executable,
            "-m",
            "epa.cli",
            str(gt_path),
            str(est_path),
            "--gt-format",
            fmt,
            "--est-format",
            fmt,
            "--t-max-diff",
            str(float(args.t_max_diff)),
            main_plot_flag,
        ],
        cwd=main_workspace,
    )

    metric_align = _metric_eval_align(str(args.align_mode))
    ape_out = out_root / "ape"
    ape_out.mkdir(parents=True, exist_ok=True)
    timings["ape_tool_s"] = _run(
        [
            sys.executable,
            "-m",
            "epa.ape_tool",
            fmt,
            str(gt_path),
            str(est_path),
            "--eval-align",
            metric_align,
            "--t_max_diff",
            str(float(args.t_max_diff)),
            *metric_plot_flag,
            "--out_dir",
            str(ape_out),
        ],
        cwd=cwd,
    )

    rpe_out = out_root / "rpe"
    rpe_out.mkdir(parents=True, exist_ok=True)
    timings["rpe_tool_s"] = _run(
        [
            sys.executable,
            "-m",
            "epa.rpe_tool",
            fmt,
            str(gt_path),
            str(est_path),
            "--eval-align",
            metric_align,
            "--delta",
            "1",
            "--delta_unit",
            "f",
            "--t_max_diff",
            str(float(args.t_max_diff)),
            *metric_plot_flag,
            "--out_dir",
            str(rpe_out),
        ],
        cwd=cwd,
    )

    traj_out = out_root / "traj"
    traj_out.mkdir(parents=True, exist_ok=True)
    traj_cmd = [
        sys.executable,
        "-m",
        "epa.traj_tool",
        "--format",
        fmt,
        "--eval-align",
        metric_align,
        "--ref",
        "1",
        *traj_plot_flag,
        "--out-dir",
        str(traj_out),
        str(gt_path),
        str(est_path),
    ]
    if fmt != "kitti":
        traj_cmd[5:5] = [
            "--sync",
            "--sync-max-diff",
            str(float(args.t_max_diff)),
        ]
    timings["traj_tool_s"] = _run(traj_cmd, cwd=cwd)

    if case_dir is not None and bool(args.run_openvins_compat):
        openvins_out = out_root / "openvins"
        openvins_out.mkdir(parents=True, exist_ok=True)
        timings["openvins_runner_s"] = _run(
            [
                sys.executable,
                "-m",
                "epa.openvins_runner",
                str(case_dir),
                "--align-mode",
                str(args.align_mode),
                *openvins_plot_flag,
                "--keep-output",
                "--save-root",
                str(openvins_out),
            ],
            cwd=cwd,
        )
    elif case_dir is not None:
        print(
            "[info] OpenVINS compatibility runner skipped for EPA-only run. "
            "Use --run-openvins-compat to include it."
        )

    print()
    timings["total_recorded_s"] = float(sum(timings.values()))
    (out_root / "timing_summary.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"Timing summary: {out_root / 'timing_summary.json'}")
    print(f"Done. Outputs saved under: {out_root}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
