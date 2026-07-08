from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .cases import BenchmarkCase, _write_tum, discover_cases, load_pose_table
from .latex_summary import write_latex_tables
from .summary import _write_public_summary_csv, _write_summary_html, _write_summary_md


def _bool_to_status(ok: bool) -> str:
    return "ok" if ok else "failed"


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return float("nan")
    return float("nan")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _parse_epa_run_dir(stdout_text: str, stderr_text: str) -> Path | None:
    combined = f"{stdout_text}\n{stderr_text}"
    match = re.search(r"Saving outputs to:\s*(\S+)", combined)
    if not match:
        return None
    return Path(match.group(1))


def _tail(text: str, num_lines: int = 40) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-num_lines:])


def _run_epa_case(
    case: BenchmarkCase,
    gt_tum_path: Path,
    est_tum_path: Path,
    *,
    repo_root: Path,
    python_bin: Path,
    epa_src: Path,
    dt_resample: float,
    quat_interp: str,
    downsample_hz: float,
    no_downsample: bool,
    mplconfigdir: Path,
    output_root: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    eval_align: str = "",
) -> dict[str, object]:
    env = os.environ.copy()
    py_paths = [str(epa_src)]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mplconfigdir)
    env["XDG_CONFIG_HOME"] = str(mplconfigdir)

    cmd = [
        str(python_bin),
        "-m",
        "epa.cli",
        "--engine",
        "modular",
        "--gt-csv",
        str(gt_tum_path),
        "--gt-format",
        "tum",
        "--est-path",
        str(est_tum_path),
        "--est-format",
        "tum",
        "--dt-resample",
        str(dt_resample),
        "--quat-interp",
        quat_interp,
        "--downsample-hz",
        str(downsample_hz),
        "--output-root",
        str(output_root),
        "--run-label",
        case.case_id,
    ]
    eval_align = str(eval_align or "").strip()
    if eval_align:
        cmd.extend(["--eval-align", eval_align])
    if bool(no_downsample):
        cmd.append("--no-downsample")
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_log_path.write_text(proc.stdout, encoding="utf-8")
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.write_text(proc.stderr, encoding="utf-8")

    result: dict[str, object] = {
        "status": _bool_to_status(proc.returncode == 0),
        "returncode": proc.returncode,
        "stdout_log": str(stdout_log_path),
        "stderr_log": str(stderr_log_path),
        "stderr_tail": _tail(proc.stderr) if proc.returncode != 0 else "",
        "run_dir": "",
        "offset_est_s": float("nan"),
        "ate_rmse_raw_m": float("nan"),
        "ate_rmse_step3_m": float("nan"),
        "improve_raw_to_step3_pct": float("nan"),
        "xcorr_peak": float("nan"),
        "xcorr_psr": float("nan"),
        "omega_improve_pct": float("nan"),
        "evo_t_offset_used_s": float("nan"),
        "evo_matches_equivalent": float("nan"),
        "sr_distance": float("nan"),
        "sr_time": float("nan"),
        "sr_distance_raw": float("nan"),
        "sr_time_raw": float("nan"),
        "sr_reliability_status": "",
        "sr_warning_explanation": "",
        "sim3_may_mask_failure": "",
        "case_status": "",
        "global_gate_failed": "",
        "global_gate_value_m": float("nan"),
        "global_gate_m": float("nan"),
        "valid_distance_m": float("nan"),
        "total_distance_m": float("nan"),
        "step3_alignment_mode": "",
        "step3_inlier_count": float("nan"),
        "step3_rejected_count": float("nan"),
        "step3_rejection_ratio": float("nan"),
        "step3_stable_segment_used": float("nan"),
        "step3_stable_solve_ratio": float("nan"),
        "sim3_scale": float("nan"),
        "sim3_reliable": "",
        "sim3_warning": "",
        "orientation_unstable": "",
        "orientation_ape_rmse_deg": float("nan"),
        "orientation_rpe_rmse_deg": float("nan"),
        "orientation_rpe_time_1s_rmse_deg": float("nan"),
        "rpe_time_1s_trans_rmse_m": float("nan"),
        "rpe_time_1s_rot_rmse_deg": float("nan"),
        "orientation_warning": "",
        "diagnosis_primary": "",
        "diagnosis_summary": "",
        "diagnosis_tags": "",
        "eval_align": eval_align,
        "sim3_stable_anchor_used": "",
        "sim3_fallback_used": "",
        "sim3_fallback_reason": "",
    }

    run_dir = _parse_epa_run_dir(proc.stdout, proc.stderr)
    if run_dir is None:
        return result

    result["run_dir"] = str(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return result
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return result

    traj = payload.get("trajectory", {})
    time_block = payload.get("time_alignment", {})
    step3_selection = payload.get("step3_selection", {})
    valid_step3 = (
        payload.get("pose_metrics", {})
        .get("valid_segment", {})
        .get("step3", {})
        .get("success", {})
    )
    eval_step3 = (
        payload.get("pose_metrics", {})
        .get("eval_alignment", {})
        .get("step3", {})
    )
    rpe_time_1s_step3 = (
        payload.get("pose_metrics", {})
        .get("rpe_time_1s", {})
        .get("step3", {})
    )
    orientation = payload.get("orientation_diagnostics", {})
    case_diagnostics = payload.get("case_diagnostics", {})
    metadata = payload.get("metadata", {})
    metadata_eval_alignment = metadata.get("eval_alignment", {})
    result["offset_est_s"] = _as_float(time_block.get("offset_est_s"))
    result["ate_rmse_raw_m"] = _as_float(traj.get("ate_rmse_raw_m"))
    result["ate_rmse_step3_m"] = _as_float(traj.get("ate_rmse_step3_m"))
    result["improve_raw_to_step3_pct"] = _as_float(traj.get("ate_rmse_improve_raw_to_step3_pct"))
    result["xcorr_peak"] = _as_float(time_block.get("xcorr_peak_normalized"))
    result["xcorr_psr"] = _as_float(time_block.get("xcorr_psr"))
    result["omega_improve_pct"] = _as_float(time_block.get("omega_rmse_improve_pct"))
    result["evo_t_offset_used_s"] = _as_float(time_block.get("evo_t_offset_used_s"))
    result["evo_matches_equivalent"] = _as_float(time_block.get("evo_matches_equivalent"))
    result["sr_distance"] = _as_float(valid_step3.get("success_rate_distance"))
    result["sr_time"] = _as_float(valid_step3.get("success_rate_time"))
    result["sr_distance_raw"] = _as_float(valid_step3.get("raw_success_rate_distance"))
    result["sr_time_raw"] = _as_float(valid_step3.get("raw_success_rate_time"))
    result["sr_reliability_status"] = str(valid_step3.get("sr_reliability_status", ""))
    result["sr_warning_explanation"] = str(valid_step3.get("sr_warning_explanation", ""))
    result["sim3_may_mask_failure"] = str(bool(valid_step3.get("sim3_may_mask_failure", False)))
    result["case_status"] = str(valid_step3.get("case_status", ""))
    result["global_gate_failed"] = _as_bool(valid_step3.get("global_gate_failed"))
    result["global_gate_value_m"] = _as_float(valid_step3.get("global_gate_value_m"))
    result["global_gate_m"] = _as_float(valid_step3.get("global_gate_m"))
    result["valid_distance_m"] = _as_float(valid_step3.get("valid_distance_m"))
    result["total_distance_m"] = _as_float(valid_step3.get("total_distance_m"))
    result["step3_alignment_mode"] = str(step3_selection.get("step3_alignment_mode", ""))
    result["step3_inlier_count"] = _as_float(step3_selection.get("step3_inlier_count"))
    result["step3_rejected_count"] = _as_float(step3_selection.get("step3_rejected_count"))
    result["step3_rejection_ratio"] = _as_float(step3_selection.get("step3_rejection_ratio"))
    result["step3_stable_segment_used"] = _as_float(step3_selection.get("step3_stable_segment_used"))
    result["step3_stable_solve_ratio"] = _as_float(step3_selection.get("step3_stable_solve_ratio"))
    result["sim3_scale"] = _as_float(eval_step3.get("sim3_scale", eval_step3.get("align_scale")))
    if "sim3" in str(eval_step3.get("align_mode", "")).lower():
        result["sim3_reliable"] = str(bool(eval_step3.get("sim3_reliable", True)))
        result["sim3_warning"] = str(eval_step3.get("sim3_warning", ""))
    result["eval_align"] = str(metadata.get("eval_align", eval_align))
    stable_used = eval_step3.get(
        "sim3_stable_anchor_used",
        metadata_eval_alignment.get("sim3_stable_anchor_used", "")
        if isinstance(metadata_eval_alignment, dict)
        else "",
    )
    result["sim3_stable_anchor_used"] = "" if stable_used == "" else str(bool(stable_used))
    fallback_used = eval_step3.get(
        "sim3_robust_fallback_used",
        metadata_eval_alignment.get("sim3_robust_fallback_used", "")
        if isinstance(metadata_eval_alignment, dict)
        else "",
    )
    result["sim3_fallback_used"] = "" if fallback_used == "" else str(bool(fallback_used))
    result["sim3_fallback_reason"] = str(
        eval_step3.get(
            "sim3_robust_fallback_reason",
            metadata_eval_alignment.get("sim3_robust_fallback_reason", "")
            if isinstance(metadata_eval_alignment, dict)
            else "",
        )
    )
    result["orientation_unstable"] = str(
        bool(orientation.get("orientation_unstable", metadata.get("orientation_unstable", False)))
    )
    result["orientation_ape_rmse_deg"] = _as_float(
        orientation.get("orientation_ape_rmse_deg", metadata.get("orientation_ape_rmse_deg"))
    )
    result["orientation_rpe_rmse_deg"] = _as_float(
        orientation.get("orientation_rpe_rmse_deg", metadata.get("orientation_rpe_rmse_deg"))
    )
    result["orientation_rpe_time_1s_rmse_deg"] = _as_float(
        orientation.get("orientation_rpe_time_1s_rmse_deg", metadata.get("orientation_rpe_time_1s_rmse_deg"))
    )
    result["rpe_time_1s_trans_rmse_m"] = _as_float(
        rpe_time_1s_step3.get("translation_part", {}).get("rmse")
    )
    result["rpe_time_1s_rot_rmse_deg"] = _as_float(
        rpe_time_1s_step3.get("rotation_angle_deg", {}).get("rmse")
    )
    result["orientation_warning"] = str(
        orientation.get("orientation_warning", metadata.get("orientation_warning", ""))
    )
    result["diagnosis_primary"] = str(
        case_diagnostics.get("diagnosis_primary", metadata.get("diagnosis_primary", ""))
    )
    result["diagnosis_summary"] = str(
        case_diagnostics.get("diagnosis_summary", metadata.get("diagnosis_summary", ""))
    )
    tags = case_diagnostics.get("diagnosis_tags", metadata.get("diagnosis_tags", ""))
    if isinstance(tags, list):
        tags = ",".join(str(tag) for tag in tags)
    result["diagnosis_tags"] = str(tags)
    return result


def _build_evo_traj(PoseTrajectory3D, data: np.ndarray):
    xyz = data[:, 1:4]
    q_xyzw = data[:, 4:8]
    q_wxyz = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])
    stamps = data[:, 0]
    return PoseTrajectory3D(
        positions_xyz=xyz,
        orientations_quat_wxyz=q_wxyz,
        timestamps=stamps,
    )


def _rmse_from_synced(traj_ref_sync, traj_est_sync) -> float:
    diff = traj_ref_sync.positions_xyz - traj_est_sync.positions_xyz
    sq = np.sum(diff * diff, axis=1)
    return float(np.sqrt(np.mean(sq)))


def _better_offset_choice(
    current: tuple[bool, int, float] | None,
    candidate: tuple[bool, int, float],
) -> bool:
    if current is None:
        return True
    curr_valid, curr_matches, curr_rmse = current
    cand_valid, cand_matches, cand_rmse = candidate
    if cand_valid and not curr_valid:
        return True
    if curr_valid and not cand_valid:
        return False
    if cand_valid and curr_valid:
        if cand_rmse < curr_rmse - 1e-12:
            return True
        if math.isclose(cand_rmse, curr_rmse, rel_tol=0.0, abs_tol=1e-12):
            return cand_matches > curr_matches
        return False
    if cand_matches > curr_matches:
        return True
    if cand_matches == curr_matches and cand_rmse < curr_rmse - 1e-12:
        return True
    return False


def _evaluate_offset(sync_module, traj_ref, traj_est, max_diff: float, offset_s: float) -> tuple[int, float]:
    traj_ref_sync, traj_est_sync = sync_module.associate_trajectories(
        traj_ref,
        traj_est,
        max_diff=max_diff,
        offset_2=offset_s,
        first_name="ref",
        snd_name="est",
    )
    matches = int(traj_ref_sync.num_poses)
    rmse_raw = _rmse_from_synced(traj_ref_sync, traj_est_sync)
    return matches, rmse_raw


def _offset_grid(min_offset: float, max_offset: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    values = np.arange(min_offset, max_offset + 0.5 * step, step)
    return [float(v) for v in values]


def _run_evo_case(
    gt_data: np.ndarray,
    est_data: np.ndarray,
    *,
    evo_repo: Path,
    t_max_diff: float,
    offset_min: float,
    offset_max: float,
    offset_coarse_step: float,
    offset_refine_window: float,
    offset_refine_step: float,
    min_match_ratio: float,
) -> dict[str, object]:
    if str(evo_repo) not in sys.path:
        sys.path.insert(0, str(evo_repo))

    metrics = importlib.import_module("evo.core.metrics")
    sync = importlib.import_module("evo.core.sync")
    trajectory = importlib.import_module("evo.core.trajectory")
    main_ape = importlib.import_module("evo.main_ape")
    PoseTrajectory3D = trajectory.PoseTrajectory3D
    ape = main_ape.ape

    traj_ref = _build_evo_traj(PoseTrajectory3D, gt_data)
    traj_est = _build_evo_traj(PoseTrajectory3D, est_data)
    min_required = max(20, int(min_match_ratio * min(traj_ref.num_poses, traj_est.num_poses)))

    best_offset = 0.0
    best_key: tuple[bool, int, float] | None = None
    sweep_evals = 0

    coarse_offsets = _offset_grid(offset_min, offset_max, offset_coarse_step)
    for offset_s in coarse_offsets:
        sweep_evals += 1
        try:
            matches, rmse_raw = _evaluate_offset(sync, traj_ref, traj_est, t_max_diff, offset_s)
        except Exception:
            continue
        key = (matches >= min_required, matches, rmse_raw)
        if _better_offset_choice(best_key, key):
            best_key = key
            best_offset = offset_s

    if best_key is None:
        return {
            "status": "failed",
            "offset_s": float("nan"),
            "matches": 0,
            "ape_raw_rmse_m": float("nan"),
            "ape_se3_rmse_m": float("nan"),
            "improve_pct": float("nan"),
            "sweep_evals": sweep_evals,
            "error": "No valid offset in coarse sweep.",
        }

    refine_start = max(offset_min, best_offset - offset_refine_window)
    refine_end = min(offset_max, best_offset + offset_refine_window)
    refine_offsets = _offset_grid(refine_start, refine_end, offset_refine_step)
    for offset_s in refine_offsets:
        sweep_evals += 1
        try:
            matches, rmse_raw = _evaluate_offset(sync, traj_ref, traj_est, t_max_diff, offset_s)
        except Exception:
            continue
        key = (matches >= min_required, matches, rmse_raw)
        if _better_offset_choice(best_key, key):
            best_key = key
            best_offset = offset_s

    try:
        traj_ref_sync, traj_est_sync = sync.associate_trajectories(
            traj_ref,
            traj_est,
            max_diff=t_max_diff,
            offset_2=best_offset,
            first_name="ref",
            snd_name="est",
        )
    except Exception as exc:
        return {
            "status": "failed",
            "offset_s": best_offset,
            "matches": 0,
            "ape_raw_rmse_m": float("nan"),
            "ape_se3_rmse_m": float("nan"),
            "improve_pct": float("nan"),
            "sweep_evals": sweep_evals,
            "error": str(exc),
        }

    pose_rel = metrics.PoseRelation.translation_part
    raw_result = ape(
        traj_ref=copy.deepcopy(traj_ref_sync),
        traj_est=copy.deepcopy(traj_est_sync),
        pose_relation=pose_rel,
        align=False,
    )
    se3_result = ape(
        traj_ref=copy.deepcopy(traj_ref_sync),
        traj_est=copy.deepcopy(traj_est_sync),
        pose_relation=pose_rel,
        align=True,
    )
    raw_rmse = float(raw_result.stats.get("rmse", float("nan")))
    se3_rmse = float(se3_result.stats.get("rmse", float("nan")))
    improve_pct = (raw_rmse - se3_rmse) / (raw_rmse + 1e-12) * 100.0
    return {
        "status": "ok",
        "offset_s": float(best_offset),
        "matches": int(traj_ref_sync.num_poses),
        "ape_raw_rmse_m": raw_rmse,
        "ape_se3_rmse_m": se3_rmse,
        "improve_pct": float(improve_pct),
        "sweep_evals": sweep_evals,
        "error": "",
    }


def _fmt(value: object, ndigits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return ""
        return f"{x:.{ndigits}f}"
    text = str(value)
    if text.strip().lower() in {"none", "nan"}:
        return ""
    try:
        x = float(text)
    except ValueError:
        return str(value)
    if math.isnan(x):
        return ""
    return f"{x:.{ndigits}f}"


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _default_data_root() -> Path:
    return Path(os.getenv("EPA_DATA_ROOT", "~/epa_data")).expanduser()


def _default_cases_root() -> str:
    override = os.getenv("EPA_CASES_ROOT", "").strip() or os.getenv("EPA_ALIGNANYTHING_ROOT", "").strip()
    if override:
        return override
    return str(_default_data_root() / "benchmark_cases")


def _default_evo_repo() -> str:
    return os.getenv("EVO_REPO", "").strip()


def _parse_jobs(value: str) -> str:
    text = str(value).strip().lower()
    if text == "auto":
        return text
    try:
        jobs = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer or 'auto'.") from exc
    if jobs < 1:
        raise argparse.ArgumentTypeError("--jobs must be >= 1 or 'auto'.")
    return str(jobs)


def _resolve_jobs(value: str, case_count: int) -> int:
    text = str(value).strip().lower()
    if text == "auto":
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, max(1, case_count)))
    return int(text)


def _resolve_cli_path(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def _resolve_cli_executable(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.absolute()
    return (repo_root / p).absolute()


def _safe_root_label(path: Path) -> str:
    name = path.expanduser().resolve().name.strip()
    if not name:
        return "cases"
    label = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._-").lower()
    return label or "cases"


def _default_output_root(repo_root: Path, cases_root: Path) -> Path:
    return repo_root / "outputs" / f"{_safe_root_label(cases_root)}_bench"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent benchmark harness for EPA over a cases root."
    )
    parser.add_argument(
        "cases_root_pos",
        nargs="?",
        metavar="cases_root",
        help="Cases root containing benchmark/ and GT/. Overrides --cases-root.",
    )
    parser.add_argument(
        "--cases-root",
        "--alignanything-root",
        dest="cases_root",
        default=_default_cases_root(),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume-run",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epa-src",
        default="src",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--evo-repo",
        default=_default_evo_repo(),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--with-evo",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--case-pattern",
        action="append",
        default=[],
        help="Regex filter for case_id (can pass multiple).",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method filter, e.g. 'rovio,svo_stereo'.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max number of cases to run.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=_parse_jobs,
        default="auto",
        help="Number of benchmark cases to run in parallel, or 'auto'. Default: auto.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover and print cases only.")
    parser.add_argument(
        "--keep-prepared",
        action="store_true",
        help="Keep prepared_tum/ files for later case reruns. Default removes them after summary generation.",
    )
    parser.add_argument(
        "--eval-align",
        default="",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--dt-resample", type=float, default=0.001, help=argparse.SUPPRESS)
    parser.add_argument("--downsample-hz", type=float, default=100.0, help=argparse.SUPPRESS)
    parser.add_argument("--no-downsample", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--quat-interp",
        choices=["linear", "slerp"],
        default="linear",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--t-max-diff",
        type=float,
        default=0.02,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--offset-min", type=float, default=-20.0, help=argparse.SUPPRESS)
    parser.add_argument("--offset-max", type=float, default=20.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--offset-coarse-step",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--offset-refine-window",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--offset-refine-step",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--latex-main-max-rows",
        type=int,
        default=18,
        help=argparse.SUPPRESS,
    )
    return parser


def _filter_cases(
    cases: list[BenchmarkCase],
    case_patterns: list[str],
    methods_csv: str,
    limit: int,
) -> list[BenchmarkCase]:
    selected = cases
    if methods_csv.strip():
        methods = {token.strip() for token in methods_csv.split(",") if token.strip()}
        selected = [case for case in selected if case.method in methods]
    if case_patterns:
        regexes = [re.compile(pattern) for pattern in case_patterns]
        selected = [case for case in selected if any(rx.search(case.case_id) for rx in regexes)]
    if limit > 0:
        selected = selected[:limit]
    return selected


def _failed_summary_row(
    case: BenchmarkCase,
    *,
    gt: str,
    est: str,
    error: str,
) -> dict[str, object]:
    return {
        "case": case.case_id,
        "dataset": case.dataset,
        "method": case.method,
        "status": "failed",
        "gt": gt,
        "est": est,
        "epa_status": "failed",
        "evo_status": "failed",
        "epa_offset_est_s": float("nan"),
        "evo_offset_s": float("nan"),
        "epa_ate_rmse_raw_m": float("nan"),
        "epa_ate_rmse_step3_m": float("nan"),
        "evo_ape_raw_rmse_m": float("nan"),
        "evo_ape_se3_rmse_m": float("nan"),
        "epa_improve_pct": float("nan"),
        "evo_improve_pct": float("nan"),
        "epa_xcorr_peak": float("nan"),
        "epa_xcorr_psr": float("nan"),
        "epa_omega_improve_pct": float("nan"),
        "epa_matches_equivalent": float("nan"),
        "epa_sr_distance": float("nan"),
        "epa_sr_time": float("nan"),
        "epa_sr_distance_raw": float("nan"),
        "epa_sr_time_raw": float("nan"),
        "epa_sr_reliability_status": "",
        "epa_sr_warning_explanation": "",
        "epa_sim3_may_mask_failure": "",
        "epa_case_status": "",
        "epa_global_gate_failed": "",
        "epa_global_gate_value_m": float("nan"),
        "epa_global_gate_m": float("nan"),
        "epa_valid_distance_m": float("nan"),
        "epa_total_distance_m": float("nan"),
        "epa_step3_alignment_mode": "",
        "epa_step3_inlier_count": float("nan"),
        "epa_step3_rejected_count": float("nan"),
        "epa_step3_rejection_ratio": float("nan"),
        "epa_step3_stable_segment_used": float("nan"),
        "epa_step3_stable_solve_ratio": float("nan"),
        "epa_sim3_scale": float("nan"),
        "epa_sim3_reliable": "",
        "epa_sim3_warning": "",
        "epa_eval_align": "",
        "epa_sim3_stable_anchor_used": "",
        "epa_sim3_fallback_used": "",
        "epa_sim3_fallback_reason": "",
        "epa_orientation_unstable": "",
        "epa_orientation_ape_rmse_deg": float("nan"),
        "epa_orientation_rpe_rmse_deg": float("nan"),
        "epa_orientation_rpe_time_1s_rmse_deg": float("nan"),
        "epa_rpe_time_1s_trans_rmse_m": float("nan"),
        "epa_rpe_time_1s_rot_rmse_deg": float("nan"),
        "epa_orientation_warning": "",
        "epa_case_diagnosis_primary": "",
        "epa_case_diagnosis_summary": "",
        "epa_case_diagnosis_tags": "",
        "evo_matches": float("nan"),
        "epa_run_dir": "",
        "epa_stdout_log": "",
        "epa_stderr_log": "",
        "evo_sweep_evals": 0,
        "error": error,
    }


def _run_benchmark_case(
    case: BenchmarkCase,
    *,
    align_root: Path,
    prepared_dir: Path,
    logs_dir: Path,
    case_json_dir: Path,
    repo_root: Path,
    python_bin: Path,
    epa_src: Path,
    dt_resample: float,
    quat_interp: str,
    downsample_hz: float,
    no_downsample: bool,
    mplconfig_root: Path,
    output_root: Path,
    evo_repo: Path,
    with_evo: bool,
    t_max_diff: float,
    offset_min: float,
    offset_max: float,
    offset_coarse_step: float,
    offset_refine_window: float,
    offset_refine_step: float,
    min_match_ratio: float,
    eval_align: str = "",
) -> dict[str, object]:
    case_payload: dict[str, object] = {
        "case": case.case_id,
        "dataset": case.dataset,
        "method": case.method,
        "sequence": case.sequence,
        "gt_path": str(case.gt_path),
        "est_path": str(case.est_path),
        "status": "failed",
        "offset_policy": "epa_internal",
    }
    try:
        gt_data = load_pose_table(case.gt_path)
        est_data = load_pose_table(case.est_path)
    except Exception as exc:
        case_payload["error"] = f"Load error: {exc}"
        case_json_path = case_json_dir / f"{case.case_id}.json"
        case_json_path.parent.mkdir(parents=True, exist_ok=True)
        case_json_path.write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
        return _failed_summary_row(
            case,
            gt=str(case.gt_path),
            est=str(case.est_path),
            error=str(exc),
        )

    gt_tum = prepared_dir / f"{case.case_id}__gt.tum"
    est_tum = prepared_dir / f"{case.case_id}__est.tum"
    _write_tum(gt_tum, gt_data)
    _write_tum(est_tum, est_data)

    epa_stdout_log = logs_dir / "epa" / f"{case.case_id}.stdout.log"
    epa_stderr_log = logs_dir / "epa" / f"{case.case_id}.stderr.log"
    epa_result = _run_epa_case(
        case,
        gt_tum,
        est_tum,
        repo_root=repo_root,
        python_bin=python_bin,
        epa_src=epa_src,
        dt_resample=dt_resample,
        quat_interp=quat_interp,
        downsample_hz=downsample_hz,
        no_downsample=no_downsample,
        mplconfigdir=mplconfig_root / case.case_id,
        output_root=output_root,
        stdout_log_path=epa_stdout_log,
        stderr_log_path=epa_stderr_log,
        eval_align=eval_align,
    )

    if with_evo:
        try:
            evo_result = _run_evo_case(
                gt_data=gt_data,
                est_data=est_data,
                evo_repo=evo_repo,
                t_max_diff=t_max_diff,
                offset_min=offset_min,
                offset_max=offset_max,
                offset_coarse_step=offset_coarse_step,
                offset_refine_window=offset_refine_window,
                offset_refine_step=offset_refine_step,
                min_match_ratio=min_match_ratio,
            )
        except Exception as exc:
            evo_result = {
                "status": "failed",
                "offset_s": float("nan"),
                "matches": 0,
                "ape_raw_rmse_m": float("nan"),
                "ape_se3_rmse_m": float("nan"),
                "improve_pct": float("nan"),
                "sweep_evals": 0,
                "error": str(exc),
            }
    else:
        evo_result = {
            "status": "not_run",
            "offset_s": float("nan"),
            "matches": 0,
            "ape_raw_rmse_m": float("nan"),
            "ape_se3_rmse_m": float("nan"),
            "improve_pct": float("nan"),
            "sweep_evals": 0,
            "error": "",
        }

    epa_ok = epa_result.get("status") == "ok"
    row = {
        "case": case.case_id,
        "dataset": case.dataset,
        "method": case.method,
        "status": _bool_to_status(epa_ok),
        "gt": str(case.gt_path.relative_to(align_root)),
        "est": str(case.est_path.relative_to(align_root)),
        "epa_status": epa_result.get("status", "failed"),
        "evo_status": evo_result.get("status", "failed"),
        "epa_offset_est_s": _as_float(epa_result.get("offset_est_s")),
        "evo_offset_s": _as_float(evo_result.get("offset_s")),
        "epa_ate_rmse_raw_m": _as_float(epa_result.get("ate_rmse_raw_m")),
        "epa_ate_rmse_step3_m": _as_float(epa_result.get("ate_rmse_step3_m")),
        "evo_ape_raw_rmse_m": _as_float(evo_result.get("ape_raw_rmse_m")),
        "evo_ape_se3_rmse_m": _as_float(evo_result.get("ape_se3_rmse_m")),
        "epa_improve_pct": _as_float(epa_result.get("improve_raw_to_step3_pct")),
        "evo_improve_pct": _as_float(evo_result.get("improve_pct")),
        "epa_xcorr_peak": _as_float(epa_result.get("xcorr_peak")),
        "epa_xcorr_psr": _as_float(epa_result.get("xcorr_psr")),
        "epa_omega_improve_pct": _as_float(epa_result.get("omega_improve_pct")),
        "epa_matches_equivalent": _as_float(epa_result.get("evo_matches_equivalent")),
        "epa_sr_distance": _as_float(epa_result.get("sr_distance")),
        "epa_sr_time": _as_float(epa_result.get("sr_time")),
        "epa_sr_distance_raw": _as_float(epa_result.get("sr_distance_raw")),
        "epa_sr_time_raw": _as_float(epa_result.get("sr_time_raw")),
        "epa_sr_reliability_status": str(epa_result.get("sr_reliability_status", "")),
        "epa_sr_warning_explanation": str(epa_result.get("sr_warning_explanation", "")),
        "epa_sim3_may_mask_failure": str(epa_result.get("sim3_may_mask_failure", "")),
        "epa_case_status": str(epa_result.get("case_status", "")),
        "epa_global_gate_failed": str(epa_result.get("global_gate_failed", "")),
        "epa_global_gate_value_m": _as_float(epa_result.get("global_gate_value_m")),
        "epa_global_gate_m": _as_float(epa_result.get("global_gate_m")),
        "epa_valid_distance_m": _as_float(epa_result.get("valid_distance_m")),
        "epa_total_distance_m": _as_float(epa_result.get("total_distance_m")),
        "epa_step3_alignment_mode": str(epa_result.get("step3_alignment_mode", "")),
        "epa_step3_inlier_count": _as_float(epa_result.get("step3_inlier_count")),
        "epa_step3_rejected_count": _as_float(epa_result.get("step3_rejected_count")),
        "epa_step3_rejection_ratio": _as_float(epa_result.get("step3_rejection_ratio")),
        "epa_step3_stable_segment_used": _as_float(epa_result.get("step3_stable_segment_used")),
        "epa_step3_stable_solve_ratio": _as_float(epa_result.get("step3_stable_solve_ratio")),
        "epa_sim3_scale": _as_float(epa_result.get("sim3_scale")),
        "epa_sim3_reliable": str(epa_result.get("sim3_reliable", "")),
        "epa_sim3_warning": str(epa_result.get("sim3_warning", "")),
        "epa_eval_align": str(epa_result.get("eval_align", "")),
        "epa_sim3_stable_anchor_used": str(epa_result.get("sim3_stable_anchor_used", "")),
        "epa_sim3_fallback_used": str(epa_result.get("sim3_fallback_used", "")),
        "epa_sim3_fallback_reason": str(epa_result.get("sim3_fallback_reason", "")),
        "epa_orientation_unstable": str(epa_result.get("orientation_unstable", "")),
        "epa_orientation_ape_rmse_deg": _as_float(epa_result.get("orientation_ape_rmse_deg")),
        "epa_orientation_rpe_rmse_deg": _as_float(epa_result.get("orientation_rpe_rmse_deg")),
        "epa_orientation_rpe_time_1s_rmse_deg": _as_float(
            epa_result.get("orientation_rpe_time_1s_rmse_deg")
        ),
        "epa_rpe_time_1s_trans_rmse_m": _as_float(epa_result.get("rpe_time_1s_trans_rmse_m")),
        "epa_rpe_time_1s_rot_rmse_deg": _as_float(epa_result.get("rpe_time_1s_rot_rmse_deg")),
        "epa_orientation_warning": str(epa_result.get("orientation_warning", "")),
        "epa_case_diagnosis_primary": str(epa_result.get("diagnosis_primary", "")),
        "epa_case_diagnosis_summary": str(epa_result.get("diagnosis_summary", "")),
        "epa_case_diagnosis_tags": str(epa_result.get("diagnosis_tags", "")),
        "evo_matches": _as_float(evo_result.get("matches")),
        "epa_run_dir": str(epa_result.get("run_dir", "")),
        "epa_stdout_log": str(epa_result.get("stdout_log", "")),
        "epa_stderr_log": str(epa_result.get("stderr_log", "")),
        "evo_sweep_evals": int(evo_result.get("sweep_evals", 0)),
        "error": "",
    }
    if epa_result.get("status") != "ok":
        row["error"] = str(epa_result.get("stderr_tail", "") or evo_result.get("error", ""))
    elif with_evo and evo_result.get("status") != "ok":
        row["error"] = str(evo_result.get("error", ""))

    case_payload.update(
        {
            "status": row["status"],
            "gt_prepared_tum": str(gt_tum),
            "est_prepared_tum": str(est_tum),
            "epa": epa_result,
            "evo": evo_result,
        }
    )
    case_json_path = case_json_dir / f"{case.case_id}.json"
    case_json_path.parent.mkdir(parents=True, exist_ok=True)
    case_json_path.write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
    return row


def _relative_or_absolute(path_text: object, root: Path) -> str:
    try:
        path = Path(str(path_text))
        return str(path.relative_to(root))
    except Exception:
        return str(path_text)


def _summary_row_from_case_payload(
    case: BenchmarkCase,
    *,
    align_root: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    epa_result = payload.get("epa", {})
    evo_result = payload.get("evo", {})
    if not isinstance(epa_result, dict) or not isinstance(evo_result, dict):
        return _failed_summary_row(
            case,
            gt=_relative_or_absolute(payload.get("gt_path", case.gt_path), align_root),
            est=_relative_or_absolute(payload.get("est_path", case.est_path), align_root),
            error=str(payload.get("error", "Incomplete cached case payload.")),
        )

    epa_ok = epa_result.get("status") == "ok"
    row = {
        "case": case.case_id,
        "dataset": case.dataset,
        "method": case.method,
        "status": _bool_to_status(epa_ok),
        "gt": _relative_or_absolute(payload.get("gt_path", case.gt_path), align_root),
        "est": _relative_or_absolute(payload.get("est_path", case.est_path), align_root),
        "epa_status": epa_result.get("status", "failed"),
        "evo_status": evo_result.get("status", "failed"),
        "epa_offset_est_s": _as_float(epa_result.get("offset_est_s")),
        "evo_offset_s": _as_float(evo_result.get("offset_s")),
        "epa_ate_rmse_raw_m": _as_float(epa_result.get("ate_rmse_raw_m")),
        "epa_ate_rmse_step3_m": _as_float(epa_result.get("ate_rmse_step3_m")),
        "evo_ape_raw_rmse_m": _as_float(evo_result.get("ape_raw_rmse_m")),
        "evo_ape_se3_rmse_m": _as_float(evo_result.get("ape_se3_rmse_m")),
        "epa_improve_pct": _as_float(epa_result.get("improve_raw_to_step3_pct")),
        "evo_improve_pct": _as_float(evo_result.get("improve_pct")),
        "epa_xcorr_peak": _as_float(epa_result.get("xcorr_peak")),
        "epa_xcorr_psr": _as_float(epa_result.get("xcorr_psr")),
        "epa_omega_improve_pct": _as_float(epa_result.get("omega_improve_pct")),
        "epa_matches_equivalent": _as_float(epa_result.get("evo_matches_equivalent")),
        "epa_sr_distance": _as_float(epa_result.get("sr_distance")),
        "epa_sr_time": _as_float(epa_result.get("sr_time")),
        "epa_sr_distance_raw": _as_float(epa_result.get("sr_distance_raw")),
        "epa_sr_time_raw": _as_float(epa_result.get("sr_time_raw")),
        "epa_sr_reliability_status": str(epa_result.get("sr_reliability_status", "")),
        "epa_sr_warning_explanation": str(epa_result.get("sr_warning_explanation", "")),
        "epa_sim3_may_mask_failure": str(epa_result.get("sim3_may_mask_failure", "")),
        "epa_case_status": str(epa_result.get("case_status", "")),
        "epa_global_gate_failed": str(epa_result.get("global_gate_failed", "")),
        "epa_global_gate_value_m": _as_float(epa_result.get("global_gate_value_m")),
        "epa_global_gate_m": _as_float(epa_result.get("global_gate_m")),
        "epa_valid_distance_m": _as_float(epa_result.get("valid_distance_m")),
        "epa_total_distance_m": _as_float(epa_result.get("total_distance_m")),
        "epa_step3_alignment_mode": str(epa_result.get("step3_alignment_mode", "")),
        "epa_step3_inlier_count": _as_float(epa_result.get("step3_inlier_count")),
        "epa_step3_rejected_count": _as_float(epa_result.get("step3_rejected_count")),
        "epa_step3_rejection_ratio": _as_float(epa_result.get("step3_rejection_ratio")),
        "epa_step3_stable_segment_used": _as_float(epa_result.get("step3_stable_segment_used")),
        "epa_step3_stable_solve_ratio": _as_float(epa_result.get("step3_stable_solve_ratio")),
        "epa_sim3_scale": _as_float(epa_result.get("sim3_scale")),
        "epa_sim3_reliable": str(epa_result.get("sim3_reliable", "")),
        "epa_sim3_warning": str(epa_result.get("sim3_warning", "")),
        "epa_eval_align": str(epa_result.get("eval_align", "")),
        "epa_sim3_stable_anchor_used": str(epa_result.get("sim3_stable_anchor_used", "")),
        "epa_sim3_fallback_used": str(epa_result.get("sim3_fallback_used", "")),
        "epa_sim3_fallback_reason": str(epa_result.get("sim3_fallback_reason", "")),
        "epa_orientation_unstable": str(epa_result.get("orientation_unstable", "")),
        "epa_orientation_ape_rmse_deg": _as_float(epa_result.get("orientation_ape_rmse_deg")),
        "epa_orientation_rpe_rmse_deg": _as_float(epa_result.get("orientation_rpe_rmse_deg")),
        "epa_orientation_rpe_time_1s_rmse_deg": _as_float(
            epa_result.get("orientation_rpe_time_1s_rmse_deg")
        ),
        "epa_rpe_time_1s_trans_rmse_m": _as_float(epa_result.get("rpe_time_1s_trans_rmse_m")),
        "epa_rpe_time_1s_rot_rmse_deg": _as_float(epa_result.get("rpe_time_1s_rot_rmse_deg")),
        "epa_orientation_warning": str(epa_result.get("orientation_warning", "")),
        "epa_case_diagnosis_primary": str(epa_result.get("diagnosis_primary", "")),
        "epa_case_diagnosis_summary": str(epa_result.get("diagnosis_summary", "")),
        "epa_case_diagnosis_tags": str(epa_result.get("diagnosis_tags", "")),
        "evo_matches": _as_float(evo_result.get("matches")),
        "epa_run_dir": str(epa_result.get("run_dir", "")),
        "epa_stdout_log": str(epa_result.get("stdout_log", "")),
        "epa_stderr_log": str(epa_result.get("stderr_log", "")),
        "evo_sweep_evals": int(evo_result.get("sweep_evals", 0)),
        "error": "",
    }
    if epa_result.get("status") != "ok":
        row["error"] = str(epa_result.get("stderr_tail", "") or evo_result.get("error", ""))
    elif evo_result.get("status") == "failed":
        row["error"] = str(evo_result.get("error", ""))
    return row


def _load_cached_summary_row(
    case: BenchmarkCase,
    *,
    align_root: Path,
    case_json_dir: Path,
) -> dict[str, object] | None:
    case_json_path = case_json_dir / f"{case.case_id}.json"
    if not case_json_path.exists():
        return None
    try:
        payload = json.loads(case_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _summary_row_from_case_payload(case, align_root=align_root, payload=payload)


def run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    cases_root_arg = str(getattr(args, "cases_root_pos", "") or getattr(args, "cases_root", ""))
    align_root = _resolve_cli_path(cases_root_arg, repo_root)
    output_root = (
        _resolve_cli_path(args.output_root, repo_root)
        if str(args.output_root).strip()
        else _default_output_root(repo_root, align_root)
    )
    python_bin = _resolve_cli_executable(args.python_bin, repo_root)
    epa_src = _resolve_cli_path(args.epa_src, repo_root)
    evo_repo = _resolve_cli_path(args.evo_repo, repo_root) if str(args.evo_repo).strip() else Path("")

    cases, unresolved = discover_cases(align_root)
    cases = _filter_cases(cases, args.case_pattern, args.methods, args.limit)
    print(f"Discovered cases: {len(cases)} (unresolved GT mappings: {len(unresolved)})")
    if args.dry_run:
        for case in cases:
            print(f"- {case.case_id} | {case.gt_path} | {case.est_path}")
        return 0

    resume_run = str(getattr(args, "resume_run", "")).strip()
    run_dir = _resolve_cli_path(resume_run, repo_root) if resume_run else _make_run_dir(output_root)
    if resume_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = run_dir / "prepared_tum"
    logs_dir = run_dir / "logs"
    case_json_dir = run_dir / "cases"
    epa_runs_dir = run_dir / "epa_runs"
    mplconfigdir = run_dir / ".mplconfig"
    mplconfigdir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "harness_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if unresolved:
        _write_csv(unresolved, run_dir / "unresolved_cases.csv")

    jobs = _resolve_jobs(args.jobs, len(cases))
    worker_kwargs = {
        "align_root": align_root,
        "prepared_dir": prepared_dir,
        "logs_dir": logs_dir,
        "case_json_dir": case_json_dir,
        "repo_root": repo_root,
        "python_bin": python_bin,
        "epa_src": epa_src,
        "dt_resample": args.dt_resample,
        "quat_interp": args.quat_interp,
        "downsample_hz": args.downsample_hz,
        "no_downsample": bool(getattr(args, "no_downsample", False)),
        "mplconfig_root": mplconfigdir,
        "output_root": epa_runs_dir,
        "evo_repo": evo_repo,
        "with_evo": bool(getattr(args, "with_evo", False)),
        "t_max_diff": args.t_max_diff,
        "offset_min": args.offset_min,
        "offset_max": args.offset_max,
        "offset_coarse_step": args.offset_coarse_step,
        "offset_refine_window": args.offset_refine_window,
        "offset_refine_step": args.offset_refine_step,
        "min_match_ratio": args.min_match_ratio,
        "eval_align": str(getattr(args, "eval_align", "")),
    }
    summary_rows_by_index: dict[int, dict[str, object]] = {}
    pending_cases: list[tuple[int, BenchmarkCase]] = []
    if resume_run:
        for index, case in enumerate(cases, start=1):
            cached = _load_cached_summary_row(case, align_root=align_root, case_json_dir=case_json_dir)
            if cached is None:
                pending_cases.append((index, case))
            else:
                summary_rows_by_index[index] = cached
        print(f"Resuming run dir: {run_dir}")
        print(f"Cached cases: {len(summary_rows_by_index)}; pending cases: {len(pending_cases)}")
    else:
        pending_cases = list(enumerate(cases, start=1))

    if jobs == 1:
        for index, case in pending_cases:
            print(f"[{index}/{len(cases)}] {case.case_id}")
            summary_rows_by_index[index] = _run_benchmark_case(case, **worker_kwargs)
    else:
        print(f"Running benchmark cases with {jobs} workers.")
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_run_benchmark_case, case, **worker_kwargs): (index, case)
                for index, case in pending_cases
            }
            completed = 0
            for future in as_completed(futures):
                index, case = futures[future]
                completed += 1
                try:
                    summary_rows_by_index[index] = future.result()
                    status = summary_rows_by_index[index].get("status", "unknown")
                    print(f"[{completed}/{len(cases)}] {case.case_id} -> {status}")
                except Exception as exc:
                    print(f"[{completed}/{len(cases)}] {case.case_id} -> failed: {exc}")
                    summary_rows_by_index[index] = _failed_summary_row(
                        case,
                        gt=str(case.gt_path),
                        est=str(case.est_path),
                        error=str(exc),
                    )

    summary_rows = [summary_rows_by_index[index] for index in range(1, len(cases) + 1)]

    _write_csv(summary_rows, run_dir / "summary.csv")
    _write_public_summary_csv(summary_rows, run_dir / "summary_public.csv")
    _write_summary_md(summary_rows, run_dir / "summary.md")
    _write_summary_html(summary_rows, run_dir / "summary.html")

    try:
        write_latex_tables(
            summary_rows,
            run_dir / "paper_tables",
            max_main_rows=max(1, int(args.latex_main_max_rows)),
        )
    except Exception as exc:
        print(f"[warn] Failed to generate LaTeX tables: {exc}")

    if not bool(getattr(args, "keep_prepared", False)) and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    done = sum(1 for row in summary_rows if row["status"] == "ok")
    print(f"Finished: {done}/{len(summary_rows)} cases epa_ok")
    print(f"Run dir: {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
