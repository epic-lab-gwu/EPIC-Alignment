from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import html
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .latex_summary import write_latex_tables


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dataset: str
    method: str
    sequence: str
    gt_path: Path
    est_path: Path


def _safe_case_id(dataset: str, sequence: str, method: str) -> str:
    raw = f"{dataset}_{sequence}_{method}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")


def _case_id_qualifier(rel_pose: Path, method: str, sequence: str) -> str:
    parts = rel_pose.parts
    if "pose" in parts:
        pose_idx = parts.index("pose")
        qualifier_parts = list(parts[1:pose_idx])
    else:
        qualifier_parts = list(parts[1:-1])
        if qualifier_parts and qualifier_parts[-1] == sequence:
            qualifier_parts = qualifier_parts[:-1]
        if qualifier_parts and qualifier_parts[-1] == method:
            qualifier_parts = qualifier_parts[:-1]
    return "_".join(part for part in qualifier_parts if part)


_GT_SUFFIXES = (".txt", ".tum", ".csv")
_EST_SUFFIXES = (".txt", ".tum", ".csv")
_EST_STEM_SUFFIXES = ("_poses", "_pose", "_trajectory", "_traj")
_GENERIC_EST_STEMS = {"poses", "pose", "trajectory", "traj"}


def _strip_est_suffix(stem: str) -> str:
    for suffix in _EST_STEM_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return "" if stem in _GENERIC_EST_STEMS else stem


def _is_est_trajectory_file(path: Path) -> bool:
    if path.suffix.lower() not in _EST_SUFFIXES:
        return False
    stem = path.stem
    if path.suffix.lower() == ".txt" and not (
        stem in _GENERIC_EST_STEMS or any(stem.endswith(suffix) for suffix in _EST_STEM_SUFFIXES)
    ):
        return False
    try:
        _sniff_trajectory_columns(path)
    except Exception:
        return False
    return True


def _with_supported_gt_suffixes(path_without_suffix: Path) -> list[Path]:
    return [path_without_suffix.with_suffix(suffix) for suffix in _GT_SUFFIXES]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _glob_gt(gt_root: Path, pattern: str) -> Path | None:
    matches: list[Path] = []
    for suffix in _GT_SUFFIXES:
        matches.extend(p for p in gt_root.glob(f"{pattern}{suffix}") if p.is_file())
    if matches:
        return sorted(matches)[0]
    return None


def _case_candidates(rel_pose: Path) -> list[tuple[str, str, str]]:
    parts = rel_pose.parts
    if len(parts) < 2:
        return []

    dataset = parts[0]
    dirs = list(parts[1:-1])
    file_base = _strip_est_suffix(Path(parts[-1]).stem)
    candidates: list[tuple[str, str, str]] = []

    def add(method: str, sequence: str) -> None:
        method = str(method).strip()
        sequence = str(sequence).strip()
        if method and sequence:
            candidates.append((dataset, method, sequence))

    if "pose" in parts:
        pose_idx = parts.index("pose")
        after_pose = list(parts[pose_idx + 1 : -1])
        if len(after_pose) >= 2:
            add(after_pose[0], after_pose[1])
        if len(after_pose) >= 1 and file_base:
            add(after_pose[0], file_base)
            add(file_base, after_pose[-1])

    if len(dirs) >= 2:
        add(dirs[-2], dirs[-1])
        add(dirs[-1], dirs[-2])
    if len(dirs) >= 1 and file_base:
        add(dirs[-1], file_base)
        add(file_base, dirs[-1])
    if file_base:
        add("default", file_base)

    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _sniff_delimiter(path: Path) -> str | None:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                return ","
            return None
    raise ValueError(f"No numeric rows found in trajectory file: {path}")


def _sniff_trajectory_columns(path: Path) -> int:
    delimiter = _sniff_delimiter(path)
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split(delimiter) if delimiter == "," else line.split()
            values = [float(col.strip()) for col in cols if str(col).strip()]
            if len(values) < 8:
                raise ValueError(f"Trajectory must have at least 8 columns: {path}")
            return len(values)
    raise ValueError(f"No numeric rows found in trajectory file: {path}")


def load_pose_table(path: Path) -> np.ndarray:
    delimiter = _sniff_delimiter(path)
    arr = np.loadtxt(path, comments="#", delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 8:
        raise ValueError(f"Trajectory must have at least 8 columns: {path}")

    data = np.asarray(arr[:, :8], dtype=float)
    finite_mask = np.isfinite(data).all(axis=1)
    data = data[finite_mask]
    if data.shape[0] < 2:
        raise ValueError(f"Trajectory has fewer than 2 valid rows: {path}")

    q = data[:, 4:8]
    q_norm = np.linalg.norm(q, axis=1)
    valid_q = q_norm > 1e-12
    data = data[valid_q]
    q = q[valid_q] / q_norm[valid_q, None]
    data[:, 4:8] = q
    if data.shape[0] < 2:
        raise ValueError(f"Trajectory has fewer than 2 valid quaternion rows: {path}")

    order = np.argsort(data[:, 0], kind="mergesort")
    data = data[order]
    t = data[:, 0]
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    data = data[keep]
    if data.shape[0] < 2:
        raise ValueError(f"Trajectory has fewer than 2 unique timestamps: {path}")
    return data


def _write_tum(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data[:, :8], fmt="%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f")


def _resolve_gt_for_case(gt_root: Path, rel_pose: Path, sequence: str) -> Path | None:
    parts = rel_pose.parts
    dataset = parts[0]

    if dataset == "euroc_mav":
        cand = _first_existing(_with_supported_gt_suffixes(gt_root / "euroc_mav" / sequence))
        if cand is not None:
            return cand

    if dataset == "grand_tour":
        match = _glob_gt(gt_root / "grand_tour", f"**/{sequence}")
        if match is not None:
            return match

    if dataset == "lamaria":
        subset = parts[1] if len(parts) > 1 else ""
        if subset:
            cand = _first_existing(_with_supported_gt_suffixes(gt_root / "lamaria" / subset / sequence))
            if cand is not None:
                return cand
        match = _glob_gt(gt_root / "lamaria", f"**/{sequence}")
        if match is not None:
            return match

    if dataset == "uzh_fpv":
        subset = parts[1] if len(parts) > 1 else ""
        if subset:
            cand = _first_existing(_with_supported_gt_suffixes(gt_root / f"uzhfpv_{subset}" / sequence))
            if cand is not None:
                return cand
        match = _glob_gt(gt_root, f"uzhfpv*/{sequence}")
        if match is not None:
            return match

    return _glob_gt(gt_root, f"**/{sequence}")


def _format_cases_root_error(
    cases_root: Path,
    *,
    missing_entries: list[str],
) -> str:
    env_data_root = os.getenv("EPA_DATA_ROOT", "").strip() or "(unset)"
    env_cases_root = (
        os.getenv("EPA_CASES_ROOT", "").strip()
        or os.getenv("EPA_ALIGNANYTHING_ROOT", "").strip()
        or "(unset)"
    )
    suggested_data_root = _default_data_root()
    suggested_cases_root = Path(_default_cases_root()).expanduser()

    detail_lines = [f"- Missing required path: {cases_root / entry}" for entry in missing_entries]
    if not cases_root.exists():
        detail_lines.insert(0, f"- Base path does not exist: {cases_root}")

    cmd_example = f"epa_bench {suggested_cases_root}"

    lines = [
        f"Invalid --cases-root: {cases_root}",
        "Expected a cases root containing both benchmark/ and GT/.",
        *detail_lines,
        "",
        "Current environment:",
        f"- EPA_DATA_ROOT={env_data_root}",
        f"- EPA_CASES_ROOT={env_cases_root}",
        "",
        "Example fix:",
        f"  export EPA_DATA_ROOT={suggested_data_root}",
        f"  export EPA_CASES_ROOT={suggested_cases_root}",
        f"  {cmd_example}",
    ]
    return "\n".join(lines)


def discover_cases(cases_root: Path) -> tuple[list[BenchmarkCase], list[dict[str, str]]]:
    bench_root = cases_root / "benchmark"
    gt_root = cases_root / "GT"
    missing_entries: list[str] = []
    if not bench_root.exists():
        missing_entries.append("benchmark")
    if not gt_root.exists():
        missing_entries.append("GT")
    if missing_entries:
        raise FileNotFoundError(
            _format_cases_root_error(
                cases_root=cases_root,
                missing_entries=missing_entries,
            )
        )

    selected_cases: list[tuple[str, str, str, Path, Path, Path]] = []
    unresolved: list[dict[str, str]] = []
    for est_path in sorted(p for p in bench_root.rglob("*") if p.is_file() and _is_est_trajectory_file(p)):
        rel_pose = est_path.relative_to(bench_root)
        candidates = _case_candidates(rel_pose)
        if not candidates:
            continue

        selected: tuple[str, str, str, Path] | None = None
        first_candidate = candidates[0]
        for dataset, method, sequence in candidates:
            gt_path = _resolve_gt_for_case(gt_root, rel_pose, sequence)
            if gt_path is not None:
                selected = (dataset, method, sequence, gt_path)
                break

        if selected is None:
            dataset, method, sequence = first_candidate
            gt_path = None
        else:
            dataset, method, sequence, gt_path = selected

        if gt_path is None:
            unresolved.append(
                {
                    "est_relpath": str(rel_pose),
                    "sequence": sequence,
                    "method": method,
                    "dataset": dataset,
                }
            )
            continue
        case_id = _safe_case_id(dataset, sequence, method)
        selected_cases.append((case_id, dataset, method, sequence, gt_path, est_path, rel_pose))

    case_id_counts: dict[str, int] = {}
    for case_id, *_ in selected_cases:
        case_id_counts[case_id] = case_id_counts.get(case_id, 0) + 1

    cases: list[BenchmarkCase] = []
    for case_id, dataset, method, sequence, gt_path, est_path, rel_pose in selected_cases:
        if case_id_counts[case_id] > 1:
            qualifier = _case_id_qualifier(rel_pose, method, sequence)
            if qualifier:
                case_id = _safe_case_id(dataset, f"{qualifier}_{sequence}", method)
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                dataset=dataset,
                method=method,
                sequence=sequence,
                gt_path=gt_path,
                est_path=est_path,
            )
        )
    return cases, unresolved


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
        "orientation_warning": "",
        "diagnosis_primary": "",
        "diagnosis_summary": "",
        "diagnosis_tags": "",
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
    orientation = payload.get("orientation_diagnostics", {})
    case_diagnostics = payload.get("case_diagnostics", {})
    metadata = payload.get("metadata", {})
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
    if str(eval_step3.get("align_mode", "")).lower() == "sim3":
        result["sim3_reliable"] = str(bool(eval_step3.get("sim3_reliable", True)))
        result["sim3_warning"] = str(eval_step3.get("sim3_warning", ""))
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
    return str(value)


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


def _write_summary_md(rows: list[dict[str, object]], path: Path) -> None:
    total = len(rows)
    epa_ok = sum(1 for row in rows if row.get("status") == "ok")
    has_evo = any(row.get("evo_status") == "ok" for row in rows)
    lines = [
        "# EPA Independent Benchmark",
        "",
        f"- total cases: {total}",
        f"- epa_ok: {epa_ok}",
        "",
        "## Table 1: EPA Metrics",
        "",
        "| case | status | raw_rmse_m (↓) | aligned_rmse_m (↓) | improve_pct (↑) | offset_s (N/A) | matches (↑) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {status} | {vr} | {va} | {vi} | {vo} | {vm} |".format(
                case=row.get("case", ""),
                status=row.get("epa_status", ""),
                vr=_fmt(row.get("epa_ate_rmse_raw_m"), 3),
                va=_fmt(row.get("epa_ate_rmse_step3_m"), 3),
                vi=_fmt(row.get("epa_improve_pct"), 2),
                vo=_fmt(row.get("epa_offset_est_s"), 3),
                vm=_fmt(row.get("epa_matches_equivalent"), 0),
            )
        )
    lines.extend(
        [
            "",
            "## Table 2: Valid Segment Status",
            "",
            "| case | case_status | SR_dist_% (↑) | SR_time_% (↑) | global_gate_failed | global_gate_value_m | valid_distance_m | step3_mode | stable_solve_% | rejected_% | sim3_scale | sim3_reliable | orientation_unstable |",
            "|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {case_status} | {sr_dist} | {sr_time} | {gate_failed} | {gate_value} | {valid_dist} | {mode} | {stable_pct} | {reject_pct} | {sim3_scale} | {sim3_reliable} | {orientation_unstable} |".format(
                case=row.get("case", ""),
                case_status=row.get("epa_case_status", ""),
                sr_dist=_fmt(_as_float(row.get("epa_sr_distance")) * 100.0, 2),
                sr_time=_fmt(_as_float(row.get("epa_sr_time")) * 100.0, 2),
                gate_failed=row.get("epa_global_gate_failed", ""),
                gate_value=_fmt(row.get("epa_global_gate_value_m"), 3),
                valid_dist=_fmt(row.get("epa_valid_distance_m"), 3),
                mode=row.get("epa_step3_alignment_mode", ""),
                stable_pct=_fmt(_as_float(row.get("epa_step3_stable_solve_ratio")) * 100.0, 2),
                reject_pct=_fmt(_as_float(row.get("epa_step3_rejection_ratio")) * 100.0, 2),
                sim3_scale=_fmt(row.get("epa_sim3_scale"), 6),
                sim3_reliable=row.get("epa_sim3_reliable", ""),
                orientation_unstable=row.get("epa_orientation_unstable", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Table 3: epa-only Metrics",
            "",
            "| case | epa_xcorr_peak (↑) | epa_xcorr_psr (↑) | epa_omega_improve_pct (↑) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {peak} | {psr} | {omega} |".format(
                case=row.get("case", ""),
                peak=_fmt(row.get("epa_xcorr_peak"), 3),
                psr=_fmt(row.get("epa_xcorr_psr"), 2),
                omega=_fmt(row.get("epa_omega_improve_pct"), 2),
            )
        )
    if has_evo:
        lines.extend(
            [
                "",
                "## Table 4: Legacy evo Baseline",
                "",
                "| case | evo_status | evo_aligned_rmse_m (↓) | evo_improve_pct (↑) | evo_offset_s (N/A) | evo_sweep_evals (N/A) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {case} | {status} | {rmse} | {improve} | {offset} | {sweeps} |".format(
                    case=row.get("case", ""),
                    status=row.get("evo_status", ""),
                    rmse=_fmt(row.get("evo_ape_se3_rmse_m"), 3),
                    improve=_fmt(row.get("evo_improve_pct"), 2),
                    offset=_fmt(row.get("evo_offset_s"), 3),
                    sweeps=_fmt(row.get("evo_sweep_evals"), 0),
                )
            )
    lines.extend(
        [
            "",
            "Note: `↑` means larger is better, `↓` means smaller is better, and `N/A` marks process parameters without a universal quality direction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rel_link(path_text: object, base_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_dir():
        candidate = path / "plots" / "step3_alignment_map.png"
        if candidate.exists():
            path = candidate
        else:
            path = path / "metrics.json"
    try:
        href = os.path.relpath(path, start=base_dir)
    except ValueError:
        href = str(path)
    label = path.name or str(path)
    return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'


def _interactive_thumb(path_text: object, base_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    run_path = Path(text)
    img = run_path / "plots" / "step3_alignment_map.png"
    interactive = run_path / "interactive_report.html"
    if interactive.exists() and not img.exists():
        try:
            href = os.path.relpath(interactive, start=base_dir)
        except ValueError:
            href = str(interactive)
        return f'<a href="{html.escape(href, quote=True)}">interactive_report.html</a>'
    if not img.exists():
        return _rel_link(path_text, base_dir)
    try:
        img_href = os.path.relpath(img, start=base_dir)
    except ValueError:
        img_href = str(img)
    try:
        target_href = os.path.relpath(interactive if interactive.exists() else img, start=base_dir)
    except ValueError:
        target_href = str(interactive if interactive.exists() else img)
    label = "interactive_report.html" if interactive.exists() else "step3_alignment_map.png"
    return (
        f'<a class="thumbLink" href="{html.escape(target_href, quote=True)}">'
        f'<img class="thumb" src="{html.escape(img_href, quote=True)}" alt="step3 alignment">'
        f'<span class="plotCount">{html.escape(label)}</span>'
        "</a>"
    )


def _summary_filter_options(rows: list[dict[str, object]], key: str) -> str:
    values = sorted({str(row.get(key, "") or "") for row in rows if str(row.get(key, "") or "")})
    options = ['<option value="">all</option>']
    options.extend(
        f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>'
        for value in values
    )
    return "".join(options)


def _write_summary_html(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    epa_ok = sum(1 for row in rows if row.get("status") == "ok")
    status_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("epa_case_status", "unknown") or "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1
    status_bits = " ".join(
        f"<span><b>{html.escape(key)}</b>: {value}</span>"
        for key, value in sorted(status_counts.items())
    )

    table_rows = []
    for row in rows:
        sr_dist = _as_float(row.get("epa_sr_distance")) * 100.0
        sr_time = _as_float(row.get("epa_sr_time")) * 100.0
        reject_pct = _as_float(row.get("epa_step3_rejection_ratio")) * 100.0
        stable_pct = _as_float(row.get("epa_step3_stable_solve_ratio")) * 100.0
        sr_data = "" if not math.isfinite(sr_dist) else f"{sr_dist:.12g}"
        gate_failed = str(row.get("epa_global_gate_failed", ""))
        case_status = str(row.get("epa_case_status", ""))
        dataset = str(row.get("dataset", ""))
        method = str(row.get("method", ""))
        step3_mode = str(row.get("epa_step3_alignment_mode", ""))
        sim3_reliable = str(row.get("epa_sim3_reliable", ""))
        orientation_unstable = str(row.get("epa_orientation_unstable", ""))
        diagnosis_primary = str(row.get("epa_case_diagnosis_primary", ""))
        diagnosis_summary = str(row.get("epa_case_diagnosis_summary", ""))
        status_class = "ok" if case_status == "valid_segment" else "warn"
        if case_status == "globally_failed" or row.get("status") != "ok":
            status_class = "bad"
        table_rows.append(
            "<tr "
            f'data-dataset="{html.escape(dataset, quote=True)}" '
            f'data-method="{html.escape(method, quote=True)}" '
            f'data-status="{html.escape(case_status, quote=True)}" '
            f'data-step3="{html.escape(step3_mode, quote=True)}" '
            f'data-sim3-reliable="{html.escape(sim3_reliable, quote=True)}" '
            f'data-orientation="{html.escape(orientation_unstable, quote=True)}" '
            f'data-diagnosis="{html.escape(diagnosis_primary, quote=True)}" '
            f'data-sr="{html.escape(sr_data, quote=True)}">'
            f"<td>{html.escape(str(row.get('case', '')))}</td>"
            f"<td>{html.escape(dataset)}</td>"
            f"<td>{html.escape(method)}</td>"
            f"<td class='{status_class}'>{html.escape(case_status)}</td>"
            f"<td>{_fmt(sr_dist, 2)}</td>"
            f"<td>{_fmt(sr_time, 2)}</td>"
            f"<td>{html.escape(gate_failed)}</td>"
            f"<td>{_fmt(row.get('epa_global_gate_value_m'), 3)}</td>"
            f"<td>{_fmt(row.get('epa_valid_distance_m'), 3)}</td>"
            f"<td>{html.escape(step3_mode)}</td>"
            f"<td>{_fmt(stable_pct, 2)}</td>"
            f"<td>{_fmt(reject_pct, 2)}</td>"
            f"<td>{_fmt(row.get('epa_sim3_scale'), 6)}</td>"
            f"<td>{html.escape(sim3_reliable)}</td>"
            f"<td>{html.escape(str(row.get('epa_sim3_warning', '')))}</td>"
            f"<td>{html.escape(orientation_unstable)}</td>"
            f"<td>{html.escape(diagnosis_primary)}</td>"
            f"<td>{html.escape(diagnosis_summary)}</td>"
            f"<td>{_fmt(row.get('epa_orientation_ape_rmse_deg'), 2)}</td>"
            f"<td>{_fmt(row.get('epa_orientation_rpe_rmse_deg'), 2)}</td>"
            f"<td>{_fmt(row.get('epa_orientation_rpe_time_1s_rmse_deg'), 2)}</td>"
            f"<td>{html.escape(str(row.get('epa_orientation_warning', '')))}</td>"
            f"<td>{_fmt(row.get('epa_ate_rmse_step3_m'), 3)}</td>"
            f"<td>{_interactive_thumb(row.get('epa_run_dir', ''), path.parent)}</td>"
            f"<td>{_rel_link(row.get('epa_run_dir', ''), path.parent)}</td>"
            "</tr>"
        )
    css = """
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#17202a;background:#fafafa}
h1{font-size:24px;margin:0 0 12px}
.summary{display:flex;gap:18px;margin:0 0 18px;color:#445}
.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;margin:0 0 16px;padding:12px;background:#fff;border:1px solid #d7dee8}
.filterGroup{display:flex;flex-direction:column;gap:4px}
.filterGroup label{font-size:11px;color:#526070;text-transform:uppercase}
.filterGroup select,.filterGroup input{height:30px;border:1px solid #c9d3df;background:#fff;padding:4px 7px;font-size:13px;min-width:112px}
.filters button{height:30px;border:1px solid #9fb1c5;background:#eef3f8;padding:4px 10px;font-size:13px;cursor:pointer}
.resultCount{font-size:13px;color:#445;margin-left:auto}
table{border-collapse:collapse;width:100%;background:white;border:1px solid #d7dee8}
th,td{padding:7px 9px;border-bottom:1px solid #e6ebf2;text-align:left;font-size:13px;white-space:nowrap}
th{position:sticky;top:0;background:#eef3f8;z-index:1}
tr:hover{background:#f6f9fc}
.thumbLink{position:relative;display:block;width:180px;height:132px}
.thumb{width:180px;height:132px;object-fit:contain;border:1px solid #d7dee8;background:#fff;display:block}
.plotCount{position:absolute;right:6px;bottom:6px;padding:2px 6px;border-radius:3px;background:rgba(23,32,42,.78);color:white;font-size:11px}
.ok{color:#146c43;font-weight:600}.warn{color:#9a6700;font-weight:600}.bad{color:#b42318;font-weight:600}
"""
    doc = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EPA Benchmark Summary</title>",
        f"<style>{css}</style>",
        "</head><body>",
        "<h1>EPA Benchmark Summary</h1>",
        f"<div class='summary'><span><b>total</b>: {total}</span><span><b>epa_ok</b>: {epa_ok}</span>{status_bits}</div>",
        "<div class='filters'>",
        f"<div class='filterGroup'><label for='filterDataset'>dataset</label><select id='filterDataset'>{_summary_filter_options(rows, 'dataset')}</select></div>",
        f"<div class='filterGroup'><label for='filterMethod'>method</label><select id='filterMethod'>{_summary_filter_options(rows, 'method')}</select></div>",
        f"<div class='filterGroup'><label for='filterStatus'>case_status</label><select id='filterStatus'>{_summary_filter_options(rows, 'epa_case_status')}</select></div>",
        f"<div class='filterGroup'><label for='filterStep3'>Step3 mode</label><select id='filterStep3'>{_summary_filter_options(rows, 'epa_step3_alignment_mode')}</select></div>",
        f"<div class='filterGroup'><label for='filterSim3'>Sim3 reliable</label><select id='filterSim3'>{_summary_filter_options(rows, 'epa_sim3_reliable')}</select></div>",
        f"<div class='filterGroup'><label for='filterOrientation'>orientation</label><select id='filterOrientation'>{_summary_filter_options(rows, 'epa_orientation_unstable')}</select></div>",
        f"<div class='filterGroup'><label for='filterDiagnosis'>diagnosis</label><select id='filterDiagnosis'>{_summary_filter_options(rows, 'epa_case_diagnosis_primary')}</select></div>",
        "<div class='filterGroup'><label for='filterSrMin'>SR min %</label><input id='filterSrMin' type='number' min='0' max='100' step='0.01'></div>",
        "<div class='filterGroup'><label for='filterSrMax'>SR max %</label><input id='filterSrMax' type='number' min='0' max='100' step='0.01'></div>",
        "<button id='resetFilters' type='button'>reset</button>",
        f"<span class='resultCount'><b id='visibleCount'>{total}</b> / {total}</span>",
        "</div>",
        "<table><thead><tr>",
        "<th>case</th><th>dataset</th><th>method</th><th>case_status</th><th>SR_dist_%</th><th>SR_time_%</th><th>global_gate_failed</th><th>gate_value_m</th><th>valid_distance_m</th><th>step3_mode</th><th>stable_solve_%</th><th>rejected_%</th><th>sim3_scale</th><th>sim3_reliable</th><th>sim3_warning</th><th>orientation_unstable</th><th>diagnosis</th><th>diagnosis_summary</th><th>APE_rot_deg</th><th>RPE_rot_deg</th><th>RPE_1s_rot_deg</th><th>orientation_warning</th><th>step3_rmse_m</th><th>interactive</th><th>run</th>",
        "</tr></thead><tbody id='summaryBody'>",
        *table_rows,
        "</tbody></table>",
        """<script>
(function(){
  const controls = {
    dataset: document.getElementById('filterDataset'),
    method: document.getElementById('filterMethod'),
    status: document.getElementById('filterStatus'),
    step3: document.getElementById('filterStep3'),
    sim3: document.getElementById('filterSim3'),
    orientation: document.getElementById('filterOrientation'),
    diagnosis: document.getElementById('filterDiagnosis'),
    srMin: document.getElementById('filterSrMin'),
    srMax: document.getElementById('filterSrMax')
  };
  const rows = Array.from(document.querySelectorAll('#summaryBody tr'));
  const visibleCount = document.getElementById('visibleCount');
  function matchSelect(row, name, control) {
    return !control.value || row.dataset[name] === control.value;
  }
  function applyFilters() {
    const minText = controls.srMin.value.trim();
    const maxText = controls.srMax.value.trim();
    const minSr = minText === '' ? -Infinity : Number(minText);
    const maxSr = maxText === '' ? Infinity : Number(maxText);
    let shown = 0;
    rows.forEach(row => {
      const sr = Number(row.dataset.sr);
      const hasSr = row.dataset.sr !== '' && Number.isFinite(sr);
      let ok = true;
      ok = ok && matchSelect(row, 'dataset', controls.dataset);
      ok = ok && matchSelect(row, 'method', controls.method);
      ok = ok && matchSelect(row, 'status', controls.status);
      ok = ok && matchSelect(row, 'step3', controls.step3);
      ok = ok && matchSelect(row, 'sim3Reliable', controls.sim3);
      ok = ok && matchSelect(row, 'orientation', controls.orientation);
      ok = ok && matchSelect(row, 'diagnosis', controls.diagnosis);
      if (minText !== '' || maxText !== '') ok = ok && hasSr && sr >= minSr && sr <= maxSr;
      row.style.display = ok ? '' : 'none';
      if (ok) shown += 1;
    });
    visibleCount.textContent = shown;
  }
  Object.values(controls).forEach(control => control.addEventListener('input', applyFilters));
  document.getElementById('resetFilters').addEventListener('click', () => {
    Object.values(controls).forEach(control => { control.value = ''; });
    applyFilters();
  });
})();
</script>""",
        "</body></html>",
    ]
    path.write_text("\n".join(doc) + "\n", encoding="utf-8")



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
        "epa_orientation_unstable": "",
        "epa_orientation_ape_rmse_deg": float("nan"),
        "epa_orientation_rpe_rmse_deg": float("nan"),
        "epa_orientation_rpe_time_1s_rmse_deg": float("nan"),
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
        "epa_orientation_unstable": str(epa_result.get("orientation_unstable", "")),
        "epa_orientation_ape_rmse_deg": _as_float(epa_result.get("orientation_ape_rmse_deg")),
        "epa_orientation_rpe_rmse_deg": _as_float(epa_result.get("orientation_rpe_rmse_deg")),
        "epa_orientation_rpe_time_1s_rmse_deg": _as_float(
            epa_result.get("orientation_rpe_time_1s_rmse_deg")
        ),
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
        "epa_orientation_unstable": str(epa_result.get("orientation_unstable", "")),
        "epa_orientation_ape_rmse_deg": _as_float(epa_result.get("orientation_ape_rmse_deg")),
        "epa_orientation_rpe_rmse_deg": _as_float(epa_result.get("orientation_rpe_rmse_deg")),
        "epa_orientation_rpe_time_1s_rmse_deg": _as_float(
            epa_result.get("orientation_rpe_time_1s_rmse_deg")
        ),
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
    python_bin = _resolve_cli_path(args.python_bin, repo_root)
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
