from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from epa.core.evaluation import apply_input_coverage_to_success
from epa.core.math_utils import compute_error_statistics
from epa.ov_io import load_ov_csv as _load_ov_eval_csv
from epa.ov_io import load_pose_file as _load_pose_file
from epa.ov_report import (
    fmt_num as _fmt,
    print_comparison_tables,
    source_counts as _format_source_counts,
    source_details as _format_source_details,
)
from .evaluate import (
    _compute_rpe_segments,
    _compute_time_rpe_1s,
    _compute_valid_rpe_segments,
    _compute_valid_segment_summary,
    _eval_quality_flags,
    _fmt_sr_config,
)
from .constants import _VALID_ALIGN_MODES


def _evaluate_pair_current(**kwargs):
    import epa.ov_eval_compat as compat

    return compat._evaluate_pair(**kwargs)


def _epa_eval_kwargs_current(args: argparse.Namespace) -> dict[str, object]:
    import epa.ov_eval_compat as compat

    return compat._epa_eval_kwargs(args)


def _eval_warning_prefix(quality: dict[str, object]) -> str:
    """Distinguish a soft warning from a rejected evaluation."""
    return "WARNING" if bool(quality.get("eval_reliable", False)) else "UNRELIABLE"


def run_format_converter(args: argparse.Namespace) -> int:
    src = Path(args.path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Input path not found: {src}")

    csv_files: list[Path] = []
    if src.is_file():
        if src.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got: {src}")
        csv_files = [src]
    else:
        csv_files = sorted(p for p in src.rglob("*.csv") if p.is_file())

    if len(csv_files) == 0:
        print("No .csv files found.")
        return 0

    for fpath in csv_files:
        t, pos, quat = _load_ov_eval_csv(fpath)
        out_path = fpath.with_suffix(".txt")
        if out_path.exists():
            print(f"[skip] output already exists: {out_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# timestamp(s) tx ty tz qx qy qz qw\n")
            for i in range(t.size):
                f.write(
                    f"{t[i]:.5f} "
                    f"{pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f} "
                    f"{quat[i, 0]:.6f} {quat[i, 1]:.6f} {quat[i, 2]:.6f} {quat[i, 3]:.6f}\n"
                )
        print(f"[ok] {fpath} -> {out_path}")

    return 0


def run_error_singlerun(args: argparse.Namespace) -> int:
    sr_m = float(getattr(args, "epa_success_threshold_m", 10.0))
    sr_mode = str(getattr(args, "epa_success_threshold_mode", "adaptive_knee"))
    sr_min_m = float(getattr(args, "epa_success_threshold_min_m", 5.0))
    sr_max_m = float(getattr(args, "epa_success_threshold_max_m", 30.0))
    sr_trim_pct = float(getattr(args, "epa_success_threshold_trim_percentile", 95.0))
    gate_mode = str(getattr(args, "epa_success_global_gate_mode", "fixed"))
    gate_m = float(getattr(args, "epa_success_global_gate_m", 30.0))
    gate_path_ratio = float(getattr(args, "epa_success_global_gate_path_ratio", 0.05))
    gate_min_m = float(getattr(args, "epa_success_global_gate_min_m", 2.0))
    gate_max_m = float(getattr(args, "epa_success_global_gate_max_m", 100.0))
    gate_pct = float(getattr(args, "epa_success_global_gate_percentile", 5.0))
    drift_threshold_mode = str(getattr(args, "epa_success_drift_threshold_mode", "adaptive"))
    drift_rpe_1s_m = float(getattr(args, "epa_success_drift_rpe_1s_m", 2.0))
    drift_ape_slope_mps = float(getattr(args, "epa_success_drift_ape_slope_mps", 1.0))
    drift_ape_jump_m = float(getattr(args, "epa_success_drift_ape_jump_m", 5.0))
    eval_res = _evaluate_pair_current(
        file_gt=Path(args.file_gt).expanduser(),
        file_est=Path(args.file_est).expanduser(),
        align_mode=str(args.align_mode),
        max_diff=float(args.max_diff),
        **_epa_eval_kwargs_current(args),
    )

    if eval_res["length_ratio"] > 1.1 or eval_res["length_ratio"] < 0.9:
        print(
            f"[WARN] Trajectory length ratio est/gt={eval_res['length_ratio']:.2f} "
            f"(est={eval_res['length_est']:.2f}m, gt={eval_res['length_gt']:.2f}m)"
        )
    eval_alignment = eval_res.get("eval_alignment", {})
    if isinstance(eval_alignment, dict) and "extrinsic_selection_reason" in eval_alignment:
        print(
            "Extrinsic selection = " + str(eval_alignment["extrinsic_selection_reason"])
            + f" | rotation_information_ratio = {float(eval_alignment['extrinsic_rotation_information_ratio']):.6f}"
            + f" | identity_position_rmse_m = {float(eval_alignment['extrinsic_identity_position_rmse_m']):.6f}"
            + f" | selected_position_rmse_m = {float(eval_alignment['extrinsic_selected_position_rmse_m']):.6f}"
        )
    if isinstance(eval_alignment, dict) and eval_alignment.get("sim3_scale_severe"):
        print(
            "[WARN] "
            f"{eval_alignment.get('sim3_warning')} "
            f"scale={_fmt(float(eval_alignment.get('sim3_scale', np.nan)), 6)}"
        )

    ate3_ori = eval_res["ate3_ori"]
    ate3_pos = eval_res["ate3_pos"]

    print("======================================")
    print("Absolute Trajectory Error")
    print("======================================")
    print(f"rmse_ori = {_fmt(ate3_ori['rmse'])} | rmse_pos = {_fmt(ate3_pos['rmse'])}")
    print(f"mean_ori = {_fmt(ate3_ori['mean'])} | mean_pos = {_fmt(ate3_pos['mean'])}")
    print(f"min_ori  = {_fmt(ate3_ori['min'])} | min_pos  = {_fmt(ate3_pos['min'])}")
    print(f"max_ori  = {_fmt(ate3_ori['max'])} | max_pos  = {_fmt(ate3_pos['max'])}")
    print(f"std_ori  = {_fmt(ate3_ori['std'])} | std_pos  = {_fmt(ate3_pos['std'])}")

    segments = [8.0, 16.0, 24.0, 32.0, 40.0]
    rpe = _compute_rpe_segments(
        gt_pos=np.asarray(eval_res["gt_pos"], dtype=float),
        gt_quat=np.asarray(eval_res["gt_quat"], dtype=float),
        est_pos=np.asarray(eval_res["est_pos"], dtype=float),
        est_quat=np.asarray(eval_res["est_quat"], dtype=float),
        segments_m=segments,
    )
    time_rpe = _compute_time_rpe_1s(
        gt_t=np.asarray(eval_res["gt_t"], dtype=float),
        gt_pos=np.asarray(eval_res["gt_pos"], dtype=float),
        gt_quat=np.asarray(eval_res["gt_quat"], dtype=float),
        est_pos=np.asarray(eval_res["est_pos"], dtype=float),
        est_quat=np.asarray(eval_res["est_quat"], dtype=float),
    )
    valid = _compute_valid_segment_summary(
        input_coverage=eval_res.get("input_coverage"),
        gt_t=np.asarray(eval_res["gt_t"], dtype=float),
        gt_pos=np.asarray(eval_res["gt_pos"], dtype=float),
        gt_quat=np.asarray(eval_res["gt_quat"], dtype=float),
        est_pos=np.asarray(eval_res["est_pos"], dtype=float),
        est_quat=np.asarray(eval_res["est_quat"], dtype=float),
        threshold_m=sr_m,
        threshold_mode=sr_mode,
        threshold_min_m=sr_min_m,
        threshold_max_m=sr_max_m,
        threshold_trim_percentile=sr_trim_pct,
        global_gate_mode=gate_mode,
        global_gate_m=gate_m,
        global_gate_path_ratio=gate_path_ratio,
        global_gate_min_m=gate_min_m,
        global_gate_max_m=gate_max_m,
        global_gate_percentile=gate_pct,
        drift_threshold_mode=drift_threshold_mode,
        drift_rpe_1s_m=drift_rpe_1s_m,
        drift_ape_slope_mps=drift_ape_slope_mps,
        drift_ape_jump_m=drift_ape_jump_m,
    )
    apply_input_coverage_to_success(
        valid["success"], eval_res.get("input_coverage"), set_primary=True
    )

    print("======================================")
    print("Relative Pose Error")
    print("======================================")
    for seg in segments:
        blk = rpe[seg]
        ori_stats = blk["ori_stats"]
        pos_stats = blk["pos_stats"]
        n = int(blk["pair_count"])
        print(
            f"seg {int(seg)} - median_ori = {_fmt(ori_stats['median'])} "
            f"| median_pos = {_fmt(pos_stats['median'])} ({n} samples)"
        )

    time_ori_stats = time_rpe["ori_stats"]
    time_pos_stats = time_rpe["pos_stats"]
    print(
        f"1s time - rmse_ori = {_fmt(time_ori_stats['rmse'])} "
        f"| rmse_pos = {_fmt(time_pos_stats['rmse'])} ({int(time_rpe['pair_count'])} samples)"
    )
    success = valid["success"]
    if isinstance(eval_alignment, dict) and str(eval_alignment.get("align_mode", "")) == "sim3":
        success["sim3_sr_distance_raw"] = success.get("success_rate_distance")
        success["sim3_sr_time_raw"] = success.get("success_rate_time")
        success["sim3_sr_reliable"] = bool(eval_alignment.get("sim3_reliable", True))
        if eval_alignment.get("sim3_scale_severe"):
            success["sim3_scale_warning"] = str(eval_alignment.get("sim3_warning", ""))
    quality = _eval_quality_flags(eval_res, success, time_rpe)
    print(
        _fmt_sr_config(
            valid,
            np.asarray(eval_res["gt_t"], dtype=float),
            np.asarray(eval_res["gt_pos"], dtype=float),
        )
    )
    print(
        f"Coverage - path = {_fmt(float(success['path_coverage_ratio']) * 100.0, 2)}% "
        f"| time = {_fmt(float(success['temporal_coverage_ratio']) * 100.0, 2)}% "
        f"| status = {success['input_coverage_status']}"
    )
    print(
        f"SR local - distance = "
        f"{_fmt(float(success['local_success_rate_distance']) * 100.0, 2)}% "
        f"| time = {_fmt(float(success['local_success_rate_time']) * 100.0, 2)}% "
        f"| valid_dist = {_fmt(success['valid_distance_m'])}/{_fmt(success['local_total_distance_m'])}m"
    )
    print(
        f"SR complete - distance = "
        f"{_fmt(float(success['complete_success_rate_distance']) * 100.0, 2)}% "
        f"| time = {_fmt(float(success['complete_success_rate_time']) * 100.0, 2)}% "
        f"| valid_dist = {_fmt(success['valid_distance_m'])}/{_fmt(success['complete_total_distance_m'])}m"
    )
    if isinstance(eval_alignment, dict) and str(eval_alignment.get("align_mode", "")) == "sim3":
        reliable = bool(eval_alignment.get("sim3_reliable", True))
        print(
            f"Sim3 scale = {_fmt(float(eval_alignment.get('sim3_scale', np.nan)), 6)} "
            f"| reliable = {str(reliable).lower()}"
        )
        fallback_used = bool(eval_alignment.get("sim3_robust_fallback_used", False))
        if fallback_used:
            reason = str(eval_alignment.get("sim3_robust_fallback_reason", "") or "")
            print(f"Sim3 fallback used = true | reason = {reason}")
    print(f"Eval reliable = {str(bool(quality['eval_reliable'])).lower()}")
    if quality["eval_warning"]:
        print(f"[{_eval_warning_prefix(quality)}] {quality['eval_warning']}")
    print("======================================")
    print(f"Aligned pairs: {int(eval_res['matched'])}")
    print(f"Eval source = {str(eval_res.get('eval_source', 'unknown'))}")
    print(
        "EPA_COMPAT_RESULT_JSON "
        + json.dumps(
            {
                "eval_source": str(eval_res.get("eval_source", "unknown")),
                "matched": int(eval_res["matched"]),
                "ape_rmse_m": float(ate3_pos["rmse"]),
                **{
                    key: value for key, value in eval_alignment.items()
                    if key.startswith("extrinsic_")
                },
                "rpe_time_1s_rmse_m": float(time_pos_stats["rmse"]),
                "sr_distance_pct": float(success["success_rate_distance"]) * 100.0,
                "sr_time_pct": float(success["success_rate_time"]) * 100.0,
                "sr_local_distance_pct": float(success["local_success_rate_distance"]) * 100.0,
                "sr_local_time_pct": float(success["local_success_rate_time"]) * 100.0,
                "path_coverage_pct": float(success["path_coverage_ratio"]) * 100.0,
                "temporal_coverage_pct": float(success["temporal_coverage_ratio"]) * 100.0,
                "input_coverage_status": str(success["input_coverage_status"]),
                "eval_reliable": bool(quality["eval_reliable"]),
                "sim3_fallback_used": bool(eval_alignment.get("sim3_robust_fallback_used", False))
                if isinstance(eval_alignment, dict)
                else False,
            },
            sort_keys=True,
        )
    )
    if args.plot:
        print(
            "[info] --plot is reserved in EPA compatibility mode. Use `epa_ape`/`epa_rpe` for full plotting."
        )
    return 0


def run_error_dataset(args: argparse.Namespace) -> int:
    file_gt = Path(args.file_gt).expanduser()
    alg_root = Path(args.folder_algorithms).expanduser()
    dataset_name = file_gt.stem
    sr_m = float(getattr(args, "epa_success_threshold_m", 10.0))
    sr_mode = str(getattr(args, "epa_success_threshold_mode", "adaptive_knee"))
    sr_min_m = float(getattr(args, "epa_success_threshold_min_m", 5.0))
    sr_max_m = float(getattr(args, "epa_success_threshold_max_m", 30.0))
    sr_trim_pct = float(getattr(args, "epa_success_threshold_trim_percentile", 95.0))
    gate_mode = str(getattr(args, "epa_success_global_gate_mode", "fixed"))
    gate_m = float(getattr(args, "epa_success_global_gate_m", 30.0))
    gate_path_ratio = float(getattr(args, "epa_success_global_gate_path_ratio", 0.05))
    gate_min_m = float(getattr(args, "epa_success_global_gate_min_m", 2.0))
    gate_max_m = float(getattr(args, "epa_success_global_gate_max_m", 100.0))
    gate_pct = float(getattr(args, "epa_success_global_gate_percentile", 5.0))
    drift_threshold_mode = str(getattr(args, "epa_success_drift_threshold_mode", "adaptive"))
    drift_rpe_1s_m = float(getattr(args, "epa_success_drift_rpe_1s_m", 2.0))
    drift_ape_slope_mps = float(getattr(args, "epa_success_drift_ape_slope_mps", 1.0))
    drift_ape_jump_m = float(getattr(args, "epa_success_drift_ape_jump_m", 5.0))
    fail_on_skipped = bool(getattr(args, "fail_on_skipped", False))
    batch_had_problem = False

    algo_dirs = sorted([p for p in alg_root.iterdir() if p.is_dir()])
    if len(algo_dirs) == 0:
        raise ValueError(f"No algorithm folders found in: {alg_root}")

    t_gt, p_gt, _ = _load_pose_file(file_gt)
    gt_length = (
        float(np.sum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))) if p_gt.shape[0] > 1 else 0.0
    )
    print(f"[COMP]: {int(t_gt.size)} poses in {dataset_name} => length of {gt_length:.2f} meters")

    segments = [7.0, 14.0, 21.0, 28.0, 35.0]
    print("======================================")

    for algo_dir in algo_dirs:
        run_dir = algo_dir / dataset_name
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"[COMP]: {algo_dir.name} has no runs for dataset {dataset_name}")
            batch_had_problem = True
            continue

        run_files = sorted(
            [p for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".csv"}]
        )
        if len(run_files) == 0:
            print(f"[COMP]: {algo_dir.name} has empty run folder for {dataset_name}")
            batch_had_problem = True
            continue

        ate_ori_rmse: list[float] = []
        ate_pos_rmse: list[float] = []
        ate2_ori_rmse: list[float] = []
        ate2_pos_rmse: list[float] = []
        rpe_ori_vals: dict[float, list[float]] = {s: [] for s in segments}
        rpe_pos_vals: dict[float, list[float]] = {s: [] for s in segments}
        time_ori_vals: list[float] = []
        time_pos_vals: list[float] = []
        sr_dist_vals: list[float] = []
        sr_time_vals: list[float] = []
        unreliable_details: list[str] = []
        failed_details: list[str] = []
        fallback_details: list[str] = []

        for run_file in run_files:
            try:
                ev = _evaluate_pair_current(
                    file_gt=file_gt,
                    file_est=run_file,
                    align_mode=str(args.align_mode),
                    max_diff=float(args.max_diff),
                    **_epa_eval_kwargs_current(args),
                )
            except Exception as exc:
                print(f"\t[warn] skipping {run_file.name}: {exc}")
                failed_details.append(f"{run_file.name}:failed")
                batch_had_problem = True
                continue
            ate_ori_rmse.append(float(ev["ate3_ori"]["rmse"]))
            ate_pos_rmse.append(float(ev["ate3_pos"]["rmse"]))
            ate2_ori_rmse.append(float(ev["ate2_ori"]["rmse"]))
            ate2_pos_rmse.append(float(ev["ate2_pos"]["rmse"]))
            eval_alignment = ev.get("eval_alignment", {})
            if isinstance(eval_alignment, dict):
                if bool(eval_alignment.get("sim3_robust_fallback_used", False)):
                    fallback_details.append(f"{run_file.name}:fallback")

            rpe = _compute_rpe_segments(
                gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                est_pos=np.asarray(ev["est_pos"], dtype=float),
                est_quat=np.asarray(ev["est_quat"], dtype=float),
                segments_m=segments,
            )
            for seg in segments:
                rpe_ori_vals[seg].extend(np.asarray(rpe[seg]["ori_values"], dtype=float).tolist())
                rpe_pos_vals[seg].extend(np.asarray(rpe[seg]["pos_values"], dtype=float).tolist())
            time_rpe = _compute_time_rpe_1s(
                gt_t=np.asarray(ev["gt_t"], dtype=float),
                gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                est_pos=np.asarray(ev["est_pos"], dtype=float),
                est_quat=np.asarray(ev["est_quat"], dtype=float),
            )
            time_ori_vals.extend(np.asarray(time_rpe["ori_values"], dtype=float).tolist())
            time_pos_vals.extend(np.asarray(time_rpe["pos_values"], dtype=float).tolist())
            valid = _compute_valid_segment_summary(
                input_coverage=ev.get("input_coverage"),
                gt_t=np.asarray(ev["gt_t"], dtype=float),
                gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                est_pos=np.asarray(ev["est_pos"], dtype=float),
                est_quat=np.asarray(ev["est_quat"], dtype=float),
                threshold_m=sr_m,
                threshold_mode=sr_mode,
                threshold_min_m=sr_min_m,
                threshold_max_m=sr_max_m,
                threshold_trim_percentile=sr_trim_pct,
                global_gate_mode=gate_mode,
                global_gate_m=gate_m,
                global_gate_path_ratio=gate_path_ratio,
                global_gate_min_m=gate_min_m,
                global_gate_max_m=gate_max_m,
                global_gate_percentile=gate_pct,
                drift_threshold_mode=drift_threshold_mode,
                drift_rpe_1s_m=drift_rpe_1s_m,
                drift_ape_slope_mps=drift_ape_slope_mps,
                drift_ape_jump_m=drift_ape_jump_m,
            )
            apply_input_coverage_to_success(
                valid["success"], ev.get("input_coverage"), set_primary=True
            )
            quality = _eval_quality_flags(ev, valid["success"], time_rpe)
            if not bool(quality["eval_reliable"]):
                unreliable_details.append(f"{run_file.name}:{quality['eval_warning']}")
            sr_dist_vals.append(float(valid["success"]["success_rate_distance"]))
            sr_time_vals.append(float(valid["success"]["success_rate_time"]))

        valid_runs = len(ate_ori_rmse)
        if valid_runs == 0:
            print(f"[COMP]: processing {algo_dir.name} algorithm")
            print(
                f"\t[warn] no valid runs for {algo_dir.name}/{dataset_name}; skipping dataset metrics"
            )
            batch_had_problem = True
            if failed_details:
                print(f"\tfailed_runs: {_format_source_details(failed_details)}")
            print("======================================")
            continue

        ate_ori = compute_error_statistics(np.asarray(ate_ori_rmse, dtype=float))
        ate_pos = compute_error_statistics(np.asarray(ate_pos_rmse, dtype=float))
        ate2_ori = compute_error_statistics(np.asarray(ate2_ori_rmse, dtype=float))
        ate2_pos = compute_error_statistics(np.asarray(ate2_pos_rmse, dtype=float))

        print(f"[COMP]: processing {algo_dir.name} algorithm")
        print(
            f"\tATE: mean_ori = {_fmt(ate_ori['mean'])} | mean_pos = {_fmt(ate_pos['mean'])} "
            f"({valid_runs}/{len(run_files)} valid runs)"
        )
        print(f"\tATE: std_ori  = {_fmt(ate_ori['std'], 5)} | std_pos  = {_fmt(ate_pos['std'], 5)}")
        print(
            f"\tATE 2D: mean_ori = {_fmt(ate2_ori['mean'])} | mean_pos = {_fmt(ate2_pos['mean'])} "
            f"({valid_runs}/{len(run_files)} valid runs)"
        )
        print(
            f"\tATE 2D: std_ori  = {_fmt(ate2_ori['std'], 5)} | std_pos  = {_fmt(ate2_pos['std'], 5)}"
        )
        if failed_details:
            print(f"\tfailed_runs: {_format_source_details(failed_details)}")
        if fallback_details:
            print(f"\tsim3_fallback: {_format_source_details(fallback_details)}")
        if unreliable_details:
            print(f"\tunreliable_runs: {_format_source_details(unreliable_details)}")

        for seg in segments:
            o_stats = compute_error_statistics(np.asarray(rpe_ori_vals[seg], dtype=float))
            p_stats = compute_error_statistics(np.asarray(rpe_pos_vals[seg], dtype=float))
            n = len(rpe_pos_vals[seg])
            print(
                f"\tRPE: seg {int(seg)} - mean_ori = {_fmt(o_stats['mean'])} "
                f"| mean_pos = {_fmt(p_stats['mean'])} ({n} samples)"
            )

        time_ori_stats = compute_error_statistics(np.asarray(time_ori_vals, dtype=float))
        time_pos_stats = compute_error_statistics(np.asarray(time_pos_vals, dtype=float))
        print(
            f"\tRPE time 1s - mean_ori = {_fmt(time_ori_stats['mean'])} "
            f"| mean_pos = {_fmt(time_pos_stats['mean'])} ({len(time_pos_vals)} samples)"
        )
        sr_dist_stats = compute_error_statistics(np.asarray(sr_dist_vals, dtype=float))
        sr_time_stats = compute_error_statistics(np.asarray(sr_time_vals, dtype=float))
        print(
            f"\tSR - distance = {_fmt(sr_dist_stats['mean'] * 100.0, 2)}% "
            f"| time = {_fmt(sr_time_stats['mean'] * 100.0, 2)}%"
        )
        print("\tNEES: n/a in EPA compatibility mode")
        print("======================================")

    if args.plot:
        print("[info] --plot is reserved in EPA compatibility mode.")
    return 1 if fail_on_skipped and batch_had_problem else 0


def run_error_comparison(args: argparse.Namespace) -> int:
    gt_root = Path(args.folder_groundtruth).expanduser()
    alg_root = Path(args.folder_algorithms).expanduser()
    sr_m = float(getattr(args, "epa_success_threshold_m", 10.0))
    sr_mode = str(getattr(args, "epa_success_threshold_mode", "adaptive_knee"))
    sr_min_m = float(getattr(args, "epa_success_threshold_min_m", 5.0))
    sr_max_m = float(getattr(args, "epa_success_threshold_max_m", 30.0))
    sr_trim_pct = float(getattr(args, "epa_success_threshold_trim_percentile", 95.0))
    gate_mode = str(getattr(args, "epa_success_global_gate_mode", "fixed"))
    gate_m = float(getattr(args, "epa_success_global_gate_m", 30.0))
    gate_path_ratio = float(getattr(args, "epa_success_global_gate_path_ratio", 0.05))
    gate_min_m = float(getattr(args, "epa_success_global_gate_min_m", 2.0))
    gate_max_m = float(getattr(args, "epa_success_global_gate_max_m", 100.0))
    gate_pct = float(getattr(args, "epa_success_global_gate_percentile", 5.0))
    drift_threshold_mode = str(getattr(args, "epa_success_drift_threshold_mode", "adaptive"))
    drift_rpe_1s_m = float(getattr(args, "epa_success_drift_rpe_1s_m", 2.0))
    drift_ape_slope_mps = float(getattr(args, "epa_success_drift_ape_slope_mps", 1.0))
    drift_ape_jump_m = float(getattr(args, "epa_success_drift_ape_jump_m", 5.0))
    fail_on_skipped = bool(getattr(args, "fail_on_skipped", False))
    batch_had_problem = False

    gt_files = sorted([p for p in gt_root.rglob("*.txt") if p.is_file()])
    algo_dirs = sorted([p for p in alg_root.iterdir() if p.is_dir()])
    if len(gt_files) == 0:
        raise ValueError(f"No groundtruth .txt files found in: {gt_root}")
    if len(algo_dirs) == 0:
        raise ValueError(f"No algorithm folders found in: {alg_root}")

    for gt in gt_files:
        t, p, _ = _load_pose_file(gt)
        length = (
            float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if p.shape[0] > 1 else 0.0
        )
        print(f"[COMP]: {int(t.size)} poses in {gt.name} => length of {length:.2f} meters")

    segments = [10.0, 20.0, 50.0, 100.0]
    ate_table: dict[str, dict[str, tuple[float, float]]] = {a.name: {} for a in algo_dirs}
    time_rpe_table: dict[str, dict[str, tuple[float, float]]] = {a.name: {} for a in algo_dirs}
    valid_ate_table: dict[str, dict[str, tuple[float, float]]] = {a.name: {} for a in algo_dirs}
    valid_time_rpe_table: dict[str, dict[str, tuple[float, float]]] = {
        a.name: {} for a in algo_dirs
    }
    success_table: dict[str, dict[str, tuple[float, float]]] = {a.name: {} for a in algo_dirs}
    valid_rpe_table: dict[str, dict[float, dict[str, tuple[float, float]]]] = {
        a.name: {s: {} for s in segments} for a in algo_dirs
    }
    source_total: dict[str, int] = {}
    source_details: list[str] = []
    failed_total: list[str] = []
    unreliable_total: list[str] = []

    print("======================================")
    for algo_dir in algo_dirs:
        print(f"[COMP]: processing {algo_dir.name} algorithm")
        dataset_dirs = {p.name: p for p in algo_dir.iterdir() if p.is_dir()}

        for gt in gt_files:
            ds = gt.stem
            if ds not in dataset_dirs:
                print(f"[COMP]: {algo_dir.name} has no runs for {ds}")
                batch_had_problem = True
                continue

            run_files = sorted(
                [
                    p
                    for p in dataset_dirs[ds].iterdir()
                    if p.is_file() and p.suffix.lower() in {".txt", ".csv"}
                ]
            )
            if len(run_files) == 0:
                batch_had_problem = True
                continue

            print(f"[COMP]: processing {algo_dir.name} algorithm => {ds} dataset")
            ate_ori_rmse: list[float] = []
            ate_pos_rmse: list[float] = []
            source_ds: dict[str, int] = {}
            source_ds_details: list[str] = []
            failed_ds: list[str] = []

            ds_rpe_ori: dict[float, list[float]] = {s: [] for s in segments}
            ds_rpe_pos: dict[float, list[float]] = {s: [] for s in segments}
            ds_time_ori: list[float] = []
            ds_time_pos: list[float] = []
            ds_sr_dist: list[float] = []
            ds_sr_time: list[float] = []
            ds_valid_ate_ori: list[float] = []
            ds_valid_ate_pos: list[float] = []
            ds_valid_time_ori: list[float] = []
            ds_valid_time_pos: list[float] = []
            ds_valid_rpe_ori: dict[float, list[float]] = {s: [] for s in segments}
            ds_valid_rpe_pos: dict[float, list[float]] = {s: [] for s in segments}
            ds_unreliable: list[str] = []
            ds_fallback_details: list[str] = []

            for run_file in run_files:
                try:
                    ev = _evaluate_pair_current(
                        file_gt=gt,
                        file_est=run_file,
                        align_mode=str(args.align_mode),
                        max_diff=float(args.max_diff),
                        **_epa_eval_kwargs_current(args),
                    )
                except Exception as exc:
                    print(f"\t[warn] skipping {run_file.name}: {exc}")
                    source_ds["failed"] = source_ds.get("failed", 0) + 1
                    source_total["failed"] = source_total.get("failed", 0) + 1
                    failed_ds.append(f"{run_file.name}:failed")
                    failed_total.append(f"{algo_dir.name}/{ds}/{run_file.name}:failed")
                    batch_had_problem = True
                    continue
                ate_ori_rmse.append(float(ev["ate3_ori"]["rmse"]))
                ate_pos_rmse.append(float(ev["ate3_pos"]["rmse"]))
                source = str(ev.get("eval_source", "unknown"))
                source_ds[source] = source_ds.get(source, 0) + 1
                source_total[source] = source_total.get(source, 0) + 1
                if source not in {"epa_step3", "epa_eval_align", "epa_posyaw"}:
                    source_ds_details.append(f"{run_file.name}:{source}")
                    source_details.append(f"{algo_dir.name}/{ds}/{run_file.name}:{source}")
                eval_alignment = ev.get("eval_alignment", {})
                if isinstance(eval_alignment, dict):
                    if bool(eval_alignment.get("sim3_robust_fallback_used", False)):
                        ds_fallback_details.append(f"{run_file.name}:fallback")

                rpe = _compute_rpe_segments(
                    gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                    gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                    est_pos=np.asarray(ev["est_pos"], dtype=float),
                    est_quat=np.asarray(ev["est_quat"], dtype=float),
                    segments_m=segments,
                )
                for seg in segments:
                    ori_vals = np.asarray(rpe[seg]["ori_values"], dtype=float).tolist()
                    pos_vals = np.asarray(rpe[seg]["pos_values"], dtype=float).tolist()
                    ds_rpe_ori[seg].extend(ori_vals)
                    ds_rpe_pos[seg].extend(pos_vals)
                time_rpe = _compute_time_rpe_1s(
                    gt_t=np.asarray(ev["gt_t"], dtype=float),
                    gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                    gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                    est_pos=np.asarray(ev["est_pos"], dtype=float),
                    est_quat=np.asarray(ev["est_quat"], dtype=float),
                )
                time_ori_vals = np.asarray(time_rpe["ori_values"], dtype=float).tolist()
                time_pos_vals = np.asarray(time_rpe["pos_values"], dtype=float).tolist()
                ds_time_ori.extend(time_ori_vals)
                ds_time_pos.extend(time_pos_vals)
                valid = _compute_valid_segment_summary(
                    input_coverage=ev.get("input_coverage"),
                    gt_t=np.asarray(ev["gt_t"], dtype=float),
                    gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                    gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                    est_pos=np.asarray(ev["est_pos"], dtype=float),
                    est_quat=np.asarray(ev["est_quat"], dtype=float),
                    threshold_m=sr_m,
                    threshold_mode=sr_mode,
                    threshold_min_m=sr_min_m,
                    threshold_max_m=sr_max_m,
                    threshold_trim_percentile=sr_trim_pct,
                    global_gate_mode=gate_mode,
                    global_gate_m=gate_m,
                    global_gate_path_ratio=gate_path_ratio,
                    global_gate_min_m=gate_min_m,
                    global_gate_max_m=gate_max_m,
                    global_gate_percentile=gate_pct,
                    drift_threshold_mode=drift_threshold_mode,
                    drift_rpe_1s_m=drift_rpe_1s_m,
                    drift_ape_slope_mps=drift_ape_slope_mps,
                    drift_ape_jump_m=drift_ape_jump_m,
                )
                apply_input_coverage_to_success(
                    valid["success"], ev.get("input_coverage"), set_primary=True
                )
                quality = _eval_quality_flags(ev, valid["success"], time_rpe)
                # Reliability warnings do not suppress SR or drift-valid metrics.
                ds_sr_dist.append(float(valid["success"]["success_rate_distance"]))
                ds_sr_time.append(float(valid["success"]["success_rate_time"]))
                if not bool(quality["eval_reliable"]):
                    detail = f"{run_file.name}:{quality['eval_warning']}"
                    ds_unreliable.append(detail)
                    unreliable_total.append(f"{algo_dir.name}/{ds}/{detail}")
                ds_valid_ate_ori.extend(
                    np.asarray(
                        valid["ape"]["_error_arrays"]["rotation_angle_deg"], dtype=float
                    ).tolist()
                )
                ds_valid_ate_pos.extend(
                    np.asarray(
                        valid["ape"]["_error_arrays"]["translation_part"], dtype=float
                    ).tolist()
                )
                ds_valid_time_ori.extend(
                    np.asarray(
                        valid["rpe_time_1s"]["_error_arrays"]["rotation_angle_deg"], dtype=float
                    ).tolist()
                )
                ds_valid_time_pos.extend(
                    np.asarray(
                        valid["rpe_time_1s"]["_error_arrays"]["translation_part"], dtype=float
                    ).tolist()
                )
                valid_rpe = _compute_valid_rpe_segments(
                    gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                    gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                    est_pos=np.asarray(ev["est_pos"], dtype=float),
                    est_quat=np.asarray(ev["est_quat"], dtype=float),
                    valid_segment_mask=np.asarray(
                        valid["success"]["valid_segment_mask"], dtype=bool
                    ),
                    segments_m=segments,
                )
                for seg in segments:
                    valid_ori_vals = np.asarray(valid_rpe[seg]["ori_values"], dtype=float).tolist()
                    valid_pos_vals = np.asarray(valid_rpe[seg]["pos_values"], dtype=float).tolist()
                    ds_valid_rpe_ori[seg].extend(valid_ori_vals)
                    ds_valid_rpe_pos[seg].extend(valid_pos_vals)

            if len(ate_ori_rmse) == 0:
                print(f"\t[warn] no valid runs for {algo_dir.name}/{ds}; skipping dataset metrics")
                batch_had_problem = True
                if failed_ds:
                    print(f"\tfailed_runs: {_format_source_details(failed_ds)}")
                continue

            ate_ori_stats = compute_error_statistics(np.asarray(ate_ori_rmse, dtype=float))
            ate_pos_stats = compute_error_statistics(np.asarray(ate_pos_rmse, dtype=float))
            ate_table[algo_dir.name][ds] = (
                float(ate_ori_stats["mean"]),
                float(ate_pos_stats["mean"]),
            )
            time_ori_stats = compute_error_statistics(np.asarray(ds_time_ori, dtype=float))
            time_pos_stats = compute_error_statistics(np.asarray(ds_time_pos, dtype=float))
            time_rpe_table[algo_dir.name][ds] = (
                float(time_ori_stats["mean"]),
                float(time_pos_stats["mean"]),
            )
            valid_ate_ori_stats = compute_error_statistics(
                np.asarray(ds_valid_ate_ori, dtype=float)
            )
            valid_ate_pos_stats = compute_error_statistics(
                np.asarray(ds_valid_ate_pos, dtype=float)
            )
            valid_ate_table[algo_dir.name][ds] = (
                float(valid_ate_ori_stats["rmse"]),
                float(valid_ate_pos_stats["rmse"]),
            )
            valid_time_ori_stats = compute_error_statistics(
                np.asarray(ds_valid_time_ori, dtype=float)
            )
            valid_time_pos_stats = compute_error_statistics(
                np.asarray(ds_valid_time_pos, dtype=float)
            )
            valid_time_rpe_table[algo_dir.name][ds] = (
                float(valid_time_ori_stats["mean"]),
                float(valid_time_pos_stats["mean"]),
            )
            for seg in segments:
                valid_rpe_ori_stats = compute_error_statistics(
                    np.asarray(ds_valid_rpe_ori[seg], dtype=float)
                )
                valid_rpe_pos_stats = compute_error_statistics(
                    np.asarray(ds_valid_rpe_pos[seg], dtype=float)
                )
                valid_rpe_table[algo_dir.name][seg][ds] = (
                    float(valid_rpe_ori_stats["rmse"]),
                    float(valid_rpe_pos_stats["rmse"]),
                )
            sr_dist_stats = compute_error_statistics(np.asarray(ds_sr_dist, dtype=float))
            sr_time_stats = compute_error_statistics(np.asarray(ds_sr_time, dtype=float))
            success_table[algo_dir.name][ds] = (
                float(sr_dist_stats["mean"]),
                float(sr_time_stats["mean"]),
            )

            print(
                f"\tATE: mean_ori = {_fmt(ate_ori_stats['mean'])} "
                f"| mean_pos = {_fmt(ate_pos_stats['mean'])} "
                f"({len(ate_ori_rmse)}/{len(run_files)} valid runs)"
            )
            print(f"\teval_source: {_format_source_counts(source_ds)}")
            if source_ds_details:
                print(f"\tnon_epa_runs: {_format_source_details(source_ds_details)}")
            if failed_ds:
                print(f"\tfailed_runs: {_format_source_details(failed_ds)}")
            if ds_fallback_details:
                print(f"\tsim3_fallback: {_format_source_details(ds_fallback_details)}")
            if ds_unreliable:
                print(f"\tunreliable_runs: {_format_source_details(ds_unreliable)}")
            for seg in segments:
                o_stats = compute_error_statistics(np.asarray(ds_rpe_ori[seg], dtype=float))
                p_stats = compute_error_statistics(np.asarray(ds_rpe_pos[seg], dtype=float))
                print(
                    f"\tRPE: seg {int(seg)} - median_ori = {_fmt(o_stats['median'], 4)} "
                    f"| median_pos = {_fmt(p_stats['median'], 4)} ({len(ds_rpe_pos[seg])} samples)"
                )
            print(
                f"\tRPE time 1s - mean_ori = {_fmt(time_ori_stats['mean'])} "
                f"| mean_pos = {_fmt(time_pos_stats['mean'])} ({len(ds_time_pos)} samples)"
            )
            print(
                "\t"
                + _fmt_sr_config(
                    valid,
                    np.asarray(ev["gt_t"], dtype=float),
                    np.asarray(ev["gt_pos"], dtype=float),
                )
            )
            print(
                f"\tSR - distance = {_fmt(sr_dist_stats['mean'] * 100.0, 2)}% "
                f"| time = {_fmt(sr_time_stats['mean'] * 100.0, 2)}%"
            )

    print("============================================")
    print(f"TOOL SOURCE: {_format_source_counts(source_total)}")
    if source_details:
        print(f"EVAL SOURCE UNKNOWN RUNS: {_format_source_details(source_details)}")
    if failed_total:
        print(f"FAILED RUNS: {_format_source_details(failed_total)}")
    if unreliable_total:
        print(f"UNRELIABLE RUNS: {_format_source_details(unreliable_total)}")
    print("============================================")
    print_comparison_tables(
        gt_files=gt_files,
        algo_names=[algo.name for algo in algo_dirs],
        ate_table=ate_table,
        time_rpe_table=time_rpe_table,
        success_table=success_table,
        valid_ate_table=valid_ate_table,
        valid_rpe_table=valid_rpe_table,
        valid_time_rpe_table=valid_time_rpe_table,
        segments=segments,
    )
    return 1 if fail_on_skipped and batch_had_problem else 0


def run_plot_trajectories(args: argparse.Namespace) -> int:
    # Plotting imports matplotlib and the complete trajectory CLI.  Keep that
    # startup cost off metric-only compatibility commands.
    from epa.traj_tool import build_parser as build_traj_parser
    from epa.traj_tool import run as run_traj

    align_mode = str(args.align_mode).lower()
    if align_mode not in _VALID_ALIGN_MODES:
        raise ValueError(f"Invalid align_mode '{args.align_mode}'")

    traj_argv: list[str] = [
        "--format",
        "tum",
        "--sync",
        "--sync-max-diff",
        str(float(args.max_diff)),
        "--ref",
        "1",
        "--plot",
    ]

    if align_mode in {"se3", "se3r", "epa_se3", "epa_se3r", "epa_se3_eval", "rotation_first_se3"}:
        traj_argv.extend(["--eval-align", "se3"])
    elif align_mode in {"se3-original", "se3_original", "se3-orginal", "se3_orginal", "position_first_se3", "umeyama_se3"}:
        traj_argv.extend(["--eval-align", "se3-original"])
    elif align_mode in {
        "se3single",
        "posyaw",
        "posyawsingle",
        "sim3",
    }:
        traj_argv.append("--align")
    if align_mode == "sim3":
        traj_argv.append("--correct-scale")

    traj_argv.append(str(args.file_gt))
    traj_argv.extend([str(x) for x in args.est_files])

    parser = build_traj_parser()
    parsed = parser.parse_args(traj_argv)
    return int(run_traj(parsed))
