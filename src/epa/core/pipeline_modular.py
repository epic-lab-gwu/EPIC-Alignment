from datetime import datetime
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from .calibration import (
    solve_world_alignment,
)
from .evaluation import (
    compute_ape_evo_style,
    compute_rpe_evo_style,
    normalize_pose_relation,
    print_metric_block,
)
from .io_utils import (
    compact_metrics_payload,
    load_estimation_trajectory,
    load_reference_trajectory,
    make_output_dir,
    save_metrics,
    write_result_bundle,
    write_run_reports,
)
from .math_utils import normalize_quat_array, rmse
from .steps import (
    _associate_gt_est,
    _compute_step2_residual_metrics,
    _compute_trajectory_metrics,
    _downsample_by_max_hz,
    _map_est_to_ref_nearest,
    _match_nearest_timestamps,
    _offset_match_diagnostics,
    _prepare_solve_eval_trajectories,
    _run_time_alignment,
    _search_direct_offset_from_matched_pairs,
    _select_gt_overlap_window,
    _solve_step2_step3,
)
from .outputs import (
    _generate_and_cleanup_metric_plots,
    _plot_alignment_map,
    _plot_piecewise_diagnostics,
    _plot_stage_alignment_maps,
    _plot_step1_outputs,
    _resolve_result_bundle_path,
    _set_axes_equal_3d,
    _write_outputs,
)
from .diagnostics import (
    _build_user_alert,
    _compute_alignment_quality,
    _compute_piecewise_alignment,
    _compute_piecewise_diagnostics,
    _compute_rigid_alignability,
    _segment_blend_weights,
    _segment_heading_angles_deg,
    _segment_ranges_by_time,
    _umeyama_transform,
)
from ..metric_cli_common import (
    align_for_eval,
    cum_distance,
    project_to_plane,
)
from ..viz.rerun_viz import log_alignment_to_rerun
from ..viz.metric_plots import generate_ape_stage_raw_plot, generate_metric_plots


def _apply_time_window(tvals, pos, quat, t_start=None, t_end=None):
    if t_start is None and t_end is None:
        return tvals, pos, quat
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    t_rel = tvals - tvals[0]
    mask = np.ones(tvals.shape[0], dtype=bool)
    if t_start is not None:
        mask &= t_rel >= float(t_start)
    if t_end is not None:
        mask &= t_rel <= float(t_end)
    if np.sum(mask) < 2:
        raise ValueError("Time window filtering kept fewer than 2 trajectory samples.")
    return tvals[mask], np.asarray(pos, dtype=float)[mask], np.asarray(quat, dtype=float)[mask]


def _finalize_step1_failure(
    *,
    run_dir: Path,
    plots_dir: Path,
    gt_path: Path,
    est_path: Path,
    gt_format: str,
    est_format: str,
    gt_topic: str,
    est_topic: str,
    args,
    time_metrics: dict,
    lag_times: np.ndarray,
    corr: np.ndarray,
    selected_offset_s: float,
    pos_gt: np.ndarray,
    pos_est: np.ndarray,
    t_gt: np.ndarray,
    t_est: np.ndarray,
    failure_reason: str,
) -> None:
    fig_corr, ax_corr = plt.subplots(figsize=(12, 4))
    ax_corr.set_title("Step 1: Cross-Correlation vs Lag")
    ax_corr.plot(lag_times, corr, color="purple", linewidth=1.2)
    ax_corr.axvline(
        selected_offset_s,
        color="red",
        linestyle="--",
        label=f"Candidate offset: {selected_offset_s:.4f}s",
    )
    ax_corr.set_xlabel("Lag (s)")
    ax_corr.set_ylabel("Correlation")
    ax_corr.grid(True, linestyle=":", alpha=0.5)
    ax_corr.legend(loc="upper right")
    fig_corr.tight_layout()
    fig_corr_path = plots_dir / "step1_cross_correlation.png"
    fig_corr.savefig(fig_corr_path, dpi=200, bbox_inches="tight")

    t_est_sync = np.asarray(t_est, dtype=float) - float(selected_offset_s)
    interp_p = interp1d(t_est_sync, pos_est, axis=0, fill_value="extrapolate")
    pr_sync = interp_p(np.asarray(t_gt, dtype=float))
    raw_err = np.linalg.norm(pr_sync - np.asarray(pos_gt, dtype=float), axis=1)
    raw_rmse = float(np.sqrt(np.mean(raw_err**2)))

    fig_raw = plt.figure(figsize=(8.8, 6.6))
    ax_raw = fig_raw.add_subplot(111, projection="3d")
    _plot_alignment_map(
        fig_raw,
        ax_raw,
        pos_ref=np.asarray(pos_gt, dtype=float),
        pos_est=np.asarray(pr_sync, dtype=float),
        errors_m=np.asarray(raw_err, dtype=float),
        title=f"Raw Diagnostic (Step 1 Failed)\nrmse={raw_rmse:.6f} m",
    )
    fig_raw.tight_layout()
    fig_raw_path = plots_dir / "step1_raw_trajectory_diagnostic_3d.png"
    fig_raw.savefig(fig_raw_path, dpi=220, bbox_inches="tight")

    metrics_payload = {
        "time_alignment": dict(time_metrics),
        "trajectory": {
            "ate_rmse_raw_m": float(raw_rmse),
            "ate_p95_raw_m": float(np.percentile(raw_err, 95)),
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": "real",
            "gt_path": str(gt_path),
            "est_path": str(est_path),
            "gt_format": str(gt_format),
            "est_format": str(est_format),
            "gt_topic": str(gt_topic),
            "est_topic": str(est_topic),
            "t_offset": float(getattr(args, "t_offset", 0.0)),
            "t_max_diff": float(getattr(args, "t_max_diff", 0.02)),
            "step1_failed": True,
            "step1_failure_reason": str(failure_reason),
            "user_alert_level": "critical",
            "user_alert_message": "Step-1 time alignment failed; outputs below are diagnostic only.",
            "user_alert_reasons": str(failure_reason),
            "output_dir": str(run_dir),
            "plot": {
                "enabled": True,
                "files": [str(fig_corr_path), str(fig_raw_path)],
            },
        },
    }
    save_metrics(run_dir, metrics_payload)
    report_zh_path, report_en_path = write_run_reports(run_dir, metrics_payload)

    print("\n--- STEP 1 FAILURE ---")
    print(f"reason: {failure_reason}")
    print("\n--- OUTPUT FILES ---")
    print(f"Saved figure: {fig_corr_path}")
    print(f"Saved figure: {fig_raw_path}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print(f"Saved metrics: {run_dir / 'metrics_summary.csv'}")
    print(f"Saved report: {report_zh_path}")
    print(f"Saved report: {report_en_path}")

    plt.close(fig_corr)
    plt.close(fig_raw)


def _run_alignment_diagnostics(
    *,
    t_gt,
    pos_gt,
    pr_corrected,
    pr_final,
    pr_final_global,
    raw_stats,
    step3_stats,
    traj_metrics,
    time_metrics,
    quality_segment_duration_s: float,
    quality_segment_overlap_ratio: float,
    quality_min_segment_samples: int,
    quality_good_rmse_m: float,
    quality_partial_rmse_m: float,
    quality_good_segment_cv: float,
    quality_good_heading_p90_deg: float,
    quality_partial_min_improve_pct: float,
    rigid_check_max_path_ratio: float,
    rigid_check_max_bbox_ratio: float,
    rigid_check_max_global_local_ratio: float,
    rigid_check_max_sim3_gain_ratio: float,
):
    alignment_quality = _compute_alignment_quality(
        t_ref=t_gt,
        pos_ref=pos_gt,
        pos_step3=pr_final,
        raw_rmse_m=raw_stats["rmse"],
        step3_rmse_m=step3_stats["rmse"],
        segment_duration_s=quality_segment_duration_s,
        overlap_ratio=quality_segment_overlap_ratio,
        min_samples=quality_min_segment_samples,
        good_rmse_m=quality_good_rmse_m,
        partial_rmse_m=quality_partial_rmse_m,
        good_seg_cv=quality_good_segment_cv,
        good_heading_p90_deg=quality_good_heading_p90_deg,
        partial_min_improve_pct=quality_partial_min_improve_pct,
    )
    quality_label = str(alignment_quality.pop("_quality_label", "unknown"))

    rigid_alignability = _compute_rigid_alignability(
        t_ref=t_gt,
        pos_ref=pos_gt,
        pos_step2=pr_corrected,
        pos_step3=pr_final_global,
        segment_duration_s=quality_segment_duration_s,
        overlap_ratio=quality_segment_overlap_ratio,
        min_samples=quality_min_segment_samples,
        max_path_ratio=rigid_check_max_path_ratio,
        max_bbox_ratio=rigid_check_max_bbox_ratio,
        max_global_local_ratio=rigid_check_max_global_local_ratio,
        max_sim3_gain_ratio=rigid_check_max_sim3_gain_ratio,
    )
    rigid_alignability_label = str(rigid_alignability.pop("_rigid_alignability_label", "unknown"))
    rigid_alignability_reasons = str(rigid_alignability.pop("_rigid_alignability_reasons", ""))

    user_alert = _build_user_alert(
        time_metrics=time_metrics,
        traj_metrics=traj_metrics,
        quality_label=quality_label,
        rigid_label=rigid_alignability_label,
        rigid_reasons=rigid_alignability_reasons,
    )
    alert_level = str(user_alert.pop("_alert_level", "ok"))
    alert_message = str(user_alert.pop("_alert_message", ""))
    alert_reasons = str(user_alert.pop("_alert_reasons", ""))

    piecewise_diag, piecewise_detail = _compute_piecewise_diagnostics(
        t_ref=t_gt,
        pos_ref=pos_gt,
        pos_step2=pr_corrected,
        pos_step3=pr_final_global,
        segment_duration_s=quality_segment_duration_s,
        overlap_ratio=quality_segment_overlap_ratio,
        min_samples=quality_min_segment_samples,
    )

    return {
        "alignment_quality": alignment_quality,
        "quality_label": quality_label,
        "rigid_alignability": rigid_alignability,
        "rigid_alignability_label": rigid_alignability_label,
        "rigid_alignability_reasons": rigid_alignability_reasons,
        "user_alert": user_alert,
        "alert_level": alert_level,
        "alert_message": alert_message,
        "alert_reasons": alert_reasons,
        "piecewise_diag": piecewise_diag,
        "piecewise_detail": piecewise_detail,
    }


def _compute_pose_metrics_by_stage(
    *,
    t_gt,
    pos_gt,
    quat_gt,
    stage_trajs,
    eval_align_mode: str,
    eval_n_to_align: int,
    eval_project_to_plane: str,
    rpe_delta,
    rpe_delta_unit,
    rpe_delta_tol,
    rpe_all_pairs,
    rpe_pairs_from_reference,
    t_max_diff: float,
    t_offset: float,
    t_start,
    t_end,
):
    ape_metrics_by_stage = {}
    rpe_metrics_by_stage = {}
    seconds_from_start = np.asarray(t_gt, dtype=float) - float(t_gt[0])
    distances_from_start = cum_distance(pos_gt)
    ref_eval_pos, ref_eval_quat = project_to_plane(
        pos_gt, quat_gt, plane=eval_project_to_plane
    )

    for stage_name, stage_data in stage_trajs.items():
        eval_pos, eval_quat = align_for_eval(
            pos_ref=pos_gt,
            quat_ref=quat_gt,
            pos_est=stage_data["pos"],
            quat_est=stage_data["quat"],
            mode=eval_align_mode,
            n_to_align=eval_n_to_align,
        )
        est_eval_pos, est_eval_quat = project_to_plane(
            eval_pos, eval_quat, plane=eval_project_to_plane
        )
        ape_metrics_by_stage[stage_name] = compute_ape_evo_style(
            pos_ref=ref_eval_pos,
            quat_ref=ref_eval_quat,
            pos_est=est_eval_pos,
            quat_est=est_eval_quat,
            include_raw=True,
        )
        rpe_metrics_by_stage[stage_name] = compute_rpe_evo_style(
            pos_ref=ref_eval_pos,
            quat_ref=ref_eval_quat,
            pos_est=est_eval_pos,
            quat_est=est_eval_quat,
            delta=rpe_delta,
            delta_unit=rpe_delta_unit,
            rel_delta_tol=rpe_delta_tol,
            all_pairs=rpe_all_pairs,
            pairs_from_reference=rpe_pairs_from_reference,
            include_raw=True,
        )
        ape_metrics_by_stage[stage_name]["_x_axis"]["seconds_from_start"] = seconds_from_start
        ape_metrics_by_stage[stage_name]["_x_axis"]["distances_from_start"] = distances_from_start
        delta_ids = rpe_metrics_by_stage[stage_name]["_x_axis"]["delta_ids"].astype(int)
        valid = (delta_ids >= 0) & (delta_ids < seconds_from_start.size)
        rpe_metrics_by_stage[stage_name]["_x_axis"]["seconds_from_start"] = seconds_from_start[delta_ids[valid]]
        rpe_metrics_by_stage[stage_name]["_x_axis"]["distances_from_start"] = distances_from_start[delta_ids[valid]]

    return {
        "ape": ape_metrics_by_stage,
        "rpe": rpe_metrics_by_stage,
        "rpe_config": {
            "delta": rpe_delta,
            "delta_unit": rpe_delta_unit,
            "delta_tol": rpe_delta_tol,
            "all_pairs": bool(rpe_all_pairs),
            "pairs_from_reference": bool(rpe_pairs_from_reference),
        },
        "eval_config": {
            "t_max_diff": float(t_max_diff),
            "t_offset": float(t_offset),
            "t_start": None if t_start is None else float(t_start),
            "t_end": None if t_end is None else float(t_end),
            "align": str(eval_align_mode),
            "n_to_align": int(eval_n_to_align),
            "project_to_plane": str(eval_project_to_plane),
        },
    }


def run_pipeline_modular(args, script_dir: Path):
    run_dir = make_output_dir(
        script_dir,
        output_root=getattr(args, "output_root", ""),
        run_label=getattr(args, "run_label", ""),
    )
    print(f"Saving outputs to: {run_dir}")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(args.gt_csv)
    if not gt_path.is_absolute():
        gt_path = script_dir / gt_path
    if not gt_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_path}")

    print("Loading GT trajectory...")
    gt_format = getattr(args, "gt_format", "csv")
    gt_topic = getattr(args, "gt_topic", "")
    t_gt, pos_gt, quat_gt = load_reference_trajectory(
        gt_path,
        gt_format=gt_format,
        gt_topic=gt_topic,
    )

    mode = "synthetic" if args.synthetic else "real"
    sanity_metrics = {
        "extrinsic_rotation_error_deg": np.nan,
        "extrinsic_translation_error_m": np.nan,
        "world_rotation_error_deg": np.nan,
        "world_translation_error_m": np.nan,
    }

    if args.synthetic:
        ARTIFICIAL_OFFSET = 0.123
        R_ext_true = R.from_euler("zyx", [45, -20, 30], degrees=True).as_matrix()
        t_ext_true = np.array([0.5, -0.2, 0.1])
        Rw_true = R.from_euler("z", 25, degrees=True).as_matrix()
        tw_true = np.array([1.5, -0.5, 0.3])

        t_est = t_gt + ARTIFICIAL_OFFSET
        pos_est = np.zeros_like(pos_gt)
        quat_est = np.zeros_like(quat_gt)
        R_gt = R.from_quat(quat_gt).as_matrix()

        for i in range(len(t_gt)):
            R_abs = R_gt[i] @ R_ext_true
            quat_est[i] = R.from_matrix(Rw_true.T @ R_abs).as_quat()
            p_offset = pos_gt[i] + R_gt[i] @ t_ext_true
            pos_est[i] = Rw_true.T @ (p_offset - tw_true)
    else:
        if args.est_path is None or args.est_path == "":
            raise ValueError("Real mode requires <gt_file> <est_file>, or --gt/--est.")

        est_path = Path(args.est_path)
        if not est_path.is_absolute():
            est_path = script_dir / est_path
        if not est_path.exists():
            raise FileNotFoundError(f"Estimation trajectory file not found: {est_path}")

        print("Loading estimation trajectory...")
        est_format = args.est_format
        est_topic = getattr(args, "est_topic", "")
        t_est, pos_est, quat_est = load_estimation_trajectory(
            est_path,
            est_format,
            est_topic=est_topic,
        )

    t_start = getattr(args, "t_start", None)
    t_end = getattr(args, "t_end", None)
    t_offset = float(getattr(args, "t_offset", 0.0))
    quality_segment_duration_s = float(getattr(args, "quality_segment_duration_s", 10.0))
    quality_segment_overlap_ratio = float(getattr(args, "quality_segment_overlap_ratio", 0.5))
    quality_good_rmse_m = float(getattr(args, "quality_good_rmse_m", 0.5))
    quality_partial_rmse_m = float(getattr(args, "quality_partial_rmse_m", 8.0))
    quality_good_segment_cv = float(getattr(args, "quality_good_segment_cv", 0.4))
    quality_good_heading_p90_deg = float(getattr(args, "quality_good_heading_p90_deg", 60.0))
    quality_partial_min_improve_pct = float(getattr(args, "quality_partial_min_improve_pct", 20.0))
    quality_min_segment_samples = int(getattr(args, "quality_min_segment_samples", 80))
    t_gt, pos_gt, quat_gt = _apply_time_window(
        t_gt, pos_gt, quat_gt, t_start=t_start, t_end=t_end
    )
    t_est = np.asarray(t_est, dtype=float) + t_offset
    t_est, pos_est, quat_est = _apply_time_window(
        t_est, pos_est, quat_est, t_start=t_start, t_end=t_end
    )

    print("--- STEP 1: TIME ALIGNMENT ---")
    dt_resample = float(getattr(args, "dt_resample", 0.001))
    offset_search_window_s = float(getattr(args, "offset_search_window_s", 0.0))
    offset_min_match_ratio = float(getattr(args, "offset_min_match_ratio", 0.3))
    evo_match_max_diff_s = float(getattr(args, "t_max_diff", 0.02))
    step1 = _run_time_alignment(
        t_gt=t_gt,
        quat_gt=quat_gt,
        t_est=t_est,
        quat_est=quat_est,
        dt_resample=dt_resample,
        offset_search_window_s=offset_search_window_s,
        offset_min_match_ratio=offset_min_match_ratio,
        evo_match_max_diff_s=evo_match_max_diff_s,
        artificial_offset_s=ARTIFICIAL_OFFSET if args.synthetic else None,
    )
    calculated_offset = float(step1["calculated_offset"])
    time_metrics = step1["time_metrics"]
    step1_forced_candidate = bool(step1["step1_forced_candidate"])
    step1_force_reason = str(step1["step1_force_reason"])
    t_uniform = step1["t_uniform"]
    sig_gt = step1["sig_gt"]
    sig_est = step1["sig_est"]
    corr = step1["corr"]
    lags = step1["lags"]

    print(f"Calculated Time Offset: {calculated_offset:.4f} s")

    print_metric_block(
        "TIME ALIGNMENT METRICS",
        time_metrics,
        unit_map={
            "offset_est_s": "s",
            "offset_err_ms": "ms",
            "omega_rmse_before": "rad/s",
            "omega_rmse_after": "rad/s",
            "omega_rmse_improve_pct": "%",
            "evo_t_offset_used_s": "s",
            "evo_match_max_diff_s": "s",
        },
    )

    title_offset = f"Known offset: {ARTIFICIAL_OFFSET}s" if args.synthetic else "Unknown offset"
    fig_corr_path, fig1_path = _plot_step1_outputs(
        plots_dir=plots_dir,
        t_uniform=t_uniform,
        sig_gt=sig_gt,
        sig_est=sig_est,
        corr=corr,
        lags=lags,
        dt_resample=dt_resample,
        calculated_offset=calculated_offset,
        title_offset=title_offset,
    )

    downsample_hz = 0.0 if bool(getattr(args, "no_downsample", False)) else float(getattr(args, "downsample_hz", 100.0))
    solve_eval = _prepare_solve_eval_trajectories(
        t_gt=t_gt,
        pos_gt=pos_gt,
        quat_gt=quat_gt,
        t_est=t_est,
        pos_est=pos_est,
        quat_est=quat_est,
        calculated_offset=calculated_offset,
        downsample_hz=downsample_hz,
        quat_interp=args.quat_interp,
    )
    t_est_sync = solve_eval["t_est_sync"]
    t_gt = solve_eval["t_gt"]
    pos_gt = solve_eval["pos_gt"]
    quat_gt = solve_eval["quat_gt"]
    pr_sync = solve_eval["pr_sync"]
    qr_sync = solve_eval["qr_sync"]
    pos_gt_solve = solve_eval["pos_gt_solve"]
    quat_gt_solve = solve_eval["quat_gt_solve"]
    pr_solve = solve_eval["pr_solve"]
    qr_solve = solve_eval["qr_solve"]
    overlap_info = solve_eval["overlap_info"]
    downsample_info = solve_eval["downsample_info"]
    if not bool(overlap_info["fallback_used"]):
        print(
            "Using overlap-only solve/eval trajectory: "
            f"{overlap_info['selected_samples']} / {overlap_info['input_samples']} "
            f"samples inside estimation time support"
        )
    else:
        print(
            "Using full GT solve/eval trajectory with extrapolation fallback: "
            f"{overlap_info['fallback_reason']}"
        )

    if bool(downsample_info["enabled"]):
        print(
            "Downsampled solve/eval trajectory: "
            f"{downsample_info['input_samples']} -> {downsample_info['output_samples']} "
            f"samples at <= {downsample_info['max_hz']:.3f} Hz"
        )

    print("\n--- STEP 2: SOLVING EXTRINSICS ---")
    solved = _solve_step2_step3(
        pr_sync=pr_sync,
        qr_sync=qr_sync,
        pos_gt_solve=pos_gt_solve,
        quat_gt_solve=quat_gt_solve,
        pr_solve=pr_solve,
        qr_solve=qr_solve,
    )
    R_calc = solved["R_calc"]
    t_calc = solved["t_calc"]
    Rw_calc = solved["Rw_calc"]
    tw_calc = solved["tw_calc"]
    pr_corrected = solved["pr_corrected"]
    pr_corrected_solve = solved["pr_corrected_solve"]
    pr_final = solved["pr_final"]
    pr_final_global = solved["pr_final_global"]
    q_step2 = solved["q_step2"]
    q_step3 = solved["q_step3"]
    step3_choice = solved["step3_choice"]
    print(f"Calculated Extrinsic Rotation Matrix:\n{np.round(R_calc, 4)}")
    print(f"Calculated Translation: {np.round(t_calc, 4)} m")

    print("\n--- STEP 3: WORLD ALIGNMENT ---")
    print(f"Calculated World Rotation Matrix:\n{np.round(Rw_calc, 4)}")
    print(f"Calculated World Translation: {np.round(tw_calc, 4)} m")
    print(
        f"Step3 selected_rmse={step3_choice['step3_rmse_selected_m']:.6f} m"
    )

    if args.synthetic:
        rot_ext_err_deg = float(np.degrees(R.from_matrix(R_calc.T @ R_ext_true).magnitude()))
        t_ext_err = float(np.linalg.norm(t_calc - t_ext_true))
        if np.all(np.isfinite(Rw_calc)) and np.all(np.isfinite(tw_calc)):
            rot_world_err_deg = float(np.degrees(R.from_matrix(Rw_calc.T @ Rw_true).magnitude()))
            t_world_err = float(np.linalg.norm(tw_calc - tw_true))
        else:
            rot_world_err_deg = float("nan")
            t_world_err = float("nan")
        print("\n--- SANITY CHECK (vs Injected Truth) ---")
        print(f"Extrinsic rotation error: {rot_ext_err_deg:.4f} deg")
        print(f"Extrinsic translation error: {t_ext_err:.4f} m")
        print(f"World rotation error: {rot_world_err_deg:.4f} deg")
        print(f"World translation error: {t_world_err:.4f} m")
        sanity_metrics = {
            "extrinsic_rotation_error_deg": rot_ext_err_deg,
            "extrinsic_translation_error_m": t_ext_err,
            "world_rotation_error_deg": rot_world_err_deg,
            "world_translation_error_m": t_world_err,
        }

    step2_metrics = _compute_step2_residual_metrics(
        pos_gt_solve=pos_gt_solve,
        quat_gt_solve=quat_gt_solve,
        pr_solve=pr_solve,
        qr_solve=qr_solve,
        R_calc=R_calc,
        t_calc=t_calc,
    )

    print_metric_block(
        "STEP 2 RESIDUAL METRICS",
        step2_metrics,
        unit_map={
            "rot_res_mean_deg": "deg",
            "rot_res_median_deg": "deg",
            "rot_res_p95_deg": "deg",
            "trans_eq_rmse_m": "m",
            "trans_eq_p95_m": "m",
        },
    )

    traj_eval = _compute_trajectory_metrics(
        pos_gt=pos_gt,
        pr_sync=pr_sync,
        pr_corrected=pr_corrected,
        pr_final=pr_final,
    )
    raw_err = traj_eval["raw_err"]
    step2_err = traj_eval["step2_err"]
    step3_err = traj_eval["step3_err"]
    raw_stats = traj_eval["raw_stats"]
    step2_stats = traj_eval["step2_stats"]
    step3_stats = traj_eval["step3_stats"]
    traj_metrics = traj_eval["traj_metrics"]

    print_metric_block(
        "TRAJECTORY METRICS",
        traj_metrics,
        unit_map={
            "ate_rmse_raw_m": "m",
            "ate_rmse_step2_m": "m",
            "ate_rmse_step3_m": "m",
            "ate_p95_raw_m": "m",
            "ate_p95_step2_m": "m",
            "ate_p95_step3_m": "m",
            "ate_rmse_improve_raw_to_step3_pct": "%",
            "ate_rmse_improve_step2_to_step3_pct": "%",
        },
    )

    rigid_check_max_path_ratio = float(getattr(args, "rigid_check_max_path_ratio", 3.0))
    rigid_check_max_bbox_ratio = float(getattr(args, "rigid_check_max_bbox_ratio", 3.0))
    rigid_check_max_global_local_ratio = float(getattr(args, "rigid_check_max_global_local_ratio", 6.0))
    rigid_check_max_sim3_gain_ratio = float(getattr(args, "rigid_check_max_sim3_gain_ratio", 0.3))

    diagnostics = _run_alignment_diagnostics(
        t_gt=t_gt,
        pos_gt=pos_gt,
        pr_corrected=pr_corrected,
        pr_final=pr_final,
        pr_final_global=pr_final_global,
        raw_stats=raw_stats,
        step3_stats=step3_stats,
        traj_metrics=traj_metrics,
        time_metrics=time_metrics,
        quality_segment_duration_s=quality_segment_duration_s,
        quality_segment_overlap_ratio=quality_segment_overlap_ratio,
        quality_min_segment_samples=quality_min_segment_samples,
        quality_good_rmse_m=quality_good_rmse_m,
        quality_partial_rmse_m=quality_partial_rmse_m,
        quality_good_segment_cv=quality_good_segment_cv,
        quality_good_heading_p90_deg=quality_good_heading_p90_deg,
        quality_partial_min_improve_pct=quality_partial_min_improve_pct,
        rigid_check_max_path_ratio=rigid_check_max_path_ratio,
        rigid_check_max_bbox_ratio=rigid_check_max_bbox_ratio,
        rigid_check_max_global_local_ratio=rigid_check_max_global_local_ratio,
        rigid_check_max_sim3_gain_ratio=rigid_check_max_sim3_gain_ratio,
    )
    alignment_quality = diagnostics["alignment_quality"]
    quality_label = diagnostics["quality_label"]
    rigid_alignability = diagnostics["rigid_alignability"]
    rigid_alignability_label = diagnostics["rigid_alignability_label"]
    rigid_alignability_reasons = diagnostics["rigid_alignability_reasons"]
    user_alert = diagnostics["user_alert"]
    alert_level = diagnostics["alert_level"]
    alert_message = diagnostics["alert_message"]
    alert_reasons = diagnostics["alert_reasons"]
    piecewise_diag = diagnostics["piecewise_diag"]
    piecewise_detail = diagnostics["piecewise_detail"]

    print("\n--- ALIGNMENT QUALITY ---")
    print(f"quality_label: {quality_label}")
    print_metric_block(
        "ALIGNMENT QUALITY METRICS",
        alignment_quality,
        unit_map={
            "step3_rmse_m": "m",
            "raw_to_step3_improve_pct": "%",
            "segment_rmse_mean_m": "m",
            "segment_rmse_std_m": "m",
            "heading_median_deg": "deg",
            "heading_p90_deg": "deg",
        },
    )

    print("\n--- USER ALERT ---")
    print(f"alert_level: {alert_level}")
    if alert_message:
        print(f"alert_message: {alert_message}")
    if alert_reasons:
        print(f"alert_reasons: {alert_reasons}")

    _plot_piecewise_diagnostics(plots_dir=plots_dir, piecewise_detail=piecewise_detail)

    stage_trajs = {
        "raw": {"pos": pr_sync, "quat": qr_sync},
        "step2": {"pos": pr_corrected, "quat": q_step2},
        "step3": {"pos": pr_final, "quat": q_step3},
    }
    eval_align_mode = str(getattr(args, "eval_align", "none"))
    eval_n_to_align = int(getattr(args, "eval_n_to_align", -1))
    eval_project_to_plane = str(getattr(args, "eval_project_to_plane", "none"))
    pose_metrics = _compute_pose_metrics_by_stage(
        t_gt=t_gt,
        pos_gt=pos_gt,
        quat_gt=quat_gt,
        stage_trajs=stage_trajs,
        eval_align_mode=eval_align_mode,
        eval_n_to_align=eval_n_to_align,
        eval_project_to_plane=eval_project_to_plane,
        rpe_delta=args.rpe_delta,
        rpe_delta_unit=args.rpe_delta_unit,
        rpe_delta_tol=args.rpe_delta_tol,
        rpe_all_pairs=args.rpe_all_pairs,
        rpe_pairs_from_reference=args.rpe_pairs_from_reference,
        t_max_diff=float(getattr(args, "t_max_diff", 0.02)),
        t_offset=float(getattr(args, "t_offset", 0.0)),
        t_start=getattr(args, "t_start", None),
        t_end=getattr(args, "t_end", None),
    )

    stage_order = ["raw", "step2", "step3"]

    print("\n--- METRICS (APE/RPE) ---")
    ape_pose_relation = normalize_pose_relation("ape", getattr(args, "ape_pose_relation", "trans_part"))
    rpe_pose_relation = normalize_pose_relation("rpe", getattr(args, "rpe_pose_relation", "trans_part"))
    for stage_name in stage_order:
        ape_t = pose_metrics["ape"][stage_name][ape_pose_relation]["rmse"]
        ape_r = pose_metrics["ape"][stage_name]["rotation_angle_deg"]["rmse"]
        rpe_t = pose_metrics["rpe"][stage_name][rpe_pose_relation]["rmse"]
        rpe_r = pose_metrics["rpe"][stage_name]["rotation_angle_deg"]["rmse"]
        pairs = pose_metrics["rpe"][stage_name]["pair_count"]
        print(
            f"{stage_name}: "
            f"APE_{ape_pose_relation}_rmse={ape_t:.6f}, "
            f"APE_rot_rmse={ape_r:.6f} deg, "
            f"RPE_{rpe_pose_relation}_rmse={rpe_t:.6f}, "
            f"RPE_rot_rmse={rpe_r:.6f} deg, "
            f"pairs={pairs}"
        )

    fig2_path, fig_step3_map_path = _plot_stage_alignment_maps(
        plots_dir=plots_dir,
        pos_gt=pos_gt,
        pr_sync=pr_sync,
        pr_corrected=pr_corrected,
        pr_final=pr_final,
    )

    rerun_info = {
        "enabled": "false",
        "status": "disabled",
        "message": "Use --rerun to enable.",
    }
    if bool(getattr(args, "rerun", False)):
        rerun_info = log_alignment_to_rerun(
            run_dir=run_dir,
            gt_xyz=pos_gt,
            raw_xyz=pr_sync,
            step2_xyz=pr_corrected,
            step3_xyz=pr_final,
            timestamps_s=t_gt,
            gt_quat=quat_gt,
            raw_quat=qr_sync,
            step2_quat=q_step2,
            step3_quat=q_step3,
            raw_error_m=raw_err,
            step2_error_m=step2_err,
            step3_error_m=step3_err,
            app_id="epa_alignment",
            spawn=not bool(getattr(args, "rerun_no_spawn", False)),
            stride=int(getattr(args, "rerun_stride", 20)),
            motion_stride=int(getattr(args, "rerun_motion_stride", 5)),
        )
        print(f"Rerun status: {rerun_info['status']} - {rerun_info['message']}")

    metrics_payload = {
        "time_alignment": time_metrics,
        "step2_residuals": step2_metrics,
        "trajectory": traj_metrics,
        "step3_selection": {
            "step3_rmse_selected_m": float(step3_choice["step3_rmse_selected_m"]),
        },
        "user_alert": user_alert,
        "alignment_quality": alignment_quality,
        "rigid_alignability": rigid_alignability,
        "piecewise_diagnostics": piecewise_diag,
        "pose_metrics": pose_metrics,
        "sanity_check": sanity_metrics,
        "estimated_params": {
            "R_ext": R_calc,
            "t_ext": t_calc,
            "R_world": Rw_calc,
            "t_world": tw_calc,
            "offset_s": calculated_offset,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "gt_path": str(gt_path),
            "est_path": str(args.est_path) if args.est_path else "",
            "gt_format": str(gt_format),
            "est_format": str(getattr(args, "est_format", "auto")),
            "gt_topic": str(gt_topic),
            "est_topic": str(getattr(args, "est_topic", "")),
            "quat_interp": args.quat_interp,
            "t_offset": float(getattr(args, "t_offset", 0.0)),
            "t_max_diff": float(getattr(args, "t_max_diff", 0.02)),
            "t_start": None if getattr(args, "t_start", None) is None else float(args.t_start),
            "t_end": None if getattr(args, "t_end", None) is None else float(args.t_end),
            "step1_forced_candidate": bool(step1_forced_candidate),
            "step1_force_reason": str(step1_force_reason),
            "ape_pose_relation": str(getattr(args, "ape_pose_relation", "trans_part")),
            "rpe_pose_relation": str(getattr(args, "rpe_pose_relation", "trans_part")),
            "eval_align": eval_align_mode,
            "eval_n_to_align": eval_n_to_align,
            "eval_project_to_plane": eval_project_to_plane,
            "overlap_selection": overlap_info,
            "downsample": downsample_info,
            "alignment_quality_label": quality_label,
            "rigid_alignability_label": rigid_alignability_label,
            "rigid_alignability_reasons": rigid_alignability_reasons,
            "user_alert_level": alert_level,
            "user_alert_message": alert_message,
            "user_alert_reasons": alert_reasons,
            "rigid_check_max_path_ratio": float(rigid_check_max_path_ratio),
            "rigid_check_max_bbox_ratio": float(rigid_check_max_bbox_ratio),
            "rigid_check_max_global_local_ratio": float(rigid_check_max_global_local_ratio),
            "rigid_check_max_sim3_gain_ratio": float(rigid_check_max_sim3_gain_ratio),
            "rerun": rerun_info,
            "output_dir": str(run_dir),
        },
    }
    output_info = _write_outputs(
        run_dir=run_dir,
        plots_dir=plots_dir,
        metrics_payload=metrics_payload,
        args=args,
    )
    bundle_path = output_info["bundle_path"]
    report_zh_path = output_info["report_zh_path"]
    report_en_path = output_info["report_en_path"]

    print("\n--- OUTPUT FILES ---")
    print(f"Saved figure: {fig_corr_path}")
    print(f"Saved figure: {fig1_path}")
    print(f"Saved figure: {fig2_path}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print(f"Saved metrics: {run_dir / 'metrics_summary.csv'}")
    print(f"Saved report: {report_zh_path}")
    print(f"Saved report: {report_en_path}")
    if bundle_path is not None:
        print(f"Saved results: {bundle_path}")
