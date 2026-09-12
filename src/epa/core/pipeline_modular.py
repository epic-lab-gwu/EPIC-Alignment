from datetime import datetime
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
import matplotlib

matplotlib.use("Agg")
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R

from .evaluation import (
    apply_input_coverage_to_success,
    normalize_pose_relation,
    print_metric_block,
)
from epa.io.outputs import make_output_dir
from epa.io.trajectory import load_estimation_trajectory, load_reference_trajectory
from .association import (
    _associate_gt_est as _associate_gt_est,
    _match_nearest_timestamps as _match_nearest_timestamps,
    _offset_match_diagnostics as _offset_match_diagnostics,
)
from .solve_eval import (
    _downsample_by_max_hz as _downsample_by_max_hz,
    _prepare_solve_eval_trajectories,
    _select_gt_overlap_window as _select_gt_overlap_window,
)
from .trajectory_alignment import (
    _compute_alignment_error_metrics,
    _compute_extrinsic_residual_metrics,
    _solve_extrinsic_and_world_alignment,
)
from .time_sync import (
    _run_time_alignment,
    _search_direct_offset_from_matched_pairs as _search_direct_offset_from_matched_pairs,
)
from .outputs import (
    _plot_piecewise_diagnostics,
    _plot_stage_alignment_maps,
    _plot_step1_outputs,
    _write_outputs,
    write_pose_state_csv,
)
from .pipeline_metrics import (
    _compute_pose_metrics_by_stage,
    _run_alignment_diagnostics,
)
from .diagnostics import (
    _build_failure_diagnosis,
    _compute_input_coverage_diagnostics,
    _diagnosis_tags_from_metrics,
    _compute_piecewise_alignment as _compute_piecewise_alignment,
)
from ..metric_cli_common import align_for_eval_with_info
from ..viz.rerun_viz import log_alignment_to_rerun


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


from .pipeline_views import (
    _annotate_sr_reliability,
    _apply_eval_alignment_from_info,
    _compute_orientation_diagnostics,
    _default_interactive_view,
    _interactive_eval_view_specs,
    _interactive_view_metric_summary,
    _print_sim3_ov_eval_terminal_metrics,
    _public_eval_align_mode,
)

def run_pipeline_modular(args, script_dir: Path):
    verbose = bool(getattr(args, "verbose", False))
    terminal_only = not bool(getattr(args, "plot", True))
    if terminal_only:
        # Preserve the terminal preamble without creating a persistent output tree.
        run_dir = Path("/tmp") / f"epa-terminal-only-{os.getpid()}"
    else:
        run_dir = make_output_dir(
            script_dir,
            output_root=getattr(args, "output_root", ""),
            run_label=getattr(args, "run_label", ""),
        )
    print(f"Saving outputs to: {run_dir}")
    plots_dir = run_dir / "plots"
    if not terminal_only:
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
    quality_threshold_mode = str(getattr(args, "quality_threshold_mode", "adaptive"))
    quality_good_rmse_ratio = float(getattr(args, "quality_good_rmse_ratio", 0.01))
    quality_partial_rmse_ratio = float(getattr(args, "quality_partial_rmse_ratio", 0.05))
    quality_critical_rmse_ratio = float(getattr(args, "quality_critical_rmse_ratio", 0.10))
    quality_good_rmse_floor_m = float(getattr(args, "quality_good_rmse_floor_m", 0.05))
    quality_partial_rmse_floor_m = float(getattr(args, "quality_partial_rmse_floor_m", 0.25))
    quality_critical_rmse_floor_m = float(getattr(args, "quality_critical_rmse_floor_m", 1.0))
    quality_min_segment_samples = int(getattr(args, "quality_min_segment_samples", 80))
    t_gt, pos_gt, quat_gt = _apply_time_window(
        t_gt, pos_gt, quat_gt, t_start=t_start, t_end=t_end
    )
    t_est = np.asarray(t_est, dtype=float) + t_offset
    t_est, pos_est, quat_est = _apply_time_window(
        t_est, pos_est, quat_est, t_start=t_start, t_end=t_end
    )
    t_gt_input = np.asarray(t_gt, dtype=float).copy()
    pos_gt_input = np.asarray(pos_gt, dtype=float).copy()
    t_est_input = np.asarray(t_est, dtype=float).copy()

    print("--- STEP 1: TIME ALIGNMENT ---")
    disable_all_calibration = bool(getattr(args, "disable_calibration", False))
    disable_time_offset_calibration = disable_all_calibration or bool(
        getattr(args, "disable_time_offset_calibration", False)
    )
    disable_extrinsic_calibration = disable_all_calibration or bool(
        getattr(args, "disable_extrinsic_calibration", False)
    )
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
        disable_time_offset_calibration=disable_time_offset_calibration,
    )
    calculated_offset = float(step1["calculated_offset"])
    time_metrics = step1["time_metrics"]
    for prefix, timestamps in (("reference", t_gt_input), ("estimate", t_est_input)):
        diffs = np.diff(np.asarray(timestamps, dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
        time_metrics[f"{prefix}_median_dt_s"] = float(np.median(diffs)) if diffs.size else float("nan")
    step1_forced_candidate = bool(step1["step1_forced_candidate"])
    step1_force_reason = str(step1["step1_force_reason"])
    t_uniform = step1["t_uniform"]
    sig_gt = step1["sig_gt"]
    sig_est = step1["sig_est"]
    corr = step1["corr"]
    lags = step1["lags"]

    print(f"Calculated Time Offset: {calculated_offset:.4f} s")

    if verbose:
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

    if terminal_only:
        fig_corr_path, fig1_path = None, None
    else:
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

    input_coverage = _compute_input_coverage_diagnostics(
        t_ref=t_gt_input,
        pos_ref=pos_gt_input,
        t_est=t_est_input,
        offset_est_s=calculated_offset,
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

    requested_eval_align_alias = str(getattr(args, "eval_align", "none")).strip().lower()
    requested_eval_align_mode = _public_eval_align_mode(requested_eval_align_alias)
    if requested_eval_align_mode in {"posyaw", "epa_posyaw"}:
        step3_global_align_mode = "posyaw"
    elif requested_eval_align_mode == "se3-original":
        step3_global_align_mode = "se3-original"
    else:
        step3_global_align_mode = "se3"
    robust_kernel = str(getattr(args, "robust_kernel", "none") or "none").strip().lower()
    robust_kernel_delta_m = getattr(args, "robust_kernel_delta_m", None)
    robust_kernel_max_iterations = int(getattr(args, "robust_kernel_max_iterations", 3))

    print("\n--- STEP 2: SOLVING EXTRINSICS ---")
    solved = _solve_extrinsic_and_world_alignment(
        pr_sync=pr_sync,
        qr_sync=qr_sync,
        pos_gt_solve=pos_gt_solve,
        quat_gt_solve=quat_gt_solve,
        pr_solve=pr_solve,
        qr_solve=qr_solve,
        global_align_mode=step3_global_align_mode,
        robust_kernel=robust_kernel,
        robust_kernel_delta_m=robust_kernel_delta_m,
        robust_kernel_max_iterations=robust_kernel_max_iterations,
        calibration_timestamps_s=t_gt,
        calibration_source_timestamps_s=solve_eval["t_est_sync"],
        disable_extrinsic_calibration=disable_extrinsic_calibration,
        compare_identity_candidate=not bool(getattr(args, "disable_identity_safeguard", False)),
    )
    R_calc = solved["R_calc"]
    t_calc = solved["t_calc"]
    Rw_calc = solved["Rw_calc"]
    tw_calc = solved["tw_calc"]
    pr_corrected = solved["pr_corrected"]
    pr_final = solved["pr_final"]
    pr_final_global = solved["pr_final_global"]
    q_step2 = solved["q_step2"]
    q_step3 = solved["q_step3"]
    step3_choice = solved["step3_choice"]
    print(
        f"Extrinsic selection: {step3_choice['extrinsic_selection_reason']} "
        f"(rotation information ratio={step3_choice['extrinsic_rotation_information_ratio']:.6f}, "
        f"identity RMSE={step3_choice['extrinsic_identity_position_rmse_m']:.6f} m, "
        f"selected RMSE={step3_choice['extrinsic_selected_position_rmse_m']:.6f} m)"
    )
    if verbose:
        print(f"Calculated Extrinsic Rotation Matrix:\n{np.round(R_calc, 4)}")
        print(f"Calculated Translation: {np.round(t_calc, 4)} m")
    else:
        print(f"Calculated Translation: {np.round(t_calc, 4)} m")

    print("\n--- FINAL WORLD ALIGNMENT ---")
    if verbose:
        print(f"Calculated World Rotation Matrix:\n{np.round(Rw_calc, 4)}")
        print(f"Calculated World Translation: {np.round(tw_calc, 4)} m")
    else:
        print(f"Calculated World Translation: {np.round(tw_calc, 4)} m")
    print(
        f"Final alignment selected_rmse={step3_choice['step3_rmse_selected_m']:.6f} m"
    )
    print(
        "Final alignment mode="
        f"{step3_choice['step3_alignment_mode']} "
        f"inliers={int(step3_choice['step3_inlier_count'])} "
        f"rejected={int(step3_choice['step3_rejected_count'])}"
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

    step2_metrics = _compute_extrinsic_residual_metrics(
        pos_gt_solve=pos_gt_solve,
        quat_gt_solve=quat_gt_solve,
        pr_solve=pr_solve,
        qr_solve=qr_solve,
        R_calc=R_calc,
        t_calc=t_calc,
    )

    if verbose:
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

    traj_eval = _compute_alignment_error_metrics(
        pos_gt=pos_gt,
        pr_sync=pr_sync,
        pr_corrected=pr_corrected,
        pr_final=pr_final,
    )
    raw_err = traj_eval["raw_err"]
    step2_err = traj_eval["step2_err"]
    step3_err = traj_eval["step3_err"]
    raw_stats = traj_eval["raw_stats"]
    step3_stats = traj_eval["step3_stats"]
    traj_metrics = traj_eval["traj_metrics"]

    if verbose:
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
        quality_threshold_mode=quality_threshold_mode,
        quality_good_rmse_ratio=quality_good_rmse_ratio,
        quality_partial_rmse_ratio=quality_partial_rmse_ratio,
        quality_critical_rmse_ratio=quality_critical_rmse_ratio,
        quality_good_rmse_floor_m=quality_good_rmse_floor_m,
        quality_partial_rmse_floor_m=quality_partial_rmse_floor_m,
        quality_critical_rmse_floor_m=quality_critical_rmse_floor_m,
        rigid_check_max_path_ratio=rigid_check_max_path_ratio,
        rigid_check_max_bbox_ratio=rigid_check_max_bbox_ratio,
        rigid_check_max_global_local_ratio=rigid_check_max_global_local_ratio,
        rigid_check_max_sim3_gain_ratio=rigid_check_max_sim3_gain_ratio,
    )
    alignment_quality = diagnostics["alignment_quality"]
    quality_label = diagnostics["quality_label"]
    alignment_quality_for_diagnosis = dict(alignment_quality)
    alignment_quality_for_diagnosis["_quality_label"] = quality_label
    rigid_alignability = diagnostics["rigid_alignability"]
    rigid_alignability_label = diagnostics["rigid_alignability_label"]
    rigid_alignability_reasons = diagnostics["rigid_alignability_reasons"]
    rigid_alignability_for_diagnosis = dict(rigid_alignability)
    rigid_alignability_for_diagnosis["_rigid_alignability_label"] = rigid_alignability_label
    rigid_alignability_for_diagnosis["_rigid_alignability_reasons"] = rigid_alignability_reasons
    user_alert = diagnostics["user_alert"]
    alert_level = diagnostics["alert_level"]
    alert_message = diagnostics["alert_message"]
    alert_reasons = diagnostics["alert_reasons"]
    piecewise_diag = diagnostics["piecewise_diag"]
    piecewise_detail = diagnostics["piecewise_detail"]

    print("\n--- ALIGNMENT QUALITY ---")
    print(f"quality_label: {quality_label}")
    if verbose:
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

    if not terminal_only:
        _plot_piecewise_diagnostics(plots_dir=plots_dir, piecewise_detail=piecewise_detail)

    stage_trajs = {
        "raw": {"pos": pr_sync, "quat": qr_sync},
        "step2": {"pos": pr_corrected, "quat": q_step2},
        "step3": {"pos": pr_final, "quat": q_step3},
    }
    eval_align_mode = requested_eval_align_mode
    if requested_eval_align_mode in {
        "se3",
        "se3-original",
        "se3r",
        "epa_se3r",
        "rotation_first_se3",
        "posyaw",
        "epa_posyaw",
    }:
        eval_align_mode = "none"
    eval_n_to_align = int(getattr(args, "eval_n_to_align", -1))
    eval_align_indices_by_stage = None
    eval_project_to_plane = str(getattr(args, "eval_project_to_plane", "none"))
    pose_metrics = _compute_pose_metrics_by_stage(
        input_coverage=input_coverage,
        t_gt=t_gt,
        pos_gt=pos_gt,
        quat_gt=quat_gt,
        stage_trajs=stage_trajs,
        eval_align_mode=eval_align_mode,
        eval_n_to_align=eval_n_to_align,
        eval_project_to_plane=eval_project_to_plane,
        eval_align_indices_by_stage=eval_align_indices_by_stage,
        eval_align_step3_only=str(eval_align_mode).lower()
        in {
            "sim3",
            "epica_sim3",
            "epica_sim3_stable",
            "epica_sim3_joint",
            "epica_sim3_trimmed",
            "epa_sim3_v1",
            "epa_sim3_v2",
        },
        rpe_delta=args.rpe_delta,
        rpe_delta_unit=args.rpe_delta_unit,
        rpe_delta_tol=args.rpe_delta_tol,
        rpe_all_pairs=args.rpe_all_pairs,
        rpe_pairs_from_reference=args.rpe_pairs_from_reference,
        success_threshold_mode=str(getattr(args, "success_threshold_mode", "adaptive_knee")),
        success_threshold_m=float(getattr(args, "success_threshold_m", 10.0)),
        success_threshold_min_m=float(getattr(args, "success_threshold_min_m", 5.0)),
        success_threshold_max_m=float(getattr(args, "success_threshold_max_m", 30.0)),
        success_threshold_trim_percentile=float(getattr(args, "success_threshold_trim_percentile", 95.0)),
        success_global_gate_mode=str(getattr(args, "success_global_gate_mode", "fixed")),
        success_global_gate_m=float(getattr(args, "success_global_gate_m", 30.0)),
        success_global_gate_path_ratio=float(getattr(args, "success_global_gate_path_ratio", 0.05)),
        success_global_gate_min_m=float(getattr(args, "success_global_gate_min_m", 2.0)),
        success_global_gate_max_m=float(getattr(args, "success_global_gate_max_m", 100.0)),
        success_global_gate_percentile=float(getattr(args, "success_global_gate_percentile", 5.0)),
        success_drift_rpe_1s_m=float(getattr(args, "success_drift_rpe_1s_m", 2.0)),
        success_drift_ape_slope_mps=float(getattr(args, "success_drift_ape_slope_mps", 1.0)),
        success_drift_ape_jump_m=float(getattr(args, "success_drift_ape_jump_m", 5.0)),
        success_drift_threshold_mode=str(getattr(args, "success_drift_threshold_mode", "adaptive")),
        t_max_diff=float(getattr(args, "t_max_diff", 0.02)),
        t_offset=float(getattr(args, "t_offset", 0.0)),
        t_start=getattr(args, "t_start", None),
        t_end=getattr(args, "t_end", None),
    )
    for stage_valid in pose_metrics.get("valid_segment", {}).values():
        if isinstance(stage_valid, dict) and isinstance(stage_valid.get("success"), dict):
            apply_input_coverage_to_success(
                stage_valid["success"], input_coverage, set_primary=True
            )
    ape_pose_relation = normalize_pose_relation("ape", getattr(args, "ape_pose_relation", "trans_part"))
    rpe_pose_relation = normalize_pose_relation("rpe", getattr(args, "rpe_pose_relation", "trans_part"))
    orientation_diagnostics = _compute_orientation_diagnostics(pose_metrics, stage="step3")
    sr_reliability = _annotate_sr_reliability(
        pose_metrics,
        stage="step3",
        orientation=orientation_diagnostics,
        ape_relation=ape_pose_relation,
        rpe_relation=rpe_pose_relation,
        input_coverage=input_coverage,
    )
    case_diagnostics = _diagnosis_tags_from_metrics(
        success=pose_metrics["valid_segment"]["step3"]["success"],
        time_metrics=time_metrics,
        traj_metrics=traj_metrics,
        step3_selection=step3_choice,
        alignment_quality=alignment_quality_for_diagnosis,
        rigid_alignability=rigid_alignability_for_diagnosis,
        orientation=orientation_diagnostics,
    )
    failure_diagnosis = _build_failure_diagnosis(
        user_alert_level=alert_level,
        input_coverage=input_coverage,
        alignment_quality=alignment_quality,
        quality_label=quality_label,
        rigid_label=rigid_alignability_label,
        rigid_reasons=rigid_alignability_reasons,
        case_diagnostics=case_diagnostics,
        orientation=orientation_diagnostics,
        sr_reliability=sr_reliability,
    )

    stage_order = ["raw", "step2", "step3"]
    if requested_eval_align_mode in {"posyaw", "epa_posyaw"}:
        terminal_eval_source = "epa_posyaw"
    elif requested_eval_align_mode == "se3-original":
        terminal_eval_source = "epa_se3_original"
    elif requested_eval_align_mode == "se3":
        terminal_eval_source = "epa_se3"
    else:
        terminal_eval_source = (
            "epa_step3" if str(eval_align_mode).lower() in {"", "none"} else "epa_eval_align"
        )

    sim3_terminal_style = str(requested_eval_align_mode).lower() in {
        "sim3",
        "epica_sim3",
        "epica_sim3_stable",
        "epica_sim3_joint",
        "epica_sim3_trimmed",
        "ov_sim3",
        "epa_sim3_v1",
        "epa_sim3_v2",
    }
    if sim3_terminal_style:
        _print_sim3_ov_eval_terminal_metrics(
            pose_metrics=pose_metrics,
            ape_pose_relation=ape_pose_relation,
            rpe_pose_relation=rpe_pose_relation,
            eval_source=terminal_eval_source,
            t_ref=t_gt,
            pos_ref=pos_gt,
            mode_label=requested_eval_align_mode,
        )
    else:
        print("\n--- METRICS (APE/RPE) ---")
        display_stage_order = stage_order if verbose else ["step3"]
        for stage_name in display_stage_order:
            ape_t = pose_metrics["ape"][stage_name][ape_pose_relation]["rmse"]
            ape_r = pose_metrics["ape"][stage_name]["rotation_angle_deg"]["rmse"]
            rpe_t = pose_metrics["rpe"][stage_name][rpe_pose_relation]["rmse"]
            rpe_r = pose_metrics["rpe"][stage_name]["rotation_angle_deg"]["rmse"]
            pairs = pose_metrics["rpe"][stage_name]["pair_count"]
            rpe_time_t = pose_metrics["rpe_time_1s"][stage_name]["translation_part"]["rmse"]
            rpe_time_r = pose_metrics["rpe_time_1s"][stage_name]["rotation_angle_deg"]["rmse"]
            time_pairs = pose_metrics["rpe_time_1s"][stage_name]["pair_count"]
            success = pose_metrics["valid_segment"][stage_name]["success"]
            valid_ape_t = pose_metrics["valid_segment"][stage_name]["ape"][ape_pose_relation]["rmse"]
            valid_rpe_t = pose_metrics["valid_segment"][stage_name]["rpe"][rpe_pose_relation]["rmse"]
            sr_dist_pct = float(success["success_rate_distance"]) * 100.0
            local_sr_dist_pct = float(success["local_success_rate_distance"]) * 100.0
            # New motion-relative success scoring reports a threshold without
            # a fixed ``threshold_m`` field.  Keep terminal reporting
            # compatible while leaving the computed metrics unchanged.
            success_threshold_m = float(
                success.get("threshold", {}).get("threshold_m", float("nan"))
            )
            print(
                f"{stage_name}: "
                f"APE_{ape_pose_relation}_rmse={ape_t:.6f}, "
                f"APE_rot_rmse={ape_r:.6f} deg, "
                f"RPE_{rpe_pose_relation}_rmse={rpe_t:.6f}, "
                f"RPE_rot_rmse={rpe_r:.6f} deg, "
                f"RPE_time_1s_trans_rmse={rpe_time_t:.6f}, "
                f"RPE_time_1s_rot_rmse={rpe_time_r:.6f} deg, "
                f"SR_complete_dist={sr_dist_pct:.2f}%, "
                f"SR_local_dist={local_sr_dist_pct:.2f}%, "
                f"valid_APE_{ape_pose_relation}_rmse={valid_ape_t:.6f}, "
                f"valid_RPE_{rpe_pose_relation}_rmse={valid_rpe_t:.6f}"
            )
            if verbose:
                print(
                    f"{stage_name}: "
                    f"pairs={pairs}, time_pairs={time_pairs}, threshold_m={success_threshold_m:g}"
                )

    # Terminal-only mode ends after the normal metric blocks.  This keeps the
    # existing terminal format while skipping all report/visualization output.
    if terminal_only:
        return

    debug_outputs = bool(getattr(args, "debug", False))
    fig2_path, fig_step3_map_path = _plot_stage_alignment_maps(
        plots_dir=plots_dir,
        pos_gt=pos_gt,
        pr_sync=pr_sync,
        pr_corrected=pr_corrected,
        pr_final=pr_final,
        include_debug=debug_outputs,
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
        "step3_selection": dict(step3_choice),
        "user_alert": user_alert,
        "alignment_quality": alignment_quality,
        "rigid_alignability": rigid_alignability,
        "piecewise_diagnostics": piecewise_diag,
        "case_diagnostics": case_diagnostics,
        "failure_diagnosis": failure_diagnosis,
        "input_coverage": input_coverage,
        "sr_reliability": sr_reliability,
        "orientation_diagnostics": orientation_diagnostics,
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
            "debug": debug_outputs,
            "step1_forced_candidate": bool(step1_forced_candidate),
            "step1_force_reason": str(step1_force_reason),
            "ape_pose_relation": str(getattr(args, "ape_pose_relation", "trans_part")),
            "rpe_pose_relation": str(getattr(args, "rpe_pose_relation", "trans_part")),
            "eval_align": requested_eval_align_mode,
            "eval_align_requested_alias": requested_eval_align_alias,
            "eval_align_effective": eval_align_mode,
            "robust_kernel": robust_kernel,
            "robust_kernel_delta_m": (
                None if robust_kernel_delta_m is None else float(robust_kernel_delta_m)
            ),
            "robust_kernel_max_iterations": robust_kernel_max_iterations,
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
            "orientation_unstable": bool(orientation_diagnostics["orientation_unstable"]),
            "orientation_warning": str(orientation_diagnostics["orientation_warning"]),
            "orientation_status": str(orientation_diagnostics["orientation_status"]),
            "orientation_ape_rmse_deg": float(orientation_diagnostics["orientation_ape_rmse_deg"]),
            "orientation_rpe_rmse_deg": float(orientation_diagnostics["orientation_rpe_rmse_deg"]),
            "orientation_rpe_time_1s_rmse_deg": float(
                orientation_diagnostics["orientation_rpe_time_1s_rmse_deg"]
            ),
            "diagnosis_primary": str(case_diagnostics["diagnosis_primary"]),
            "diagnosis_summary": str(case_diagnostics["diagnosis_summary"]),
            "diagnosis_tags": ",".join(str(x) for x in case_diagnostics["diagnosis_tags"]),
            "failure_diagnosis_level": str(failure_diagnosis["failure_diagnosis_level"]),
            "failure_diagnosis_hard_reasons": ",".join(
                str(x) for x in failure_diagnosis["failure_diagnosis_hard_reasons"]
            ),
            "failure_diagnosis_soft_reasons": ",".join(
                str(x) for x in failure_diagnosis["failure_diagnosis_soft_reasons"]
            ),
            "trajectory_diagnosis_level": str(
                failure_diagnosis["trajectory_diagnosis_level"]
            ),
            "trajectory_diagnosis_hard_reasons": ",".join(
                str(x) for x in failure_diagnosis["trajectory_diagnosis_hard_reasons"]
            ),
            "trajectory_diagnosis_soft_reasons": ",".join(
                str(x) for x in failure_diagnosis["trajectory_diagnosis_soft_reasons"]
            ),
            "calibration_confidence": str(failure_diagnosis["calibration_confidence"]),
            "calibration_confidence_reasons": ",".join(
                str(x) for x in failure_diagnosis["calibration_confidence_reasons"]
            ),
            "coverage_status": str(input_coverage["coverage_status"]),
            "temporal_coverage_ratio": float(input_coverage["temporal_coverage_ratio"]),
            "path_coverage_ratio": float(input_coverage["path_coverage_ratio"]),
            "rigid_check_max_path_ratio": float(rigid_check_max_path_ratio),
            "rigid_check_max_bbox_ratio": float(rigid_check_max_bbox_ratio),
            "rigid_check_max_global_local_ratio": float(rigid_check_max_global_local_ratio),
            "rigid_check_max_sim3_gain_ratio": float(rigid_check_max_sim3_gain_ratio),
            "rerun": rerun_info,
            "output_dir": str(run_dir),
        },
    }
    pose_state_csv = write_pose_state_csv(
        run_dir / "pose_states.csv",
        timestamps_s=t_gt,
        stages={
            "gt": (pos_gt, quat_gt),
            "raw": (pr_sync, qr_sync),
            "step2": (pr_corrected, q_step2),
            "step3": (pr_final, q_step3),
        },
    )
    metrics_payload["metadata"]["pose_state_csv"] = str(pose_state_csv)
    metrics_payload["metadata"]["eval_source"] = terminal_eval_source
    metrics_payload["metadata"]["eval_alignment"] = pose_metrics.get("eval_alignment", {}).get("step3", {})
    if str(eval_align_mode).lower() in {"", "none"}:
        interactive_step3_metrics = pose_metrics
    else:
        interactive_step3_metrics = _compute_pose_metrics_by_stage(
            input_coverage=input_coverage,
            t_gt=t_gt,
            pos_gt=pos_gt,
            quat_gt=quat_gt,
            stage_trajs={"step3": {"pos": pr_final, "quat": q_step3}},
            eval_align_mode="none",
            eval_n_to_align=-1,
            eval_project_to_plane=eval_project_to_plane,
            rpe_delta=args.rpe_delta,
            rpe_delta_unit=args.rpe_delta_unit,
            rpe_delta_tol=args.rpe_delta_tol,
            rpe_all_pairs=args.rpe_all_pairs,
            rpe_pairs_from_reference=args.rpe_pairs_from_reference,
            success_threshold_mode=str(getattr(args, "success_threshold_mode", "adaptive_knee")),
            success_threshold_m=float(getattr(args, "success_threshold_m", 10.0)),
            success_threshold_min_m=float(getattr(args, "success_threshold_min_m", 5.0)),
            success_threshold_max_m=float(getattr(args, "success_threshold_max_m", 30.0)),
            success_threshold_trim_percentile=float(getattr(args, "success_threshold_trim_percentile", 95.0)),
            success_global_gate_mode=str(getattr(args, "success_global_gate_mode", "fixed")),
            success_global_gate_m=float(getattr(args, "success_global_gate_m", 30.0)),
            success_global_gate_path_ratio=float(getattr(args, "success_global_gate_path_ratio", 0.05)),
            success_global_gate_min_m=float(getattr(args, "success_global_gate_min_m", 2.0)),
            success_global_gate_max_m=float(getattr(args, "success_global_gate_max_m", 100.0)),
            success_global_gate_percentile=float(getattr(args, "success_global_gate_percentile", 5.0)),
            success_drift_rpe_1s_m=float(getattr(args, "success_drift_rpe_1s_m", 2.0)),
            success_drift_ape_slope_mps=float(getattr(args, "success_drift_ape_slope_mps", 1.0)),
            success_drift_ape_jump_m=float(getattr(args, "success_drift_ape_jump_m", 5.0)),
            success_drift_threshold_mode=str(getattr(args, "success_drift_threshold_mode", "adaptive")),
            t_max_diff=float(getattr(args, "t_max_diff", 0.02)),
            t_offset=float(getattr(args, "t_offset", 0.0)),
            t_start=getattr(args, "t_start", None),
            t_end=getattr(args, "t_end", None),
        )
    _annotate_sr_reliability(
        interactive_step3_metrics,
        stage="step3",
        orientation=orientation_diagnostics,
        ape_relation=ape_pose_relation,
        rpe_relation=rpe_pose_relation,
        input_coverage=input_coverage,
    )
    interactive_trajectory_views: dict[str, dict] = {}
    interactive_view_metrics: dict[str, dict] = {
        "step3": _interactive_view_metric_summary(
            interactive_step3_metrics,
            stage="step3",
            ape_relation=ape_pose_relation,
            rpe_relation=rpe_pose_relation,
        )
    }
    default_interactive_view = _default_interactive_view(requested_eval_align_mode)

    def _json_float_or_none(value):
        try:
            value_f = float(value)
        except Exception:
            return None
        return value_f if np.isfinite(value_f) else None

    for view_key, align_mode, label in _interactive_eval_view_specs(requested_eval_align_mode):
        try:
            if align_mode == eval_align_mode or (
                align_mode == "posyaw" and requested_eval_align_mode in {"posyaw", "epa_posyaw"}
            ):
                view_pose_metrics = pose_metrics
            else:
                view_pose_metrics = _compute_pose_metrics_by_stage(
                    input_coverage=input_coverage,
                    t_gt=t_gt,
                    pos_gt=pos_gt,
                    quat_gt=quat_gt,
                    stage_trajs={"step3": {"pos": pr_final, "quat": q_step3}},
                    eval_align_mode=align_mode,
                    eval_n_to_align=-1,
                    eval_project_to_plane=eval_project_to_plane,
                    rpe_delta=args.rpe_delta,
                    rpe_delta_unit=args.rpe_delta_unit,
                    rpe_delta_tol=args.rpe_delta_tol,
                    rpe_all_pairs=args.rpe_all_pairs,
                    rpe_pairs_from_reference=args.rpe_pairs_from_reference,
                    success_threshold_mode=str(getattr(args, "success_threshold_mode", "adaptive_knee")),
                    success_threshold_m=float(getattr(args, "success_threshold_m", 10.0)),
                    success_threshold_min_m=float(getattr(args, "success_threshold_min_m", 5.0)),
                    success_threshold_max_m=float(getattr(args, "success_threshold_max_m", 30.0)),
                    success_threshold_trim_percentile=float(getattr(args, "success_threshold_trim_percentile", 95.0)),
                    success_global_gate_mode=str(getattr(args, "success_global_gate_mode", "fixed")),
                    success_global_gate_m=float(getattr(args, "success_global_gate_m", 30.0)),
                    success_global_gate_path_ratio=float(getattr(args, "success_global_gate_path_ratio", 0.05)),
                    success_global_gate_min_m=float(getattr(args, "success_global_gate_min_m", 2.0)),
                    success_global_gate_max_m=float(getattr(args, "success_global_gate_max_m", 100.0)),
                    success_global_gate_percentile=float(getattr(args, "success_global_gate_percentile", 5.0)),
                    success_drift_rpe_1s_m=float(getattr(args, "success_drift_rpe_1s_m", 2.0)),
                    success_drift_ape_slope_mps=float(getattr(args, "success_drift_ape_slope_mps", 1.0)),
                    success_drift_ape_jump_m=float(getattr(args, "success_drift_ape_jump_m", 5.0)),
                    success_drift_threshold_mode=str(getattr(args, "success_drift_threshold_mode", "adaptive")),
                    t_max_diff=float(getattr(args, "t_max_diff", 0.02)),
                    t_offset=float(getattr(args, "t_offset", 0.0)),
                    t_start=getattr(args, "t_start", None),
                    t_end=getattr(args, "t_end", None),
                )
            _annotate_sr_reliability(
                view_pose_metrics,
                stage="step3",
                orientation=orientation_diagnostics,
                ape_relation=ape_pose_relation,
                rpe_relation=rpe_pose_relation,
                input_coverage=input_coverage,
            )
            view_info = view_pose_metrics.get("eval_alignment", {}).get("step3", {})
            try:
                view_pos, view_quat = _apply_eval_alignment_from_info(pr_final, q_step3, view_info)
            except Exception:
                view_pos, view_quat, view_info = align_for_eval_with_info(
                    pos_ref=pos_gt,
                    quat_ref=quat_gt,
                    pos_est=pr_final,
                    quat_est=q_step3,
                    mode=align_mode,
                    n_to_align=-1,
                    t_ref=t_gt,
                )
            interactive_trajectory_views[view_key] = {
                "label": label,
                "pos": view_pos,
                "quat": view_quat,
                "meta": {
                    "align_mode": align_mode,
                    "align_scale": _json_float_or_none(view_info.get("align_scale", np.nan)),
                    "solver": str(view_info.get("sim3_solver", "ov_position_only_umeyama")),
                    "anchor_samples": int(view_info.get("sim3_anchor_samples", 0) or 0),
                    "anchor_status": str(view_info.get("sim3_anchor_status", "")),
                    "consensus_status": str(view_info.get("sim3_consensus_status", "")),
                    "confidence": str(view_info.get("sim3_confidence", "")),
                    "consensus_count": int(view_info.get("sim3_consensus_count", 0) or 0),
                    "candidate_reliable_count": int(view_info.get("sim3_candidate_reliable_count", 0) or 0),
                    "stable_anchor_used": bool(view_info.get("sim3_stable_anchor_used", False)),
                    "fallback_used": bool(view_info.get("sim3_robust_fallback_used", False)),
                    "fallback_reason": str(view_info.get("sim3_robust_fallback_reason", "")),
                },
            }
            interactive_view_metrics[view_key] = _interactive_view_metric_summary(
                view_pose_metrics,
                stage="step3",
                ape_relation=ape_pose_relation,
                rpe_relation=rpe_pose_relation,
            )
        except Exception as exc:
            interactive_trajectory_views[view_key] = {
                "label": f"{label} unavailable",
                "pos": pr_final,
                "quat": q_step3,
                "meta": {
                    "align_mode": align_mode,
                    "solver": "unavailable",
                    "error": str(exc),
                },
            }
            interactive_view_metrics[view_key] = {
                "case_status": "unavailable",
                "error": str(exc),
            }
    output_info = _write_outputs(
        run_dir=run_dir,
        plots_dir=plots_dir,
        metrics_payload=metrics_payload,
        args=args,
        interactive_payload={
            "title": f"EPA Interactive Report: {run_dir.name}",
            "pos_gt": pos_gt,
            "pr_sync": pr_sync,
            "pr_corrected": pr_corrected,
            "pr_final": pr_final,
            "trajectory_views": interactive_trajectory_views,
            "trajectory_view_metrics": interactive_view_metrics,
            "default_trajectory_view": default_interactive_view,
            "timestamps_s": t_gt,
            "time_alignment": {
                "t_uniform": t_uniform,
                "sig_gt": sig_gt,
                "sig_est": sig_est,
                "corr": corr,
                "lags": lags,
                "dt_resample": dt_resample,
                "calculated_offset": calculated_offset,
            },
        },
    )
    bundle_path = output_info["bundle_path"]
    report_zh_path = output_info["report_zh_path"]
    report_en_path = output_info["report_en_path"]

    print("\n--- OUTPUT FILES ---")
    if verbose:
        print(f"Saved figure: {fig_corr_path}")
        print(f"Saved figure: {fig1_path}")
        if fig2_path is not None:
            print(f"Saved figure: {fig2_path}")
        print(f"Saved figure: {fig_step3_map_path}")
    print(f"Saved outputs: {run_dir}")
    print(f"Saved plots: {plots_dir}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    if verbose:
        print(f"Saved metrics: {run_dir / 'metrics_summary.csv'}")
        print(f"Saved report: {report_zh_path}")
        print(f"Saved report: {report_en_path}")
    if bundle_path is not None:
        print(f"Saved results: {bundle_path}")
