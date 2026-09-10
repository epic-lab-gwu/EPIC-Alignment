from __future__ import annotations

import numpy as np

from .diagnostics import (
    _build_user_alert,
    _compute_alignment_quality,
    _compute_piecewise_diagnostics,
    _compute_rigid_alignability,
)
from .evaluation import (
    compute_ape,
    compute_rpe,
    compute_valid_segment_metrics,
)
from ..metric_cli_common import align_for_eval_with_info, cum_distance, project_to_plane


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
    quality_threshold_mode: str,
    quality_good_rmse_ratio: float,
    quality_partial_rmse_ratio: float,
    quality_critical_rmse_ratio: float,
    quality_good_rmse_floor_m: float,
    quality_partial_rmse_floor_m: float,
    quality_critical_rmse_floor_m: float,
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
        threshold_mode=quality_threshold_mode,
        good_rmse_ratio=quality_good_rmse_ratio,
        partial_rmse_ratio=quality_partial_rmse_ratio,
        critical_rmse_ratio=quality_critical_rmse_ratio,
        good_rmse_floor_m=quality_good_rmse_floor_m,
        partial_rmse_floor_m=quality_partial_rmse_floor_m,
        critical_rmse_floor_m=quality_critical_rmse_floor_m,
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
        alignment_quality=alignment_quality,
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
    eval_align_indices_by_stage: dict[str, np.ndarray] | None = None,
    eval_align_step3_only: bool = False,
    rpe_delta,
    rpe_delta_unit,
    rpe_delta_tol,
    rpe_all_pairs,
    rpe_pairs_from_reference,
    success_threshold_mode: str,
    success_threshold_m: float,
    success_threshold_min_m: float,
    success_threshold_max_m: float,
    success_threshold_trim_percentile: float,
    success_global_gate_mode: str,
    success_global_gate_m: float,
    success_global_gate_path_ratio: float,
    success_global_gate_min_m: float,
    success_global_gate_max_m: float,
    success_global_gate_percentile: float,
    success_drift_rpe_1s_m: float,
    success_drift_ape_slope_mps: float,
    success_drift_ape_jump_m: float,
    success_drift_threshold_mode: str,
    t_max_diff: float,
    t_offset: float,
    t_start,
    t_end,
    input_coverage=None,
):
    ape_metrics_by_stage = {}
    rpe_metrics_by_stage = {}
    rpe_time_1s_by_stage = {}
    valid_metrics_by_stage = {}
    eval_alignment_by_stage = {}
    seconds_from_start = np.asarray(t_gt, dtype=float) - float(t_gt[0])
    distances_from_start = cum_distance(pos_gt)
    ref_eval_pos, ref_eval_quat = project_to_plane(pos_gt, quat_gt, plane=eval_project_to_plane)

    for stage_name, stage_data in stage_trajs.items():
        stage_eval_align_mode = eval_align_mode
        if bool(eval_align_step3_only) and stage_name != "step3":
            stage_eval_align_mode = "none"
        stage_align_indices = None
        if eval_align_indices_by_stage is not None:
            stage_align_indices = eval_align_indices_by_stage.get(stage_name)
        eval_pos, eval_quat, eval_align_info = align_for_eval_with_info(
            pos_ref=pos_gt,
            quat_ref=quat_gt,
            pos_est=stage_data["pos"],
            quat_est=stage_data["quat"],
            mode=stage_eval_align_mode,
            n_to_align=eval_n_to_align,
            t_ref=t_gt,
            align_indices=stage_align_indices,
        )
        eval_alignment_by_stage[stage_name] = eval_align_info
        est_eval_pos, est_eval_quat = project_to_plane(
            eval_pos, eval_quat, plane=eval_project_to_plane
        )
        ape_metrics_by_stage[stage_name] = compute_ape(
            pos_ref=ref_eval_pos,
            quat_ref=ref_eval_quat,
            pos_est=est_eval_pos,
            quat_est=est_eval_quat,
            include_raw=True,
        )
        rpe_metrics_by_stage[stage_name] = compute_rpe(
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
        rpe_time_1s_by_stage[stage_name] = compute_rpe(
            pos_ref=ref_eval_pos,
            quat_ref=ref_eval_quat,
            pos_est=est_eval_pos,
            quat_est=est_eval_quat,
            delta=1.0,
            delta_unit="s",
            max_pairs=0,
            rel_delta_tol=0.1,
            all_pairs=True,
            pairs_from_reference=True,
            timestamps=t_gt,
            include_raw=True,
        )
        ape_metrics_by_stage[stage_name]["_x_axis"]["seconds_from_start"] = seconds_from_start
        ape_metrics_by_stage[stage_name]["_x_axis"]["distances_from_start"] = distances_from_start
        delta_ids = rpe_metrics_by_stage[stage_name]["_x_axis"]["delta_ids"].astype(int)
        valid = (delta_ids >= 0) & (delta_ids < seconds_from_start.size)
        rpe_metrics_by_stage[stage_name]["_x_axis"]["seconds_from_start"] = seconds_from_start[
            delta_ids[valid]
        ]
        rpe_metrics_by_stage[stage_name]["_x_axis"]["distances_from_start"] = distances_from_start[
            delta_ids[valid]
        ]
        time_delta_ids = rpe_time_1s_by_stage[stage_name]["_x_axis"]["delta_ids"].astype(int)
        valid_time = (time_delta_ids >= 0) & (time_delta_ids < seconds_from_start.size)
        rpe_time_1s_by_stage[stage_name]["_x_axis"]["seconds_from_start"] = seconds_from_start[
            time_delta_ids[valid_time]
        ]
        rpe_time_1s_by_stage[stage_name]["_x_axis"]["distances_from_start"] = distances_from_start[
            time_delta_ids[valid_time]
        ]
        valid_metrics_by_stage[stage_name] = compute_valid_segment_metrics(
            timestamps=t_gt, pos_ref=ref_eval_pos, quat_ref=ref_eval_quat,
            pos_est=est_eval_pos, quat_est=est_eval_quat, input_coverage=input_coverage,
            ape_block=ape_metrics_by_stage[stage_name],
            rpe_block=rpe_metrics_by_stage[stage_name],
            rpe_time_1s_block=rpe_time_1s_by_stage[stage_name],
            include_raw=True,
        )
        if "sim3" in str(eval_align_mode).lower():
            valid_metrics_by_stage[stage_name]["success"]["sim3_sr_distance_raw"] = (
                valid_metrics_by_stage[stage_name]["success"].get("success_rate_distance")
            )
            valid_metrics_by_stage[stage_name]["success"]["sim3_sr_time_raw"] = (
                valid_metrics_by_stage[stage_name]["success"].get("success_rate_time")
            )
            valid_metrics_by_stage[stage_name]["success"]["sim3_sr_reliable"] = bool(
                eval_align_info.get("sim3_reliable", True)
            )
            if eval_align_info.get("sim3_scale_severe"):
                valid_metrics_by_stage[stage_name]["success"]["sim3_scale_warning"] = str(
                    eval_align_info.get("sim3_warning", "")
                )

    return {
        "ape": ape_metrics_by_stage,
        "rpe": rpe_metrics_by_stage,
        "rpe_time_1s": rpe_time_1s_by_stage,
        "valid_segment": valid_metrics_by_stage,
        "eval_alignment": eval_alignment_by_stage,
        "rpe_config": {
            "delta": rpe_delta,
            "delta_unit": rpe_delta_unit,
            "delta_tol": rpe_delta_tol,
            "all_pairs": bool(rpe_all_pairs),
            "pairs_from_reference": bool(rpe_pairs_from_reference),
        },
        "rpe_time_1s_config": {
            "delta": 1.0,
            "delta_unit": "s",
            "delta_tol": 0.1,
            "all_pairs": True,
            "pairs_from_reference": True,
        },
        "valid_segment_config": {
            "policy": "rpe_1s_motion_relative",
            "relative_ratio": 3.0,
            "small_translation_m": 0.1,
            "small_rotation_deg": 1.0,
            "absolute_translation_m": 0.3,
            "absolute_rotation_deg": 3.0,
            "unscored_interval_policy": "sequential_rpe_fallback",
            "valid_only_realign_below_sr": 0.5,
            "success_rate_primary": "distance",
            "fail_definition": "An interval fails if its 1s RPE checks fail, or if no 1s pair covers it and its consecutive-pose RPE fails.",
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
