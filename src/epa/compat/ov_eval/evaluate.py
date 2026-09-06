from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.alignment.modes import public_align_mode, resolve_metric_eval_align_mode
from epa.core.evaluation import (
    compute_ape,
    compute_path_length,
    compute_rpe,
    compute_valid_segment_metrics,
    resolve_success_threshold,
)
from epa.core.diagnostics import _compute_input_coverage_diagnostics
from epa.core.math_utils import compute_error_statistics
from epa.core.solve_eval import _prepare_solve_eval_trajectories
from epa.metric_cli_common import align_for_eval_with_info, project_to_plane
from epa.core.time_sync import _run_time_alignment
from epa.core.trajectory_alignment import _solve_extrinsic_and_world_alignment
from epa.ov_align import associate_est_gt as _associate_est_gt
from epa.ov_io import load_pose_file as _load_pose_file
from epa.ov_report import fmt_num as _fmt

from .constants import (
    _DEFAULT_EPA_DOWNSAMPLE_HZ,
    _DEFAULT_EPA_DT_RESAMPLE,
    _DEFAULT_EPA_OFFSET_MIN_MATCH_RATIO,
    _DEFAULT_EPA_QUAT_INTERP,
    _STRICT_ASSOCIATION_GAP_MIN_JUMP_M,
    _STRICT_ASSOCIATION_GAP_MIN_SPEED_MPS,
    _STRICT_ASSOCIATION_GAP_MOTION_FACTOR,
    _STRICT_ASSOCIATION_GAP_SPEED_FACTOR,
    _STRICT_ASSOCIATION_HIGH_EST_MATCH_RATIO,
    _STRICT_ASSOCIATION_MAX_GAP_FACTOR,
    _STRICT_ASSOCIATION_MIN_EST_MATCH_RATIO,
    _STRICT_ASSOCIATION_MIN_EST_SPAN_RATIO,
    _ORIENTATION_WARNING_APE_RMSE_DEG,
    _ORIENTATION_WARNING_MIN_SR,
    _ORIENTATION_WARNING_RPE_RMSE_DEG,
)


def _median_positive_dt_s(timestamps: np.ndarray) -> float:
    tvals = np.asarray(timestamps, dtype=float).reshape(-1)
    if tvals.size < 2:
        return float("nan")
    diffs = np.diff(tvals)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return float("nan")
    return float(np.nanmedian(diffs))


def _duration_s(timestamps: np.ndarray) -> float:
    tvals = np.asarray(timestamps, dtype=float).reshape(-1)
    tvals = tvals[np.isfinite(tvals)]
    if tvals.size < 2:
        return 0.0
    return max(0.0, float(np.max(tvals) - np.min(tvals)))


def _max_positive_gap_s(timestamps: np.ndarray) -> float:
    tvals = np.asarray(timestamps, dtype=float).reshape(-1)
    tvals = tvals[np.isfinite(tvals)]
    if tvals.size < 2:
        return float("nan")
    diffs = np.diff(np.sort(tvals))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return float("nan")
    return float(np.max(diffs))


def _timeline_decision_for_strict_association(
    *,
    t_est: np.ndarray,
    t_gt: np.ndarray,
    t_matched: np.ndarray,
    i_est: np.ndarray,
    sparse_matched: int,
    pos_est_matched: np.ndarray | None = None,
) -> dict[str, object]:
    matched_count = int(sparse_matched)
    est_count = int(np.asarray(t_est).reshape(-1).size)
    gt_count = int(np.asarray(t_gt).reshape(-1).size)
    est_t = np.asarray(t_est, dtype=float).reshape(-1)
    matched_t = np.asarray(t_matched, dtype=float).reshape(-1)
    matched_ids = np.asarray(i_est, dtype=int).reshape(-1)
    est_duration = _duration_s(est_t)
    gt_duration = _duration_s(np.asarray(t_gt, dtype=float))
    matched_duration = _duration_s(matched_t)
    est_match_ratio = float(matched_count) / max(float(est_count), 1.0)
    gt_match_ratio = float(matched_count) / max(float(gt_count), 1.0)
    matched_span_ratio_est = matched_duration / max(est_duration, 1e-12)
    matched_span_ratio_gt = matched_duration / max(gt_duration, 1e-12)
    est_median_dt_s = _median_positive_dt_s(est_t)
    gt_median_dt_s = _median_positive_dt_s(np.asarray(t_gt, dtype=float))
    matched_max_gap_s = _max_positive_gap_s(matched_t)
    full_est_max_gap_s = _max_positive_gap_s(est_t)
    gap_limit_s = float("nan")
    if np.isfinite(est_median_dt_s):
        gap_limit_s = max(float(_STRICT_ASSOCIATION_MAX_GAP_FACTOR) * est_median_dt_s, 2.0 * float(est_median_dt_s))
    has_large_matched_gap = bool(
        np.isfinite(matched_max_gap_s)
        and np.isfinite(gap_limit_s)
        and matched_max_gap_s > gap_limit_s
    )
    matched_pos = (
        np.asarray(pos_est_matched, dtype=float)
        if pos_est_matched is not None
        else np.empty((0, 3), dtype=float)
    )
    gap_jump_m = float("nan")
    gap_speed_mps = float("nan")
    nominal_step_m = float("nan")
    nominal_speed_mps = float("nan")
    has_large_motion_gap = False
    matched_dt = np.diff(matched_t) if matched_t.size > 1 else np.asarray([], dtype=float)
    if matched_pos.ndim == 2 and matched_pos.shape[0] == matched_t.size and matched_t.size > 1:
        matched_step = np.linalg.norm(np.diff(matched_pos, axis=0), axis=1)
        finite_steps = matched_step[np.isfinite(matched_step)]
        if finite_steps.size > 0:
            nominal_step_m = float(np.nanmedian(finite_steps))
        speeds = matched_step / np.maximum(matched_dt, 1e-12)
        finite_speeds = speeds[np.isfinite(speeds)]
        if finite_speeds.size > 0:
            nominal_speed_mps = float(np.nanmedian(finite_speeds))
        large_gap_mask = (
            np.isfinite(matched_dt)
            & np.isfinite(matched_step)
            & (matched_dt > gap_limit_s)
        )
        if np.any(large_gap_mask):
            gap_steps = matched_step[large_gap_mask]
            gap_speeds = speeds[large_gap_mask]
            gap_jump_m = float(np.nanmax(gap_steps))
            finite_gap_speeds = gap_speeds[np.isfinite(gap_speeds)]
            if finite_gap_speeds.size > 0:
                gap_speed_mps = float(np.nanmax(finite_gap_speeds))
            motion_jump_limit_m = max(
                float(_STRICT_ASSOCIATION_GAP_MIN_JUMP_M),
                float(_STRICT_ASSOCIATION_GAP_MOTION_FACTOR)
                * max(nominal_step_m, 1e-12),
            )
            motion_speed_limit_mps = max(
                float(_STRICT_ASSOCIATION_GAP_MIN_SPEED_MPS),
                float(_STRICT_ASSOCIATION_GAP_SPEED_FACTOR)
                * max(nominal_speed_mps, 1e-12),
            )
            has_large_motion_gap = bool(
                np.isfinite(gap_jump_m)
                and np.isfinite(gap_speed_mps)
                and gap_jump_m > motion_jump_limit_m
                and gap_speed_mps > motion_speed_limit_mps
            )
    has_large_unmatched_gap = bool(
        has_large_matched_gap
        and (
            not np.isfinite(full_est_max_gap_s)
            or not np.isfinite(est_median_dt_s)
            or matched_max_gap_s > max(full_est_max_gap_s * 1.5, gap_limit_s)
        )
    )
    matched_est_span_ratio = 0.0
    if est_count > 1 and matched_ids.size > 0:
        matched_est_span_ratio = float(int(np.max(matched_ids)) - int(np.min(matched_ids)) + 1) / float(est_count)
    low_rate_estimate = bool(np.isfinite(est_median_dt_s) and est_median_dt_s >= 0.25)
    low_rate_gt = bool(np.isfinite(gt_median_dt_s) and gt_median_dt_s >= 0.25)
    enough_pairs = matched_count >= 3
    representative_count = est_match_ratio >= float(_STRICT_ASSOCIATION_MIN_EST_MATCH_RATIO)
    near_complete_count = est_match_ratio >= float(_STRICT_ASSOCIATION_HIGH_EST_MATCH_RATIO)
    representative_span = (
        matched_span_ratio_est >= float(_STRICT_ASSOCIATION_MIN_EST_SPAN_RATIO)
        or matched_est_span_ratio >= float(_STRICT_ASSOCIATION_MIN_EST_SPAN_RATIO)
    )
    use_strict = bool(
        enough_pairs
        and representative_count
        and (near_complete_count or representative_span)
        and not has_large_unmatched_gap
        and not has_large_motion_gap
    )
    reasons: list[str] = []
    if not enough_pairs:
        reasons.append("strict association has fewer than 3 matched pairs")
    if not representative_count:
        reasons.append("strict association does not cover enough estimate samples")
    if representative_count and not (near_complete_count or representative_span):
        reasons.append("strict matches cover estimate count but not estimate time span")
    if has_large_matched_gap:
        if has_large_motion_gap:
            reasons.append("strict matches contain a large timestamp gap with abnormal estimate motion")
        elif has_large_unmatched_gap:
            reasons.append("strict matches contain a large unmatched timestamp gap")
    if use_strict:
        if has_large_matched_gap:
            reasons.append("strict association covers the estimate timeline; timestamp gap motion is stable")
        else:
            reasons.append("strict association covers the estimate timeline without large gaps")

    return {
        "strict_association_timeline_used": use_strict,
        "timeline_policy_reason": "; ".join(reasons) if reasons else "strict association diagnostics unavailable",
        "sparse_matched": matched_count,
        "sparse_est_match_ratio": est_match_ratio,
        "sparse_gt_match_ratio": gt_match_ratio,
        "strict_est_span_ratio": matched_span_ratio_est,
        "strict_gt_span_ratio": matched_span_ratio_gt,
        "strict_est_index_span_ratio": matched_est_span_ratio,
        "strict_max_gap_s": matched_max_gap_s,
        "strict_full_est_max_gap_s": full_est_max_gap_s,
        "strict_gap_limit_s": gap_limit_s,
        "strict_has_large_gap": has_large_matched_gap,
        "strict_has_large_unmatched_gap": has_large_unmatched_gap,
        "strict_has_large_motion_gap": has_large_motion_gap,
        "strict_gap_jump_m": gap_jump_m,
        "strict_gap_speed_mps": gap_speed_mps,
        "strict_nominal_step_m": nominal_step_m,
        "strict_nominal_speed_mps": nominal_speed_mps,
        "estimate_median_dt_s": est_median_dt_s,
        "gt_median_dt_s": gt_median_dt_s,
        "low_rate_estimate": low_rate_estimate,
        "low_rate_gt": low_rate_gt,
        "strict_min_est_match_ratio": float(_STRICT_ASSOCIATION_MIN_EST_MATCH_RATIO),
        "strict_high_est_match_ratio": float(_STRICT_ASSOCIATION_HIGH_EST_MATCH_RATIO),
        "strict_min_est_span_ratio": float(_STRICT_ASSOCIATION_MIN_EST_SPAN_RATIO),
        "strict_max_gap_factor": float(_STRICT_ASSOCIATION_MAX_GAP_FACTOR),
        "strict_gap_motion_factor": float(_STRICT_ASSOCIATION_GAP_MOTION_FACTOR),
        "strict_gap_speed_factor": float(_STRICT_ASSOCIATION_GAP_SPEED_FACTOR),
        "strict_gap_min_jump_m": float(_STRICT_ASSOCIATION_GAP_MIN_JUMP_M),
        "strict_gap_min_speed_mps": float(_STRICT_ASSOCIATION_GAP_MIN_SPEED_MPS),
    }


def _public_align_mode(raw_mode: str) -> str:
    mode = resolve_metric_eval_align_mode(
        raw_mode,
        default="none",
        collapse_sim3_aliases=True,
        legacy_origin=False,
    )
    if mode == "none":
        return "none"
    return public_align_mode(mode, default=mode)


def _associate_or_resample_ov_style(
    *,
    t_est: np.ndarray,
    p_est: np.ndarray,
    q_est: np.ndarray,
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    q_gt: np.ndarray,
    max_diff: float,
    downsample_hz: float,
    quat_interp: str,
    allow_resampled_fallback: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    sparse_matched = 0
    sparse_assoc_failed = False
    sparse_assoc_failure_reason = ""
    strict_decision: dict[str, object] | None = None
    try:
        _, p_est_m, q_est_m, t_gt_m, p_gt_m, q_gt_m, i_est_m = _associate_est_gt(
            t_est=t_est,
            p_est=p_est,
            q_est=q_est,
            t_gt=t_gt,
            p_gt=p_gt,
            q_gt=q_gt,
            max_diff=float(max_diff),
            offset=0.0,
        )
        sparse_matched = int(t_gt_m.size)
        strict_decision = _timeline_decision_for_strict_association(
            t_est=t_est,
            t_gt=t_gt,
            t_matched=t_gt_m,
            i_est=i_est_m,
            sparse_matched=sparse_matched,
            pos_est_matched=p_est_m,
        )
        use_strict_timeline = bool(strict_decision["strict_association_timeline_used"])
        if (not bool(allow_resampled_fallback)) or use_strict_timeline:
            strict_decision["strict_association_timeline_used"] = bool(use_strict_timeline)
            strict_decision["auto_sparse_low_rate_estimate_used"] = bool(use_strict_timeline)
            return (
                t_gt_m,
                p_gt_m,
                q_gt_m,
                p_est_m,
                q_est_m,
                {
                    "timeline_policy": "strict_timestamp_association",
                    "sparse_association_failed": False,
                    "sparse_association_failure_reason": "",
                    "resampled_fallback_used": False,
                    "dense_timeline_used": False,
                    **strict_decision,
                },
            )
    except ValueError as exc:
        strict_error = exc
        sparse_assoc_failed = True
        sparse_assoc_failure_reason = str(exc)
        if not bool(allow_resampled_fallback):
            raise strict_error

    solve_eval = _prepare_solve_eval_trajectories(
        t_gt=t_gt,
        pos_gt=p_gt,
        quat_gt=q_gt,
        t_est=t_est,
        pos_est=p_est,
        quat_est=q_est,
        calculated_offset=0.0,
        downsample_hz=float(downsample_hz),
        quat_interp=str(quat_interp),
    )
    overlap_info = solve_eval.get("overlap_info", {})
    overlap_samples = int(overlap_info.get("overlap_samples", 0) or 0)
    t_gt_m = np.asarray(solve_eval["t_gt"], dtype=float)
    if t_gt_m.size < 3 or overlap_samples < 3:
        if sparse_assoc_failed:
            raise strict_error
        raise ValueError(
            "Dense overlap produced fewer than 3 samples; "
            f"overlap_samples={overlap_samples}."
        )

    fallback_decision = strict_decision or _timeline_decision_for_strict_association(
        t_est=t_est,
        t_gt=t_gt,
        t_matched=np.asarray([], dtype=float),
        i_est=np.asarray([], dtype=int),
        sparse_matched=sparse_matched,
        pos_est_matched=None,
    )
    fallback_decision["timeline_policy_reason"] = (
        sparse_assoc_failure_reason
        if sparse_assoc_failed
        else str(fallback_decision["timeline_policy_reason"])
    )
    return (
        t_gt_m,
        np.asarray(solve_eval["pos_gt"], dtype=float),
        np.asarray(solve_eval["quat_gt"], dtype=float),
        np.asarray(solve_eval["pr_sync"], dtype=float),
        np.asarray(solve_eval["qr_sync"], dtype=float),
        {
            "timeline_policy": "gt_resampled_association_fallback"
            if sparse_assoc_failed
            else "dense_overlap",
            "sparse_association_failed": bool(sparse_assoc_failed),
            "sparse_association_failure_reason": sparse_assoc_failure_reason,
            "resampled_fallback_used": bool(sparse_assoc_failed),
            "auto_sparse_low_rate_estimate_used": False,
            "dense_timeline_used": True,
            "overlap_info": overlap_info,
            "downsample_info": solve_eval.get("downsample_info", {}),
            **fallback_decision,
        },
    )


def _run_eval_time_alignment(
    *,
    t_gt: np.ndarray,
    q_gt: np.ndarray,
    t_est: np.ndarray,
    q_est: np.ndarray,
    dt_resample: float,
    offset_min_match_ratio: float,
    max_diff: float,
    disable_time_offset_calibration: bool,
    verbose: bool,
) -> dict:
    if disable_time_offset_calibration:
        return {
            "calculated_offset": 0.0,
            "time_metrics": {
                "offset_est_s": 0.0,
                "evo_t_offset_used_s": 0.0,
                "time_offset_calibration_disabled": 1.0,
            },
        }
    output_context = (
        contextlib.nullcontext()
        if verbose
        else contextlib.redirect_stdout(io.StringIO())
    )
    with output_context:
        return _run_time_alignment(
            t_gt=t_gt,
            quat_gt=q_gt,
            t_est=t_est,
            quat_est=q_est,
            dt_resample=float(dt_resample),
            offset_search_window_s=0.0,
            offset_min_match_ratio=float(offset_min_match_ratio),
            evo_match_max_diff_s=float(max_diff),
            artificial_offset_s=None,
            disable_time_offset_calibration=disable_time_offset_calibration,
        )


def _evaluate_pair_ov_style(
    file_gt: Path,
    file_est: Path,
    align_mode: str,
    max_diff: float,
    *,
    downsample_hz: float = _DEFAULT_EPA_DOWNSAMPLE_HZ,
    quat_interp: str = _DEFAULT_EPA_QUAT_INTERP,
    allow_resampled_fallback: bool = True,
    disable_extrinsic_calibration: bool = False,
    calibrate_time: bool = False,
    dt_resample: float = _DEFAULT_EPA_DT_RESAMPLE,
    offset_min_match_ratio: float = _DEFAULT_EPA_OFFSET_MIN_MATCH_RATIO,
    disable_time_offset_calibration: bool = False,
    verbose: bool = False,
) -> dict:
    t_gt, p_gt, q_gt = _load_pose_file(file_gt)
    t_est, p_est, q_est = _load_pose_file(file_est)

    len_gt = (
        float(np.sum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))) if p_gt.shape[0] > 1 else 0.0
    )
    len_est = (
        float(np.sum(np.linalg.norm(np.diff(p_est, axis=0), axis=1))) if p_est.shape[0] > 1 else 0.0
    )
    ratio = len_est / (len_gt + 1e-12)
    step1 = (
        _run_eval_time_alignment(
            t_gt=t_gt,
            q_gt=q_gt,
            t_est=t_est,
            q_est=q_est,
            dt_resample=dt_resample,
            offset_min_match_ratio=offset_min_match_ratio,
            max_diff=max_diff,
            disable_time_offset_calibration=disable_time_offset_calibration,
            verbose=verbose,
        )
        if calibrate_time
        else {"calculated_offset": 0.0, "time_metrics": {}}
    )
    calculated_offset = float(step1["calculated_offset"])
    input_coverage = _compute_input_coverage_diagnostics(
        t_ref=t_gt,
        pos_ref=p_gt,
        t_est=t_est,
        offset_est_s=calculated_offset,
    )

    t_gt_m, p_gt_m, q_gt_m, p_est_m, q_est_m, timeline_info = _associate_or_resample_ov_style(
        t_est=t_est - calculated_offset,
        p_est=p_est,
        q_est=q_est,
        t_gt=t_gt,
        p_gt=p_gt,
        q_gt=q_gt,
        max_diff=float(max_diff),
        downsample_hz=float(downsample_hz),
        quat_interp=str(quat_interp),
        allow_resampled_fallback=bool(allow_resampled_fallback),
    )

    requested_align_mode = str(align_mode).lower()
    epa_align_mode = "sim3" if requested_align_mode == "epa_sim3" else requested_align_mode
    p_est_aligned, q_est_aligned, align_info = align_for_eval_with_info(
        pos_ref=p_gt_m,
        quat_ref=q_gt_m,
        pos_est=p_est_m,
        quat_est=q_est_m,
        mode=epa_align_mode,
        n_to_align=-1,
        disable_extrinsic_calibration=disable_extrinsic_calibration,
    )
    align_info.update(timeline_info)
    align_info["requested_align_mode"] = requested_align_mode
    align_info["eval_source"] = "epa_eval_align"
    if calibrate_time:
        align_info["time_offset_calibration_disabled"] = bool(disable_time_offset_calibration)

    ape3 = compute_ape(
        pos_ref=p_gt_m,
        quat_ref=q_gt_m,
        pos_est=p_est_aligned,
        quat_est=q_est_aligned,
        include_raw=True,
    )

    p_gt_xy, q_gt_xy = project_to_plane(p_gt_m, q_gt_m, "xy")
    p_est_xy, q_est_xy = project_to_plane(p_est_aligned, q_est_aligned, "xy")
    ape2 = compute_ape(
        pos_ref=p_gt_xy,
        quat_ref=q_gt_xy,
        pos_est=p_est_xy,
        quat_est=q_est_xy,
        include_raw=True,
    )

    return {
        "matched": int(t_gt_m.size),
        "length_ratio": float(ratio),
        "length_gt": float(len_gt),
        "length_est": float(len_est),
        "input_coverage": input_coverage,
        "calculated_offset": calculated_offset,
        "time_metrics": dict(step1["time_metrics"]),
        "gt_t": t_gt_m,
        "gt_pos": p_gt_m,
        "gt_quat": q_gt_m,
        "est_pos": p_est_aligned,
        "est_quat": q_est_aligned,
        "ate3_ori": dict(ape3["rotation_angle_deg"]),
        "ate3_pos": dict(ape3["translation_part"]),
        "ate2_ori": dict(ape2["rotation_angle_deg"]),
        "ate2_pos": dict(ape2["translation_part"]),
        "eval_alignment": align_info,
        "eval_source": "epa_eval_align",
    }


def _evaluate_pair_epa_step3(
    file_gt: Path,
    file_est: Path,
    max_diff: float,
    *,
    eval_align_mode: str = "none",
    dt_resample: float = _DEFAULT_EPA_DT_RESAMPLE,
    offset_min_match_ratio: float = _DEFAULT_EPA_OFFSET_MIN_MATCH_RATIO,
    downsample_hz: float = _DEFAULT_EPA_DOWNSAMPLE_HZ,
    quat_interp: str = _DEFAULT_EPA_QUAT_INTERP,
    verbose: bool = False,
    allow_resampled_fallback: bool = True,
    disable_time_offset_calibration: bool = False,
    disable_extrinsic_calibration: bool = False,
    disable_identity_safeguard: bool = False,
) -> dict:
    t_gt, p_gt, q_gt = _load_pose_file(file_gt)
    t_est, p_est, q_est = _load_pose_file(file_est)

    len_gt = (
        float(np.sum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))) if p_gt.shape[0] > 1 else 0.0
    )
    len_est = (
        float(np.sum(np.linalg.norm(np.diff(p_est, axis=0), axis=1))) if p_est.shape[0] > 1 else 0.0
    )
    ratio = len_est / (len_gt + 1e-12)

    step1 = _run_eval_time_alignment(
        t_gt=t_gt,
        q_gt=q_gt,
        t_est=t_est,
        q_est=q_est,
        dt_resample=dt_resample,
        offset_min_match_ratio=offset_min_match_ratio,
        max_diff=max_diff,
        disable_time_offset_calibration=disable_time_offset_calibration,
        verbose=verbose,
    )
    input_coverage = _compute_input_coverage_diagnostics(
        t_ref=t_gt,
        pos_ref=p_gt,
        t_est=t_est,
        offset_est_s=float(step1["calculated_offset"]),
    )
    sparse_assoc_failed = False
    sparse_assoc_failure_reason = ""
    try:
        _, p_est_sparse, q_est_sparse, t_gt_sparse, p_gt_sparse, q_gt_sparse, i_est_sparse = _associate_est_gt(
            t_est=t_est,
            p_est=p_est,
            q_est=q_est,
            t_gt=t_gt,
            p_gt=p_gt,
            q_gt=q_gt,
            max_diff=float(max_diff),
            offset=-float(step1["calculated_offset"]),
        )
    except ValueError as exc:
        sparse_assoc_failed = True
        sparse_assoc_failure_reason = str(exc)
        p_est_sparse = np.empty((0, 3), dtype=float)
        q_est_sparse = np.empty((0, 4), dtype=float)
        t_gt_sparse = np.empty((0,), dtype=float)
        p_gt_sparse = np.empty((0, 3), dtype=float)
        q_gt_sparse = np.empty((0, 4), dtype=float)
        i_est_sparse = np.empty((0,), dtype=int)
    strict_decision = _timeline_decision_for_strict_association(
        t_est=t_est,
        t_gt=t_gt,
        t_matched=t_gt_sparse,
        i_est=i_est_sparse,
        sparse_matched=int(t_gt_sparse.size),
        pos_est_matched=p_est_sparse,
    )
    if sparse_assoc_failed and sparse_assoc_failure_reason:
        strict_decision["timeline_policy_reason"] = sparse_assoc_failure_reason
    use_resampled_fallback = bool(t_gt_sparse.size < 3)
    if use_resampled_fallback and not bool(allow_resampled_fallback):
        reason = sparse_assoc_failure_reason or (
            "Strict timestamp association produced fewer than 3 matches."
        )
        raise ValueError(f"EPA resampled fallback disabled; {reason}")
    prefer_sparse_timeline = bool(strict_decision["strict_association_timeline_used"])
    use_dense_timeline = bool(allow_resampled_fallback and not prefer_sparse_timeline)
    if use_dense_timeline:
        solve_eval = _prepare_solve_eval_trajectories(
            t_gt=t_gt,
            pos_gt=p_gt,
            quat_gt=q_gt,
            t_est=t_est,
            pos_est=p_est,
            quat_est=q_est,
            calculated_offset=float(step1["calculated_offset"]),
            downsample_hz=float(downsample_hz),
            quat_interp=str(quat_interp),
        )
        t_gt_m = np.asarray(solve_eval["t_gt"], dtype=float)
        p_gt_m = np.asarray(solve_eval["pos_gt"], dtype=float)
        q_gt_m = np.asarray(solve_eval["quat_gt"], dtype=float)
        p_est_m = np.asarray(solve_eval["pr_sync"], dtype=float)
        q_est_m = np.asarray(solve_eval["qr_sync"], dtype=float)
        pos_gt_solve = np.asarray(solve_eval["pos_gt_solve"], dtype=float)
        quat_gt_solve = np.asarray(solve_eval["quat_gt_solve"], dtype=float)
        p_est_solve = np.asarray(solve_eval["pr_solve"], dtype=float)
        q_est_solve = np.asarray(solve_eval["qr_solve"], dtype=float)
        timeline_policy = (
            "gt_resampled_association_fallback" if use_resampled_fallback else "dense_overlap"
        )
    else:
        t_gt_m = t_gt_sparse
        p_gt_m = p_gt_sparse
        q_gt_m = q_gt_sparse
        p_est_m = p_est_sparse
        q_est_m = q_est_sparse
        pos_gt_solve = p_gt_sparse
        quat_gt_solve = q_gt_sparse
        p_est_solve = p_est_sparse
        q_est_solve = q_est_sparse
        timeline_policy = "sparse_est_association"
    eval_mode = str(eval_align_mode or "none").strip().lower()
    if eval_mode in {"epa_se3_eval", "epa_se3"}:
        eval_mode = "se3"
    elif eval_mode in {"epa_se3r", "rotation_first_se3"}:
        eval_mode = "se3"
    elif eval_mode in {"se3_orginal", "se3-orginal", "se3_original"}:
        eval_mode = "se3-original"
    elif eval_mode == "epa_step3":
        eval_mode = "none"
    if eval_mode in {"posyaw", "epa_posyaw"}:
        global_align_mode = "posyaw"
    elif eval_mode in {"se3", "se3r"}:
        global_align_mode = "se3"
    elif eval_mode == "se3-original":
        global_align_mode = "se3-original"
    else:
        global_align_mode = "se3"

    solved = _solve_extrinsic_and_world_alignment(
        pr_sync=p_est_m,
        qr_sync=q_est_m,
        pos_gt_solve=pos_gt_solve,
        quat_gt_solve=quat_gt_solve,
        pr_solve=p_est_solve,
        qr_solve=q_est_solve,
        global_align_mode=global_align_mode,
        calibration_timestamps_s=t_gt_m,
        calibration_source_timestamps_s=np.asarray(t_est, dtype=float)
        - float(step1["calculated_offset"]),
        disable_extrinsic_calibration=disable_extrinsic_calibration,
        compare_identity_candidate=not bool(disable_identity_safeguard),
    )

    gt_t = np.asarray(t_gt_m, dtype=float)
    gt_pos = np.asarray(p_gt_m, dtype=float)
    gt_quat = np.asarray(q_gt_m, dtype=float)
    est_pos = np.asarray(solved["pr_final"], dtype=float)
    est_quat = np.asarray(solved["q_step3"], dtype=float)
    eval_alignment: dict[str, object] = {
        "align_mode": (
            "posyaw_step3"
            if global_align_mode == "posyaw"
            else (
                "se3_step3"
                if global_align_mode == "se3"
                else ("se3_original_step3" if global_align_mode == "se3-original" else "none")
            )
        ),
        "step3_global_align_mode": global_align_mode,
        "step3_alignment_mode": str(solved.get("step3_choice", {}).get("step3_alignment_mode", "")),
        "step3_orientation_mode": str(solved.get("step3_choice", {}).get("orientation_mode", "")),
        "timeline_policy": timeline_policy,
        "dense_timeline_used": bool(use_dense_timeline),
        "auto_sparse_low_rate_estimate_used": bool(prefer_sparse_timeline),
        "sparse_association_failed": bool(sparse_assoc_failed),
        "sparse_association_failure_reason": sparse_assoc_failure_reason,
        "resampled_fallback_used": bool(use_resampled_fallback),
        "time_offset_calibration_disabled": bool(disable_time_offset_calibration),
        "extrinsic_calibration_disabled": bool(disable_extrinsic_calibration),
        **{
            key: value for key, value in solved.get("step3_choice", {}).items()
            if key.startswith("extrinsic_") and key != "extrinsic_calibration_disabled"
        },
        **strict_decision,
    }
    if eval_mode not in {
        "",
        "none",
        "se3",
        "se3r",
        "se3-original",
        "posyaw",
        "epa_posyaw",
    }:
        est_pos, est_quat, eval_alignment = align_for_eval_with_info(
            pos_ref=gt_pos,
            quat_ref=gt_quat,
            pos_est=est_pos,
            quat_est=est_quat,
            mode=eval_mode,
            n_to_align=-1,
        )

    ape3 = compute_ape(
        pos_ref=gt_pos,
        quat_ref=gt_quat,
        pos_est=est_pos,
        quat_est=est_quat,
        include_raw=True,
    )

    p_gt_xy, q_gt_xy = project_to_plane(gt_pos, gt_quat, "xy")
    p_est_xy, q_est_xy = project_to_plane(est_pos, est_quat, "xy")
    ape2 = compute_ape(
        pos_ref=p_gt_xy,
        quat_ref=q_gt_xy,
        pos_est=p_est_xy,
        quat_est=q_est_xy,
        include_raw=True,
    )

    return {
        "matched": int(gt_t.size),
        "length_ratio": float(ratio),
        "length_gt": float(len_gt),
        "length_est": float(len_est),
        "input_coverage": input_coverage,
        "calculated_offset": float(step1["calculated_offset"]),
        "time_metrics": dict(step1["time_metrics"]),
        "gt_t": gt_t,
        "gt_pos": gt_pos,
        "gt_quat": gt_quat,
        "est_pos": est_pos,
        "est_quat": est_quat,
        "eval_source": (
            "epa_posyaw"
            if eval_mode in {"posyaw", "epa_posyaw"}
            else (
                "epa_se3"
                if eval_mode in {"se3", "se3r"}
                else (
                    "epa_se3_original"
                    if eval_mode == "se3-original"
                    else ("epa_step3" if eval_mode in {"", "none"} else "epa_eval_align")
                )
            )
        ),
        "eval_alignment": eval_alignment,
        "ate3_ori": dict(ape3["rotation_angle_deg"]),
        "ate3_pos": dict(ape3["translation_part"]),
        "ate2_ori": dict(ape2["rotation_angle_deg"]),
        "ate2_pos": dict(ape2["translation_part"]),
    }


def _evaluate_pair(
    file_gt: Path,
    file_est: Path,
    align_mode: str,
    max_diff: float,
    *,
    epa_dt_resample: float = _DEFAULT_EPA_DT_RESAMPLE,
    epa_offset_min_match_ratio: float = _DEFAULT_EPA_OFFSET_MIN_MATCH_RATIO,
    epa_downsample_hz: float = _DEFAULT_EPA_DOWNSAMPLE_HZ,
    epa_quat_interp: str = _DEFAULT_EPA_QUAT_INTERP,
    epa_no_fallback: bool = False,
    epa_verbose_fallback: bool = False,
    epa_disable_time_offset_calibration: bool = False,
    epa_disable_extrinsic_calibration: bool = False,
    epa_disable_calibration: bool = False,
    epa_disable_identity_safeguard: bool = False,
) -> dict:
    requested_align_alias = str(align_mode).lower()
    requested_align_mode = _public_align_mode(requested_align_alias)
    if requested_align_mode in {"se3", "se3-original", "posyaw"}:
        try:
            result = _evaluate_pair_epa_step3(
                file_gt=file_gt,
                file_est=file_est,
                max_diff=float(max_diff),
                eval_align_mode=(
                    "posyaw" if requested_align_mode == "posyaw" else requested_align_mode
                ),
                dt_resample=float(epa_dt_resample),
                offset_min_match_ratio=float(epa_offset_min_match_ratio),
                downsample_hz=float(epa_downsample_hz),
                quat_interp=str(epa_quat_interp),
                verbose=bool(epa_verbose_fallback),
                allow_resampled_fallback=not bool(epa_no_fallback),
                disable_identity_safeguard=bool(epa_disable_identity_safeguard),
                disable_time_offset_calibration=(
                    bool(epa_disable_calibration)
                    or bool(epa_disable_time_offset_calibration)
                ),
                disable_extrinsic_calibration=(
                    bool(epa_disable_calibration)
                    or bool(epa_disable_extrinsic_calibration)
                ),
            )
            result["requested_align_mode"] = requested_align_mode
            if requested_align_alias != requested_align_mode:
                result["requested_align_alias"] = requested_align_alias
            return result
        except Exception as exc:
            param_msg = (
                f"epa_dt_resample={float(epa_dt_resample):.6g}, "
                f"epa_offset_min_match_ratio={float(epa_offset_min_match_ratio):.6g}"
            )
            raise RuntimeError(
                f"EPA Step3 evaluation failed for {file_est.name}; {param_msg}: {exc}"
            ) from exc

    result = _evaluate_pair_ov_style(
        file_gt=file_gt,
        file_est=file_est,
        align_mode=requested_align_mode,
        max_diff=float(max_diff),
        downsample_hz=float(epa_downsample_hz),
        quat_interp=str(epa_quat_interp),
        allow_resampled_fallback=not bool(epa_no_fallback),
        disable_extrinsic_calibration=(
            bool(epa_disable_calibration)
            or bool(epa_disable_extrinsic_calibration)
        ),
        calibrate_time=requested_align_mode == "sim3",
        dt_resample=float(epa_dt_resample),
        offset_min_match_ratio=float(epa_offset_min_match_ratio),
        disable_time_offset_calibration=(
            bool(epa_disable_calibration)
            or bool(epa_disable_time_offset_calibration)
        ),
        verbose=bool(epa_verbose_fallback),
    )
    result["requested_align_mode"] = requested_align_mode
    if requested_align_alias != requested_align_mode:
        result["requested_align_alias"] = requested_align_alias
    if result.get("eval_source") != "epa_eval_align":
        result["eval_source"] = "epa_eval_align"
    return result


def _epa_eval_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "epa_disable_identity_safeguard": bool(
            getattr(args, "epa_disable_identity_safeguard", False)
        ),
        "epa_disable_time_offset_calibration": bool(
            getattr(args, "epa_disable_time_offset_calibration", False)
        ),
        "epa_disable_extrinsic_calibration": bool(
            getattr(args, "epa_disable_extrinsic_calibration", False)
        ),
        "epa_disable_calibration": bool(
            getattr(args, "epa_disable_calibration", False)
        ),
        "epa_dt_resample": float(getattr(args, "epa_dt_resample", _DEFAULT_EPA_DT_RESAMPLE)),
        "epa_offset_min_match_ratio": float(
            getattr(args, "epa_offset_min_match_ratio", _DEFAULT_EPA_OFFSET_MIN_MATCH_RATIO)
        ),
        "epa_downsample_hz": float(getattr(args, "epa_downsample_hz", _DEFAULT_EPA_DOWNSAMPLE_HZ)),
        "epa_quat_interp": str(getattr(args, "epa_quat_interp", _DEFAULT_EPA_QUAT_INTERP)),
        "epa_no_fallback": bool(getattr(args, "epa_no_fallback", False)),
        "epa_verbose_fallback": bool(getattr(args, "epa_verbose_fallback", False)),
    }


def _orientation_quality_warning(
    eval_res: dict, success: dict, time_rpe: dict
) -> dict[str, object]:
    sr_distance = float(success.get("success_rate_distance", 0.0))
    ape_rmse = float(eval_res.get("ate3_ori", {}).get("rmse", np.nan))
    rpe_rmse = float(time_rpe.get("ori_stats", {}).get("rmse", np.nan))
    high_translation_sr = bool(
        np.isfinite(sr_distance) and sr_distance >= _ORIENTATION_WARNING_MIN_SR
    )
    ape_bad = bool(np.isfinite(ape_rmse) and ape_rmse >= _ORIENTATION_WARNING_APE_RMSE_DEG)
    rpe_bad = bool(np.isfinite(rpe_rmse) and rpe_rmse >= _ORIENTATION_WARNING_RPE_RMSE_DEG)
    unstable = bool(high_translation_sr and (ape_bad or rpe_bad))
    warning = ""
    if unstable:
        warning = (
            "Translation SR is high but rotation error is unstable; "
            "treat this run as evaluation-suspect until pose convention/data mapping is checked."
        )
    return {
        "orientation_unstable": unstable,
        "orientation_warning": warning,
        "orientation_translation_sr_distance": sr_distance,
        "orientation_ape_rmse_deg": ape_rmse,
        "orientation_rpe_time_1s_rmse_deg": rpe_rmse,
        "orientation_min_sr_for_warning": _ORIENTATION_WARNING_MIN_SR,
        "orientation_ape_rmse_threshold_deg": _ORIENTATION_WARNING_APE_RMSE_DEG,
        "orientation_rpe_time_1s_rmse_threshold_deg": _ORIENTATION_WARNING_RPE_RMSE_DEG,
    }


def _eval_quality_flags(eval_res: dict, success: dict, time_rpe: dict) -> dict[str, object]:
    eval_alignment = eval_res.get("eval_alignment", {})
    sim3_reliable = True
    sim3_warning = ""
    if (
        isinstance(eval_alignment, dict)
        and str(eval_alignment.get("align_mode", "")).lower() == "sim3"
    ):
        sim3_reliable = bool(eval_alignment.get("sim3_reliable", True))
        sim3_warning = str(eval_alignment.get("sim3_warning", "") or "")

    orientation = _orientation_quality_warning(eval_res, success, time_rpe)
    orientation_unstable = bool(orientation["orientation_unstable"])
    warnings: list[str] = []
    if not sim3_reliable:
        fallback = str(
            eval_alignment.get("sim3_failure_reason", "")
            or eval_alignment.get("sim3_confidence", "")
            or eval_alignment.get("sim3_consensus_status", "")
            or eval_alignment.get("sim3_anchor_status", "")
            or "Sim3 transform is unreliable; interpret SR together with the reliability status."
        )
        warnings.append(sim3_warning or fallback)
    if orientation_unstable and orientation["orientation_warning"]:
        warnings.append(str(orientation["orientation_warning"]))
    coverage_status = str(success.get("input_coverage_status", "ok"))
    if coverage_status in {"warning", "failed"}:
        warnings.append(
            "Reference support is incomplete; complete-reference SR counts "
            "the unsupported part as unsuccessful."
        )

    return {
        **orientation,
        "sim3_reliable": sim3_reliable,
        "input_coverage_status": coverage_status,
        "eval_reliable": bool(
            sim3_reliable and not orientation_unstable and coverage_status != "failed"
        ),
        "eval_warning": " ".join(warnings),
    }


def _drift_rate_percent(values, segment_m: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or float(segment_m) <= 0.0:
        return float("nan")
    return float(np.mean(vals / float(segment_m)) * 100.0)


def _compute_ov_eval_comparison_indices(
    accum_distances: np.ndarray, distance: float, max_dist_diff: float = 0.5
) -> np.ndarray:
    distances = np.asarray(accum_distances, dtype=float).reshape(-1)
    comparisons = np.full(distances.size, -1, dtype=int)
    target_delta = float(distance)
    max_diff = float(max_dist_diff)
    if distances.size == 0:
        return comparisons
    targets = distances + target_delta
    insert_ids = np.searchsorted(distances, targets, side="left")
    candidates = insert_ids[:, None] + np.array([-1, 0, 1], dtype=int)
    starts = np.arange(distances.size, dtype=int)[:, None]
    valid = (candidates >= starts) & (candidates >= 0) & (candidates < distances.size)
    safe_candidates = np.clip(candidates, 0, distances.size - 1)
    errors = np.abs(distances[safe_candidates] - targets[:, None])
    errors[~valid] = np.inf
    best_columns = np.argmin(errors, axis=1)
    rows = np.arange(distances.size, dtype=int)
    best_errors = errors[rows, best_columns]
    accepted = best_errors < max_diff
    comparisons[accepted] = candidates[rows[accepted], best_columns[accepted]]
    return comparisons


def _ov_eval_error_statistics(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {
            "rmse": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sse": 0.0,
        }
    return compute_error_statistics(arr)


def _pose_matrix_ov_eval(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return T


def _compute_rpe_segments_ov_eval_style(
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    est_pos: np.ndarray,
    est_quat: np.ndarray,
    segments_m: list[float],
    *,
    valid_segment_mask: np.ndarray | None = None,
) -> dict[float, dict[str, np.ndarray | dict[str, float] | int]]:
    gt_pos = np.asarray(gt_pos, dtype=float)
    est_pos = np.asarray(est_pos, dtype=float)
    gt_quat = np.asarray(gt_quat, dtype=float)
    est_quat = np.asarray(est_quat, dtype=float)
    accum_distances = np.zeros(gt_pos.shape[0], dtype=float)
    if gt_pos.shape[0] > 1:
        accum_distances[1:] = np.cumsum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))

    valid_sample_mask = None
    valid_edge_mask = None
    if valid_segment_mask is not None:
        valid_mask = np.asarray(valid_segment_mask, dtype=bool).reshape(-1)
        if valid_mask.size == gt_pos.shape[0]:
            valid_sample_mask = valid_mask
        elif valid_mask.size == max(0, gt_pos.shape[0] - 1):
            valid_edge_mask = valid_mask

    if gt_pos.shape[0] == 0:
        gt_rotations = np.empty((0, 3, 3), dtype=float)
        est_rotations = np.empty((0, 3, 3), dtype=float)
    else:
        gt_rotations = R.from_quat(gt_quat).as_matrix()
        est_rotations = R.from_quat(est_quat).as_matrix()

    invalid_sample_prefix = None
    invalid_edge_prefix = None
    if valid_sample_mask is not None:
        invalid_sample_prefix = np.concatenate(
            ([0], np.cumsum(~valid_sample_mask, dtype=int))
        )
    if valid_edge_mask is not None:
        invalid_edge_prefix = np.concatenate(([0], np.cumsum(~valid_edge_mask, dtype=int)))

    out: dict[float, dict[str, np.ndarray | dict[str, float] | int]] = {}
    for seg in segments_m:
        comparisons = _compute_ov_eval_comparison_indices(
            accum_distances, float(seg), max_dist_diff=0.5
        )
        starts = np.flatnonzero(comparisons >= 0)
        ends = comparisons[starts]
        keep = np.ones(starts.size, dtype=bool)
        if invalid_sample_prefix is not None:
            keep &= (
                invalid_sample_prefix[ends + 1] - invalid_sample_prefix[starts]
            ) == 0
        if invalid_edge_prefix is not None:
            keep &= (invalid_edge_prefix[ends] - invalid_edge_prefix[starts]) == 0
        starts = starts[keep]
        ends = ends[keep]
        pair_ids = np.column_stack((starts, ends)).astype(int, copy=False)

        if pair_ids.shape[0] == 0:
            ori_arr = np.empty(0, dtype=float)
            pos_arr = np.empty(0, dtype=float)
        else:
            est_rot_start = est_rotations[starts]
            est_rot_end = est_rotations[ends]
            gt_rot_start = gt_rotations[starts]
            gt_rot_end = gt_rotations[ends]

            est_relative_rot = np.einsum(
                "nij,njk->nik", np.swapaxes(est_rot_start, 1, 2), est_rot_end
            )
            gt_relative_rot = np.einsum(
                "nij,njk->nik", np.swapaxes(gt_rot_start, 1, 2), gt_rot_end
            )
            est_relative_pos = np.einsum(
                "nij,nj->ni",
                np.swapaxes(est_rot_start, 1, 2),
                est_pos[ends] - est_pos[starts],
            )
            gt_relative_pos = np.einsum(
                "nij,nj->ni",
                np.swapaxes(gt_rot_start, 1, 2),
                gt_pos[ends] - gt_pos[starts],
            )

            error_rot_c2 = np.einsum(
                "nij,njk->nik", np.swapaxes(gt_relative_rot, 1, 2), est_relative_rot
            )
            error_pos_c2 = np.einsum(
                "nij,nj->ni",
                np.swapaxes(gt_relative_rot, 1, 2),
                est_relative_pos - gt_relative_pos,
            )
            error_rot_world = np.einsum(
                "nij,njk,nlk->nil", est_rot_end, error_rot_c2, est_rot_end
            )
            error_pos_world = np.einsum("nij,nj->ni", est_rot_end, error_pos_c2)

            pos_arr = np.linalg.norm(error_pos_world, axis=1)
            ori_arr = np.degrees(R.from_matrix(error_rot_world).magnitude())
        out[float(seg)] = {
            "ori_values": ori_arr,
            "pos_values": pos_arr,
            "ori_stats": _ov_eval_error_statistics(ori_arr),
            "pos_stats": _ov_eval_error_statistics(pos_arr),
            "pair_count": int(pos_arr.size),
            "pair_ids": pair_ids,
        }
    return out


def _compute_rpe_segments(
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    est_pos: np.ndarray,
    est_quat: np.ndarray,
    segments_m: list[float],
) -> dict[float, dict[str, np.ndarray | dict[str, float] | int]]:
    return _compute_rpe_segments_ov_eval_style(
        gt_pos=gt_pos,
        gt_quat=gt_quat,
        est_pos=est_pos,
        est_quat=est_quat,
        segments_m=segments_m,
    )


def _compute_time_rpe_1s(
    gt_t: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    est_pos: np.ndarray,
    est_quat: np.ndarray,
) -> dict[str, np.ndarray | dict[str, float] | int]:
    blk = compute_rpe(
        pos_ref=gt_pos,
        quat_ref=gt_quat,
        pos_est=est_pos,
        quat_est=est_quat,
        delta=1.0,
        delta_unit="s",
        rel_delta_tol=0.1,
        all_pairs=True,
        pairs_from_reference=True,
        timestamps=np.asarray(gt_t, dtype=float),
        include_raw=True,
    )
    ori_vals = np.asarray(blk["_error_arrays"]["rotation_angle_deg"], dtype=float)
    pos_vals = np.asarray(blk["_error_arrays"]["translation_part"], dtype=float)
    return {
        "ori_values": ori_vals,
        "pos_values": pos_vals,
        "ori_stats": compute_error_statistics(ori_vals),
        "pos_stats": compute_error_statistics(pos_vals),
        "pair_count": int(blk.get("pair_count", int(pos_vals.size))),
    }


def _compute_valid_segment_summary(
    gt_t: np.ndarray,
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    est_pos: np.ndarray,
    est_quat: np.ndarray,
    threshold_m: float = 10.0,
    threshold_mode: str = "adaptive_knee",
    threshold_min_m: float = 5.0,
    threshold_max_m: float = 30.0,
    threshold_trim_percentile: float = 95.0,
    global_gate_mode: str = "fixed",
    global_gate_m: float = 30.0,
    global_gate_path_ratio: float = 0.05,
    global_gate_min_m: float = 2.0,
    global_gate_max_m: float = 100.0,
    global_gate_percentile: float = 5.0,
    drift_threshold_mode: str = "adaptive",
    drift_rpe_1s_m: float = 2.0,
    drift_ape_slope_mps: float = 1.0,
    drift_ape_jump_m: float = 5.0,
) -> dict:
    ape = compute_ape(
        pos_ref=gt_pos,
        quat_ref=gt_quat,
        pos_est=est_pos,
        quat_est=est_quat,
        include_raw=True,
    )
    rpe = compute_rpe(
        pos_ref=gt_pos,
        quat_ref=gt_quat,
        pos_est=est_pos,
        quat_est=est_quat,
        delta=1.0,
        delta_unit="f",
        include_raw=True,
    )
    rpe_time = compute_rpe(
        pos_ref=gt_pos,
        quat_ref=gt_quat,
        pos_est=est_pos,
        quat_est=est_quat,
        delta=1.0,
        delta_unit="s",
        rel_delta_tol=0.1,
        all_pairs=True,
        pairs_from_reference=True,
        timestamps=np.asarray(gt_t, dtype=float),
        include_raw=True,
    )
    resolved_threshold_m, threshold_info = resolve_success_threshold(
        ape["_error_arrays"]["translation_part"],
        mode=str(threshold_mode),
        fixed_threshold_m=float(threshold_m),
        min_threshold_m=float(threshold_min_m),
        max_threshold_m=float(threshold_max_m),
        trim_percentile=float(threshold_trim_percentile),
    )
    valid = compute_valid_segment_metrics(
        timestamps=gt_t,
        pos_ref=gt_pos,
        ape_block=ape,
        rpe_block=rpe,
        rpe_time_1s_block=rpe_time,
        threshold_m=float(resolved_threshold_m),
        threshold_info=threshold_info,
        global_gate_mode=str(global_gate_mode),
        global_gate_m=float(global_gate_m),
        global_gate_path_ratio=float(global_gate_path_ratio),
        global_gate_min_m=float(global_gate_min_m),
        global_gate_max_m=float(global_gate_max_m),
        global_gate_percentile=float(global_gate_percentile),
        drift_threshold_mode=str(drift_threshold_mode),
        drift_rpe_1s_m=float(drift_rpe_1s_m),
        drift_ape_slope_mps=float(drift_ape_slope_mps),
        drift_ape_jump_m=float(drift_ape_jump_m),
        include_raw=True,
        include_masks=True,
    )
    return valid


def _compute_valid_rpe_segments(
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    est_pos: np.ndarray,
    est_quat: np.ndarray,
    valid_segment_mask: np.ndarray,
    segments_m: list[float],
) -> dict[float, dict[str, np.ndarray | dict[str, float] | int]]:
    return _compute_rpe_segments_ov_eval_style(
        gt_pos=gt_pos,
        gt_quat=gt_quat,
        est_pos=est_pos,
        est_quat=est_quat,
        segments_m=segments_m,
        valid_segment_mask=np.asarray(valid_segment_mask, dtype=bool),
    )


def _fmt_sr_config(valid: dict, gt_t: np.ndarray, gt_pos: np.ndarray) -> str:
    success = valid["success"]
    threshold = success.get("threshold", {})
    threshold_m = float(threshold.get("threshold_m", success.get("threshold_m", np.nan)))
    threshold_mode = str(threshold.get("mode", "unknown"))
    gate_m = float(success.get("global_gate_m", np.nan))
    gate_mode = str(success.get("global_gate_mode", "unknown"))
    gt_t = np.asarray(gt_t, dtype=float).reshape(-1)
    duration_s = float(
        success.get(
            "complete_total_time_s",
            gt_t[-1] - gt_t[0] if gt_t.size > 1 else 0.0,
        )
    )
    path_m = float(
        success.get(
            "complete_total_distance_m",
            success.get("global_gate_path_length_m", compute_path_length(gt_pos)),
        )
    )
    return (
        f"SR config: GT path={_fmt(path_m, 2)}m | time={_fmt(duration_s, 2)}s | "
        f"threshold={_fmt(threshold_m, 2)}m({threshold_mode}) | "
        f"gate={_fmt(gate_m, 2)}m({gate_mode})"
    )
