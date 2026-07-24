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
    _LOW_RATE_ESTIMATE_STRICT_MIN_DT_S,
    _LOW_RATE_ESTIMATE_STRICT_MIN_EST_MATCH_RATIO,
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


def _prefer_sparse_timeline_for_low_rate_estimate(
    *,
    t_est: np.ndarray,
    sparse_matched: int,
) -> bool:
    if int(sparse_matched) < 3:
        return False
    est_count = int(np.asarray(t_est).reshape(-1).size)
    if est_count <= 0:
        return False
    est_match_ratio = float(sparse_matched) / float(est_count)
    if est_match_ratio < float(_LOW_RATE_ESTIMATE_STRICT_MIN_EST_MATCH_RATIO):
        return False
    est_dt_s = _median_positive_dt_s(np.asarray(t_est, dtype=float))
    return bool(np.isfinite(est_dt_s) and est_dt_s >= float(_LOW_RATE_ESTIMATE_STRICT_MIN_DT_S))


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
    try:
        _, p_est_m, q_est_m, t_gt_m, p_gt_m, q_gt_m, _ = _associate_est_gt(
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
        auto_sparse = _prefer_sparse_timeline_for_low_rate_estimate(
            t_est=t_est,
            sparse_matched=sparse_matched,
        )
        if (not bool(allow_resampled_fallback)) or auto_sparse:
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
                    "sparse_matched": sparse_matched,
                    "sparse_est_match_ratio": float(sparse_matched)
                    / max(float(np.asarray(t_est).reshape(-1).size), 1.0),
                    "estimate_median_dt_s": _median_positive_dt_s(t_est),
                    "auto_sparse_low_rate_estimate_used": bool(
                        bool(allow_resampled_fallback) and auto_sparse
                    ),
                    "dense_timeline_used": False,
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
            "sparse_matched": sparse_matched,
            "sparse_est_match_ratio": float(sparse_matched)
            / max(float(np.asarray(t_est).reshape(-1).size), 1.0),
            "estimate_median_dt_s": _median_positive_dt_s(t_est),
            "auto_sparse_low_rate_estimate_used": False,
            "dense_timeline_used": True,
            "overlap_info": overlap_info,
            "downsample_info": solve_eval.get("downsample_info", {}),
        },
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

    t_gt_m, p_gt_m, q_gt_m, p_est_m, q_est_m, timeline_info = _associate_or_resample_ov_style(
        t_est=t_est,
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
    )
    align_info.update(timeline_info)
    align_info["requested_align_mode"] = requested_align_mode
    align_info["eval_source"] = "epa_eval_align"

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

    if bool(verbose):
        step1 = _run_time_alignment(
            t_gt=t_gt,
            quat_gt=q_gt,
            t_est=t_est,
            quat_est=q_est,
            dt_resample=float(dt_resample),
            offset_search_window_s=0.0,
            offset_min_match_ratio=float(offset_min_match_ratio),
            evo_match_max_diff_s=float(max_diff),
            artificial_offset_s=None,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            step1 = _run_time_alignment(
                t_gt=t_gt,
                quat_gt=q_gt,
                t_est=t_est,
                quat_est=q_est,
                dt_resample=float(dt_resample),
                offset_search_window_s=0.0,
                offset_min_match_ratio=float(offset_min_match_ratio),
                evo_match_max_diff_s=float(max_diff),
                artificial_offset_s=None,
            )
    sparse_assoc_failed = False
    sparse_assoc_failure_reason = ""
    try:
        _, p_est_sparse, q_est_sparse, t_gt_sparse, p_gt_sparse, q_gt_sparse, _ = _associate_est_gt(
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
    sparse_gt_match_ratio = float(t_gt_sparse.size) / max(float(t_gt.size), 1.0)
    sparse_est_match_ratio = float(t_gt_sparse.size) / max(float(t_est.size), 1.0)
    use_resampled_fallback = bool(t_gt_sparse.size < 3)
    if use_resampled_fallback and not bool(allow_resampled_fallback):
        reason = sparse_assoc_failure_reason or (
            "Strict timestamp association produced fewer than 3 matches."
        )
        raise ValueError(f"EPA resampled fallback disabled; {reason}")
    prefer_sparse_timeline = _prefer_sparse_timeline_for_low_rate_estimate(
        t_est=t_est,
        sparse_matched=int(t_gt_sparse.size),
    )
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
    elif eval_mode == "epa_step3":
        eval_mode = "none"
    global_align_mode = "posyaw" if eval_mode in {"posyaw", "epa_posyaw"} else "se3"

    solved = _solve_extrinsic_and_world_alignment(
        pr_sync=p_est_m,
        qr_sync=q_est_m,
        pos_gt_solve=pos_gt_solve,
        quat_gt_solve=quat_gt_solve,
        pr_solve=p_est_solve,
        qr_solve=q_est_solve,
        global_align_mode=global_align_mode,
    )

    gt_t = np.asarray(t_gt_m, dtype=float)
    gt_pos = np.asarray(p_gt_m, dtype=float)
    gt_quat = np.asarray(q_gt_m, dtype=float)
    est_pos = np.asarray(solved["pr_final"], dtype=float)
    est_quat = np.asarray(solved["q_step3"], dtype=float)
    eval_alignment: dict[str, object] = {
        "align_mode": "posyaw_step3" if global_align_mode == "posyaw" else "none",
        "step3_global_align_mode": global_align_mode,
        "step3_alignment_mode": str(solved.get("step3_choice", {}).get("step3_alignment_mode", "")),
        "timeline_policy": timeline_policy,
        "sparse_gt_match_ratio": sparse_gt_match_ratio,
        "sparse_est_match_ratio": sparse_est_match_ratio,
        "sparse_matched": int(t_gt_sparse.size),
        "estimate_median_dt_s": _median_positive_dt_s(t_est),
        "dense_timeline_used": bool(use_dense_timeline),
        "auto_sparse_low_rate_estimate_used": bool(prefer_sparse_timeline),
        "sparse_association_failed": bool(sparse_assoc_failed),
        "sparse_association_failure_reason": sparse_assoc_failure_reason,
        "resampled_fallback_used": bool(use_resampled_fallback),
    }
    if eval_mode not in {"", "none", "posyaw", "epa_posyaw"}:
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
        "gt_t": gt_t,
        "gt_pos": gt_pos,
        "gt_quat": gt_quat,
        "est_pos": est_pos,
        "est_quat": est_quat,
        "eval_source": (
            "epa_posyaw"
            if eval_mode in {"posyaw", "epa_posyaw"}
            else ("epa_step3" if eval_mode in {"", "none"} else "epa_eval_align")
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
) -> dict:
    requested_align_alias = str(align_mode).lower()
    requested_align_mode = _public_align_mode(requested_align_alias)
    if requested_align_mode in {"se3", "posyaw"}:
        try:
            result = _evaluate_pair_epa_step3(
                file_gt=file_gt,
                file_est=file_est,
                max_diff=float(max_diff),
                eval_align_mode="posyaw" if requested_align_mode == "posyaw" else "none",
                dt_resample=float(epa_dt_resample),
                offset_min_match_ratio=float(epa_offset_min_match_ratio),
                downsample_hz=float(epa_downsample_hz),
                quat_interp=str(epa_quat_interp),
                verbose=bool(epa_verbose_fallback),
                allow_resampled_fallback=not bool(epa_no_fallback),
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
    )
    result["requested_align_mode"] = requested_align_mode
    if requested_align_alias != requested_align_mode:
        result["requested_align_alias"] = requested_align_alias
    if result.get("eval_source") != "epa_eval_align":
        result["eval_source"] = "epa_eval_align"
    return result


def _epa_eval_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
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

    return {
        **orientation,
        "sim3_reliable": sim3_reliable,
        "eval_reliable": bool(sim3_reliable and not orientation_unstable),
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
    for idx, insert_idx in enumerate(insert_ids):
        best_error = max_diff
        best_idx = -1
        for end_idx in (int(insert_idx) - 1, int(insert_idx), int(insert_idx) + 1):
            if end_idx < idx or end_idx < 0 or end_idx >= distances.size:
                continue
            err = abs(float(distances[end_idx]) - float(targets[idx]))
            if err < best_error:
                best_idx = int(end_idx)
                best_error = err
        comparisons[idx] = best_idx
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

    out: dict[float, dict[str, np.ndarray | dict[str, float] | int]] = {}
    for seg in segments_m:
        comparisons = _compute_ov_eval_comparison_indices(
            accum_distances, float(seg), max_dist_diff=0.5
        )
        ori_vals: list[float] = []
        pos_vals: list[float] = []
        pair_ids: list[tuple[int, int]] = []
        for id_start, id_end_raw in enumerate(comparisons):
            id_end = int(id_end_raw)
            if id_end == -1:
                continue
            if valid_sample_mask is not None and not bool(
                np.all(valid_sample_mask[id_start : id_end + 1])
            ):
                continue
            if valid_edge_mask is not None and not bool(np.all(valid_edge_mask[id_start:id_end])):
                continue

            T_c1 = _pose_matrix_ov_eval(est_pos[id_start], est_quat[id_start])
            T_c2 = _pose_matrix_ov_eval(est_pos[id_end], est_quat[id_end])
            T_m1 = _pose_matrix_ov_eval(gt_pos[id_start], gt_quat[id_start])
            T_m2 = _pose_matrix_ov_eval(gt_pos[id_end], gt_quat[id_end])

            T_c1_c2 = np.linalg.inv(T_c1) @ T_c2
            T_m1_m2 = np.linalg.inv(T_m1) @ T_m2
            T_error_in_c2 = np.linalg.inv(T_m1_m2) @ T_c1_c2
            T_c2_rot = np.eye(4, dtype=float)
            T_c2_rot[:3, :3] = T_c2[:3, :3]
            T_c2_rot_inv = np.eye(4, dtype=float)
            T_c2_rot_inv[:3, :3] = T_c2[:3, :3].T
            T_error_in_w = T_c2_rot @ T_error_in_c2 @ T_c2_rot_inv

            pos_vals.append(float(np.linalg.norm(T_error_in_w[:3, 3])))
            ori_vals.append(float(np.degrees(R.from_matrix(T_error_in_w[:3, :3]).magnitude())))
            pair_ids.append((int(id_start), int(id_end)))

        ori_arr = np.asarray(ori_vals, dtype=float)
        pos_arr = np.asarray(pos_vals, dtype=float)
        out[float(seg)] = {
            "ori_values": ori_arr,
            "pos_values": pos_arr,
            "ori_stats": _ov_eval_error_statistics(ori_arr),
            "pos_stats": _ov_eval_error_statistics(pos_arr),
            "pair_count": int(pos_arr.size),
            "pair_ids": np.asarray(pair_ids, dtype=int).reshape(-1, 2)
            if pair_ids
            else np.empty((0, 2), dtype=int),
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
    duration_s = float(gt_t[-1] - gt_t[0]) if gt_t.size > 1 else 0.0
    path_m = float(success.get("global_gate_path_length_m", compute_path_length(gt_pos)))
    return (
        f"SR config: GT path={_fmt(path_m, 2)}m | time={_fmt(duration_s, 2)}s | "
        f"threshold={_fmt(threshold_m, 2)}m({threshold_mode}) | "
        f"gate={_fmt(gate_m, 2)}m({gate_mode})"
    )
