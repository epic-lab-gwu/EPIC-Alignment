import numpy as np
from scipy.spatial.transform import Rotation as R

from .calibration import solve_world_alignment
from .math_utils import normalize_quat_array
from ..metric_cli_common import cum_distance


def _umeyama_transform(src_xyz, dst_xyz, with_scale=False):
    src = np.asarray(src_xyz, dtype=float)
    dst = np.asarray(dst_xyz, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Umeyama alignment requires at least 3 paired 3D points.")

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R_align = U @ S @ Vt

    scale = 1.0
    if with_scale:
        var_src = np.mean(np.sum(src_centered**2, axis=1))
        if var_src < 1e-12:
            raise ValueError("Degenerate source trajectory for scale alignment.")
        scale = float(np.trace(np.diag(D) @ S) / var_src)
    t_align = mu_dst - scale * (R_align @ mu_src)
    return scale, R_align, t_align


def _segment_ranges_by_time(
    tvals,
    *,
    segment_duration_s: float,
    overlap_ratio: float = 0.5,
    min_samples: int = 40,
):
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    n = int(tvals.size)
    if n <= 0:
        return []
    if n < int(min_samples):
        return [(0, n)]

    if segment_duration_s <= 0.0:
        return [(0, n)]
    overlap_ratio = float(np.clip(overlap_ratio, 0.0, 0.95))

    dts = np.diff(tvals)
    finite_dts = dts[np.isfinite(dts) & (dts > 1e-9)]
    if finite_dts.size == 0:
        return [(0, n)]
    dt_med = float(np.median(finite_dts))
    seg_len = max(int(min_samples), int(round(float(segment_duration_s) / dt_med)))
    seg_len = min(seg_len, n)
    step = max(1, int(round(seg_len * (1.0 - overlap_ratio))))

    ranges = []
    start = 0
    while start < n:
        end = min(n, start + seg_len)
        if end - start >= int(min_samples):
            ranges.append((start, end))
        if end >= n:
            break
        start += step

    if not ranges:
        ranges = [(0, n)]
    elif ranges[-1][1] < n:
        tail_start = max(0, n - seg_len)
        if n - tail_start >= int(min_samples):
            ranges.append((tail_start, n))
    return ranges


def _segment_heading_angles_deg(pos_ref, pos_est):
    pref = np.asarray(pos_ref, dtype=float)
    pest = np.asarray(pos_est, dtype=float)
    if pref.shape[0] < 3 or pest.shape[0] < 3:
        return np.array([], dtype=float)
    v_ref = np.diff(pref, axis=0)
    v_est = np.diff(pest, axis=0)
    n_ref = np.linalg.norm(v_ref, axis=1)
    n_est = np.linalg.norm(v_est, axis=1)
    valid = (n_ref > 1e-9) & (n_est > 1e-9)
    if not np.any(valid):
        return np.array([], dtype=float)
    cosang = np.sum(v_ref[valid] * v_est[valid], axis=1) / (n_ref[valid] * n_est[valid] + 1e-12)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _compute_alignment_quality(
    *,
    t_ref,
    pos_ref,
    pos_step3,
    raw_rmse_m: float,
    step3_rmse_m: float,
    segment_duration_s: float,
    overlap_ratio: float,
    min_samples: int,
    good_rmse_m: float,
    partial_rmse_m: float,
    good_seg_cv: float,
    good_heading_p90_deg: float,
    partial_min_improve_pct: float,
):
    err = np.linalg.norm(np.asarray(pos_step3, dtype=float) - np.asarray(pos_ref, dtype=float), axis=1)
    ranges = _segment_ranges_by_time(
        t_ref,
        segment_duration_s=segment_duration_s,
        overlap_ratio=overlap_ratio,
        min_samples=min_samples,
    )

    seg_rmse = []
    for s, e in ranges:
        if e - s < max(3, int(min_samples)):
            continue
        seg_rmse.append(float(np.sqrt(np.mean(err[s:e] ** 2))))
    seg_rmse = np.asarray(seg_rmse, dtype=float)
    seg_rmse_mean = float(np.nanmean(seg_rmse)) if seg_rmse.size else float(np.nan)
    seg_rmse_std = float(np.nanstd(seg_rmse)) if seg_rmse.size else float(np.nan)
    seg_rmse_cv = float(seg_rmse_std / (seg_rmse_mean + 1e-12)) if seg_rmse.size else float(np.nan)

    heading_deg = _segment_heading_angles_deg(pos_ref, pos_step3)
    heading_p90 = float(np.percentile(heading_deg, 90)) if heading_deg.size else float(np.nan)
    heading_median = float(np.median(heading_deg)) if heading_deg.size else float(np.nan)

    improve_pct = float((raw_rmse_m - step3_rmse_m) / (raw_rmse_m + 1e-12) * 100.0)
    is_good = (
        np.isfinite(step3_rmse_m)
        and np.isfinite(seg_rmse_cv)
        and np.isfinite(heading_p90)
        and step3_rmse_m <= good_rmse_m
        and seg_rmse_cv <= good_seg_cv
        and heading_p90 <= good_heading_p90_deg
    )
    is_partial = (
        np.isfinite(step3_rmse_m)
        and np.isfinite(improve_pct)
        and (step3_rmse_m <= partial_rmse_m or improve_pct >= partial_min_improve_pct)
    )
    if is_good:
        label = "good_align"
        label_code = 2.0
    elif is_partial:
        label = "partial_align"
        label_code = 1.0
    else:
        label = "poor_align"
        label_code = 0.0

    return {
        "quality_label_code": float(label_code),
        "step3_rmse_m": float(step3_rmse_m),
        "raw_to_step3_improve_pct": float(improve_pct),
        "segment_count": float(seg_rmse.size),
        "segment_rmse_mean_m": float(seg_rmse_mean),
        "segment_rmse_std_m": float(seg_rmse_std),
        "segment_rmse_cv": float(seg_rmse_cv),
        "heading_median_deg": float(heading_median),
        "heading_p90_deg": float(heading_p90),
        "_quality_label": label,
    }


def _compute_rigid_alignability(
    *,
    t_ref,
    pos_ref,
    pos_step2,
    pos_step3,
    segment_duration_s: float,
    overlap_ratio: float,
    min_samples: int,
    max_path_ratio: float,
    max_bbox_ratio: float,
    max_global_local_ratio: float,
    max_sim3_gain_ratio: float,
):
    pref = np.asarray(pos_ref, dtype=float)
    p2 = np.asarray(pos_step2, dtype=float)
    p3 = np.asarray(pos_step3, dtype=float)
    if pref.shape != p2.shape or pref.shape != p3.shape:
        raise ValueError("Rigid alignability check requires pos_ref/pos_step2/pos_step3 with equal shape.")
    if pref.shape[0] < 3:
        raise ValueError("Rigid alignability check requires at least 3 samples.")

    path_ref = cum_distance(pref)[-1] if pref.shape[0] > 1 else 0.0
    path_est = cum_distance(p2)[-1] if p2.shape[0] > 1 else 0.0
    path_ratio_raw = float(path_est / (path_ref + 1e-12))
    path_ratio_sym = float(max(path_ratio_raw, 1.0 / (path_ratio_raw + 1e-12)))

    bbox_ref = float(np.linalg.norm(np.max(pref, axis=0) - np.min(pref, axis=0)))
    bbox_est = float(np.linalg.norm(np.max(p2, axis=0) - np.min(p2, axis=0)))
    bbox_ratio_raw = float(bbox_est / (bbox_ref + 1e-12))
    bbox_ratio_sym = float(max(bbox_ratio_raw, 1.0 / (bbox_ratio_raw + 1e-12)))

    err_global = np.linalg.norm(p3 - pref, axis=1)
    ranges = _segment_ranges_by_time(
        t_ref,
        segment_duration_s=segment_duration_s,
        overlap_ratio=overlap_ratio,
        min_samples=min_samples,
    )
    local_rmse = []
    global_seg_rmse = []
    for s, e in ranges:
        if e - s < max(3, int(min_samples)):
            continue
        try:
            R_seg, t_seg = solve_world_alignment(p2[s:e], pref[s:e])
        except Exception:
            continue
        pred_local = (R_seg @ p2[s:e].T).T + t_seg
        local_rmse.append(float(np.sqrt(np.mean(np.sum((pred_local - pref[s:e]) ** 2, axis=1)))))
        global_seg_rmse.append(float(np.sqrt(np.mean(err_global[s:e] ** 2))))
    local_rmse = np.asarray(local_rmse, dtype=float)
    global_seg_rmse = np.asarray(global_seg_rmse, dtype=float)

    if local_rmse.size and global_seg_rmse.size:
        global_local_ratio = float(np.median(global_seg_rmse / (local_rmse + 1e-12)))
        local_median = float(np.median(local_rmse))
        global_median = float(np.median(global_seg_rmse))
    else:
        global_local_ratio = float(np.nan)
        local_median = float(np.nan)
        global_median = float(np.nan)

    sim3_rmse = float(np.nan)
    sim3_scale = float(np.nan)
    sim3_gain_ratio = float(np.nan)
    sim3_scale_log10_abs = float(np.nan)
    try:
        s_sim3, R_sim3, t_sim3 = _umeyama_transform(p2, pref, with_scale=True)
        pred_sim3 = s_sim3 * (R_sim3 @ p2.T).T + t_sim3
        sim3_err = np.linalg.norm(pred_sim3 - pref, axis=1)
        sim3_rmse = float(np.sqrt(np.mean(sim3_err**2)))
        se3_rmse = float(np.sqrt(np.mean(err_global**2)))
        sim3_scale = float(s_sim3)
        sim3_gain_ratio = float((se3_rmse - sim3_rmse) / (se3_rmse + 1e-12))
        if sim3_scale > 0.0:
            sim3_scale_log10_abs = float(abs(np.log10(sim3_scale)))
    except Exception:
        pass

    severe_scale_mismatch = bool(
        (np.isfinite(sim3_scale_log10_abs) and sim3_scale_log10_abs >= 1.0)
        or (np.isfinite(path_ratio_sym) and path_ratio_sym >= 20.0)
        or (np.isfinite(bbox_ratio_sym) and bbox_ratio_sym >= 20.0)
    )

    failures = []
    if np.isfinite(path_ratio_sym) and path_ratio_sym > float(max_path_ratio):
        failures.append("path_ratio")
    if np.isfinite(bbox_ratio_sym) and bbox_ratio_sym > float(max_bbox_ratio):
        failures.append("bbox_ratio")
    if np.isfinite(global_local_ratio) and global_local_ratio > float(max_global_local_ratio):
        failures.append("global_local_ratio")
    if np.isfinite(sim3_gain_ratio) and sim3_gain_ratio > float(max_sim3_gain_ratio):
        failures.append("sim3_gain")
    if severe_scale_mismatch:
        failures.append("scale_mismatch_severe")

    alignable = len(failures) == 0
    label = "rigidly_alignable" if alignable else "not_rigidly_alignable"

    return {
        "rigid_alignability_code": 1.0 if alignable else 0.0,
        "path_length_ratio_sym": float(path_ratio_sym),
        "bbox_diag_ratio_sym": float(bbox_ratio_sym),
        "segment_local_se3_rmse_median_m": float(local_median),
        "segment_global_se3_rmse_median_m": float(global_median),
        "segment_global_local_rmse_ratio": float(global_local_ratio),
        "sim3_rmse_m": float(sim3_rmse),
        "sim3_scale": float(sim3_scale),
        "sim3_scale_log10_abs": float(sim3_scale_log10_abs),
        "sim3_gain_ratio": float(sim3_gain_ratio),
        "scale_mismatch_severe_code": 1.0 if severe_scale_mismatch else 0.0,
        "rigid_check_fail_count": float(len(failures)),
        "_rigid_alignability_label": label,
        "_rigid_alignability_reasons": ",".join(failures),
    }


def _compute_piecewise_diagnostics(
    *,
    t_ref,
    pos_ref,
    pos_step2,
    pos_step3,
    segment_duration_s: float,
    overlap_ratio: float,
    min_samples: int,
):
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    pref = np.asarray(pos_ref, dtype=float)
    p2 = np.asarray(pos_step2, dtype=float)
    p3 = np.asarray(pos_step3, dtype=float)

    ranges = _segment_ranges_by_time(
        t_ref,
        segment_duration_s=segment_duration_s,
        overlap_ratio=overlap_ratio,
        min_samples=min_samples,
    )

    seg_mid_s = []
    seg_global_rmse = []
    seg_local_rmse = []
    for s, e in ranges:
        if e - s < max(3, int(min_samples)):
            continue

        mid = 0.5 * (float(t_ref[s]) + float(t_ref[e - 1])) - float(t_ref[0])
        seg_mid_s.append(mid)

        err_g = np.linalg.norm(p3[s:e] - pref[s:e], axis=1)
        seg_global_rmse.append(float(np.sqrt(np.mean(err_g**2))))

        try:
            R_seg, t_seg = solve_world_alignment(p2[s:e], pref[s:e])
            pred_local = (R_seg @ p2[s:e].T).T + t_seg
            err_l = np.linalg.norm(pred_local - pref[s:e], axis=1)
            seg_local_rmse.append(float(np.sqrt(np.mean(err_l**2))))
        except Exception:
            seg_local_rmse.append(float(np.nan))

    seg_mid_s = np.asarray(seg_mid_s, dtype=float)
    seg_global_rmse = np.asarray(seg_global_rmse, dtype=float)
    seg_local_rmse = np.asarray(seg_local_rmse, dtype=float)

    valid = np.isfinite(seg_global_rmse) & np.isfinite(seg_local_rmse)
    if np.any(valid):
        g = seg_global_rmse[valid]
        l = seg_local_rmse[valid]
        gap = g - l
        ratio = g / (l + 1e-12)

        peak_idx = int(np.argmax(g))
        if g.size >= 2 and peak_idx < g.size - 1:
            post_best = float(np.min(g[peak_idx + 1 :]))
            recovery_pct = float((g[peak_idx] - post_best) / (g[peak_idx] + 1e-12) * 100.0)
        else:
            recovery_pct = float(np.nan)

        q = max(1, g.size // 4)
        early = float(np.mean(g[:q]))
        late = float(np.mean(g[-q:]))
        early_late_delta = late - early

        metrics = {
            "piecewise_segment_count": float(g.size),
            "piecewise_global_rmse_median_m": float(np.median(g)),
            "piecewise_local_rmse_median_m": float(np.median(l)),
            "piecewise_global_local_ratio_median": float(np.median(ratio)),
            "piecewise_gap_median_m": float(np.median(gap)),
            "piecewise_gap_p90_m": float(np.percentile(gap, 90)),
            "piecewise_gap_max_m": float(np.max(gap)),
            "piecewise_peak_recovery_pct": float(recovery_pct),
            "piecewise_early_late_delta_m": float(early_late_delta),
        }
    else:
        metrics = {
            "piecewise_segment_count": 0.0,
            "piecewise_global_rmse_median_m": float(np.nan),
            "piecewise_local_rmse_median_m": float(np.nan),
            "piecewise_global_local_ratio_median": float(np.nan),
            "piecewise_gap_median_m": float(np.nan),
            "piecewise_gap_p90_m": float(np.nan),
            "piecewise_gap_max_m": float(np.nan),
            "piecewise_peak_recovery_pct": float(np.nan),
            "piecewise_early_late_delta_m": float(np.nan),
        }

    details = {
        "segment_mid_s": seg_mid_s,
        "segment_global_rmse_m": seg_global_rmse,
        "segment_local_rmse_m": seg_local_rmse,
    }
    return metrics, details


def _diagnosis_tags_from_metrics(
    *,
    success: dict,
    time_metrics: dict,
    traj_metrics: dict,
    step3_selection: dict,
    alignment_quality: dict,
    rigid_alignability: dict,
    orientation: dict,
) -> dict:
    tags = []
    reasons = {}

    def finite_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(np.nan)

    def strict_bool(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return False

    xcorr_peak = finite_float(time_metrics.get("xcorr_peak_normalized", np.nan))
    xcorr_psr = finite_float(time_metrics.get("xcorr_psr", np.nan))
    gate_value = finite_float(success.get("global_gate_value_m", np.nan))
    gate_m = finite_float(success.get("global_gate_m", np.nan))
    stable_used = finite_float(step3_selection.get("step3_stable_segment_used", np.nan))
    stable_ratio = finite_float(step3_selection.get("step3_stable_solve_ratio", np.nan))
    sr_distance = finite_float(success.get("success_rate_distance", np.nan))
    case_status = str(success.get("case_status", ""))
    quality_label = str(alignment_quality.get("_quality_label", ""))
    rigid_label = str(rigid_alignability.get("_rigid_alignability_label", ""))
    rigid_reasons = str(rigid_alignability.get("_rigid_alignability_reasons", ""))
    orientation_unstable = strict_bool(orientation.get("orientation_unstable", False))
    orientation_warning = str(orientation.get("orientation_warning", "") or "").strip()
    piecewise_ratio = finite_float(rigid_alignability.get("segment_global_local_rmse_ratio", np.nan))
    sim3_scale = finite_float(rigid_alignability.get("sim3_scale", np.nan))
    low_performance = case_status in {"globally_unstable", "globally_failed"} or (
        np.isfinite(sr_distance) and sr_distance < 0.2
    )

    if (np.isfinite(xcorr_peak) and xcorr_peak < 0.75) or (np.isfinite(xcorr_psr) and xcorr_psr < 6.0):
        tags.append("time_alignment_weak")
        reasons.setdefault("time_alignment_weak", "weak xcorr peak/psr")

    if low_performance and (
        (np.isfinite(stable_ratio) and stable_ratio < 0.15)
        or (np.isfinite(stable_used) and stable_used < 0.5 and not np.isfinite(stable_ratio))
    ):
        tags.append("step3_no_reliable_stable_segment")
        reasons.setdefault("step3_no_reliable_stable_segment", "no stable segment or too little stable solve coverage")

    if (
        case_status in {"globally_unstable", "globally_failed"}
        and np.isfinite(gate_value)
        and np.isfinite(gate_m)
        and gate_value > gate_m
    ):
        tags.append("global_gate_too_large")
        reasons.setdefault("global_gate_too_large", f"global gate value {gate_value:.3g} exceeds gate {gate_m:.3g}")

    if case_status in {"globally_unstable", "globally_failed"} and np.isfinite(piecewise_ratio) and piecewise_ratio > 4.0:
        tags.append("trajectory_jump")
        reasons.setdefault("trajectory_jump", "global/local segment error ratio is large")

    if (
        ("scale_mismatch_severe" in str(rigid_reasons))
        or (np.isfinite(sim3_scale) and (sim3_scale <= 0.1 or sim3_scale >= 10.0))
    ):
        tags.append("scale_or_unit_suspect")
        reasons.setdefault("scale_or_unit_suspect", "scale mismatch indicators are strong")

    if rigid_label != "rigidly_alignable" and (
        "path_ratio" in rigid_reasons
        or "bbox_ratio" in rigid_reasons
        or "global_local_ratio" in rigid_reasons
        or "sim3_gain" in rigid_reasons
    ):
        tags.append("gt_mapping_suspect")
        reasons.setdefault("gt_mapping_suspect", "rigid alignability checks failed")

    if orientation_unstable or orientation_warning:
        tags.append("orientation_unstable")
        reasons.setdefault("orientation_unstable", orientation_warning or "rotation error unstable")

    if quality_label == "poor_align" and case_status in {"globally_unstable", "globally_failed"}:
        tags.append("alignment_poor")
        reasons.setdefault("alignment_poor", "step3 remains poor after diagnostics")

    tags = list(dict.fromkeys(tags))
    return {
        "diagnosis_tags": tags,
        "diagnosis_reasons": reasons,
        "diagnosis_summary": "; ".join(tags),
        "diagnosis_primary": tags[0] if tags else "",
        "diagnosis_count": float(len(tags)),
    }


def _build_user_alert(
    *,
    time_metrics,
    traj_metrics,
    quality_label: str,
    rigid_label: str,
    rigid_reasons: str,
):
    issues = []
    peak = float(time_metrics.get("xcorr_peak_normalized", np.nan))
    psr = float(time_metrics.get("xcorr_psr", np.nan))
    match_ratio = float(time_metrics.get("evo_matches_ratio_equivalent", np.nan))
    omega_gain = float(time_metrics.get("omega_rmse_improve_pct", np.nan))
    offset_est = float(time_metrics.get("offset_est_s", np.nan))
    step1_forced_candidate = float(time_metrics.get("step1_forced_candidate_code", 0.0))
    step3_rmse = float(traj_metrics.get("ate_rmse_step3_m", np.nan))
    improve_pct = float(traj_metrics.get("ate_rmse_improve_raw_to_step3_pct", np.nan))
    strong_final_alignment = (
        np.isfinite(step3_rmse)
        and np.isfinite(improve_pct)
        and step3_rmse <= 0.3
        and improve_pct >= 80.0
    )

    if (np.isfinite(peak) and peak < 0.75) or (np.isfinite(psr) and psr < 6.0):
        issues.append("time_alignment_low_confidence")
    if step1_forced_candidate > 0.5:
        issues.append("time_alignment_forced_candidate")
    if np.isfinite(match_ratio) and match_ratio < 0.5:
        issues.append("time_overlap_low")
    if np.isfinite(omega_gain) and omega_gain < 5.0 and np.isfinite(offset_est) and abs(offset_est) > 0.02:
        issues.append("time_alignment_weak_gain")
    if quality_label == "partial_align":
        if not strong_final_alignment:
            issues.append("alignment_partial")
    elif quality_label == "poor_align":
        issues.append("alignment_poor")
    if rigid_label != "rigidly_alignable":
        if not strong_final_alignment:
            issues.append("rigid_diagnostic_flag")
    rigid_reason_tokens = {tok.strip() for tok in str(rigid_reasons).split(",") if tok.strip()}
    if "scale_mismatch_severe" in rigid_reason_tokens:
        issues.append("scale_mismatch_severe")
    if np.isfinite(step3_rmse) and step3_rmse > 10.0:
        issues.append("step3_rmse_high")
    if np.isfinite(improve_pct) and improve_pct < 10.0:
        issues.append("step3_improvement_small")

    if strong_final_alignment:
        issues = [x for x in issues if x in {"scale_mismatch_severe", "step3_rmse_high"}]

    critical = any(
        item in issues
        for item in (
            "alignment_poor",
            "scale_mismatch_severe",
            "step3_rmse_high",
        )
    )
    warning = (not critical) and len(issues) > 0

    if critical:
        level = "critical"
        level_code = 2.0
    elif warning:
        level = "warning"
        level_code = 1.0
    else:
        level = "ok"
        level_code = 0.0

    reason_map = {
        "time_alignment_low_confidence": "Weak time-alignment confidence",
        "time_alignment_forced_candidate": "Step-1 gating failed; continuing with the highest-confidence candidate offset",
        "time_overlap_low": "Low effective time overlap",
        "time_alignment_weak_gain": "Limited improvement from time alignment",
        "alignment_partial": "Residual local misalignment remains",
        "alignment_poor": "Overall alignment quality is poor",
        "rigid_diagnostic_flag": "Rigid-diagnostic checks flagged potential model mismatch",
        "scale_mismatch_severe": "Severe scale mismatch between trajectory and ground truth",
        "step3_rmse_high": "High absolute Step-3 error",
        "step3_improvement_small": "Limited Step-3 improvement over raw",
    }
    reasons_en = [reason_map[item] for item in issues if item in reason_map]

    if level == "ok":
        message_en = "The result is stable overall; Step-3 output is reliable."
    elif level == "warning":
        message_en = "Suspicious indicators were detected; please review plots and raw data."
    else:
        if "scale_mismatch_severe" in issues:
            message_en = "Critical issue: VIO and ground truth differ by orders of magnitude in scale; Step-3 may be unreliable."
        else:
            message_en = "Critical issue detected; Step-3 may be unreliable. Check data pairing and time alignment first."

    return {
        "alert_level_code": float(level_code),
        "alert_count": float(len(issues)),
        "_alert_level": level,
        "_alert_message": message_en,
        "_alert_reasons": "; ".join(reasons_en),
    }


def _segment_blend_weights(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=float)
    if length == 1:
        return np.ones(1, dtype=float)
    x = np.linspace(-1.0, 1.0, int(length))
    return np.maximum(1e-3, 1.0 - np.abs(x))


def _compute_piecewise_alignment(
    *,
    t_ref,
    pos_ref,
    pos_base,
    quat_base,
    fallback_pos,
    fallback_quat,
    segment_duration_s: float,
    overlap_ratio: float,
    min_samples: int,
    with_scale: bool = True,
):
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    pref = np.asarray(pos_ref, dtype=float)
    pbase = np.asarray(pos_base, dtype=float)
    qbase = normalize_quat_array(np.asarray(quat_base, dtype=float))
    pfallback = np.asarray(fallback_pos, dtype=float)
    qfallback = normalize_quat_array(np.asarray(fallback_quat, dtype=float))

    if pref.shape != pbase.shape or pref.shape[0] < 3:
        return None

    ranges = _segment_ranges_by_time(
        t_ref,
        segment_duration_s=segment_duration_s,
        overlap_ratio=overlap_ratio,
        min_samples=min_samples,
    )
    if not ranges:
        return None

    n = pref.shape[0]
    pos_acc = np.zeros((n, 3), dtype=float)
    quat_acc = np.zeros((n, 4), dtype=float)
    weight_acc = np.zeros(n, dtype=float)
    seg_rmse = []
    valid_segments = 0

    for s, e in ranges:
        if e - s < max(3, int(min_samples)):
            continue
        try:
            scale_seg, R_seg, t_seg = _umeyama_transform(pbase[s:e], pref[s:e], with_scale=with_scale)
        except Exception:
            continue

        pred_seg = scale_seg * (R_seg @ pbase[s:e].T).T + t_seg
        q_seg = normalize_quat_array((R.from_matrix(R_seg) * R.from_quat(qbase[s:e])).as_quat())
        w_seg = _segment_blend_weights(e - s)

        pos_acc[s:e] += w_seg[:, None] * pred_seg
        for local_idx, global_idx in enumerate(range(s, e)):
            q_curr = np.asarray(q_seg[local_idx], dtype=float)
            if weight_acc[global_idx] > 0.0 and np.dot(quat_acc[global_idx], q_curr) < 0.0:
                q_curr = -q_curr
            quat_acc[global_idx] += w_seg[local_idx] * q_curr
        weight_acc[s:e] += w_seg
        seg_rmse.append(float(np.sqrt(np.mean(np.sum((pred_seg - pref[s:e]) ** 2, axis=1)))))
        valid_segments += 1

    if valid_segments == 0:
        return None

    pos_out = np.asarray(pfallback, dtype=float).copy()
    quat_out = np.asarray(qfallback, dtype=float).copy()
    valid = weight_acc > 1e-9
    if not np.any(valid):
        return None

    pos_out[valid] = pos_acc[valid] / weight_acc[valid, None]
    quat_norm = np.linalg.norm(quat_acc, axis=1)
    quat_valid = valid & (quat_norm > 1e-9)
    if np.any(quat_valid):
        quat_out[quat_valid] = quat_acc[quat_valid] / quat_norm[quat_valid, None]
    quat_out = normalize_quat_array(quat_out)

    err = np.linalg.norm(pos_out - pref, axis=1)
    seg_rmse = np.asarray(seg_rmse, dtype=float)
    return {
        "pos": pos_out,
        "quat": quat_out,
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "p95_m": float(np.percentile(err, 95)),
        "coverage_ratio": float(np.mean(valid)),
        "segment_count": float(valid_segments),
        "segment_rmse_median_m": float(np.median(seg_rmse)) if seg_rmse.size else float("nan"),
        "mode": "piecewise_local_sim3" if with_scale else "piecewise_local_se3",
    }
