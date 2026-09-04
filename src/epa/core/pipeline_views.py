from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..alignment.modes import public_align_mode, resolve_metric_eval_align_mode
from .evaluation import apply_input_coverage_to_success
from ..metric_cli_common import cum_distance


def _finite_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _fmt_ov_eval_terminal(value, nd: int = 3) -> str:
    val = _finite_float(value)
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{int(nd)}f}"


def _int_or_zero(value) -> int:
    val = _finite_float(value, default=0.0)
    return int(val) if np.isfinite(val) else 0


def _stats_block(metrics: dict, key: str) -> dict:
    block = metrics.get(key, {}) if isinstance(metrics, dict) else {}
    return block if isinstance(block, dict) else {}


def _latex_escape_terminal(text: object) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _print_sim3_ov_eval_terminal_metrics(
    *,
    pose_metrics: dict,
    ape_pose_relation: str,
    rpe_pose_relation: str,
    eval_source: str,
    t_ref: np.ndarray,
    pos_ref: np.ndarray,
    mode_label: str,
    stage: str = "step3",
) -> None:
    ape = pose_metrics.get("ape", {}).get(stage, {})
    rpe = pose_metrics.get("rpe", {}).get(stage, {})
    rpe_time = pose_metrics.get("rpe_time_1s", {}).get(stage, {})
    success = pose_metrics.get("valid_segment", {}).get(stage, {}).get("success", {})
    eval_alignment = pose_metrics.get("eval_alignment", {}).get(stage, {})

    ate_ori = _stats_block(ape, "rotation_angle_deg")
    ate_pos = _stats_block(ape, ape_pose_relation)
    if not ate_pos:
        ate_pos = _stats_block(ape, "translation_part")

    print("======================================")
    print("Absolute Trajectory Error")
    print("======================================")
    print(f"rmse_ori = {_fmt_ov_eval_terminal(ate_ori.get('rmse'))} | rmse_pos = {_fmt_ov_eval_terminal(ate_pos.get('rmse'))}")
    print(f"mean_ori = {_fmt_ov_eval_terminal(ate_ori.get('mean'))} | mean_pos = {_fmt_ov_eval_terminal(ate_pos.get('mean'))}")
    print(f"min_ori  = {_fmt_ov_eval_terminal(ate_ori.get('min'))} | min_pos  = {_fmt_ov_eval_terminal(ate_pos.get('min'))}")
    print(f"max_ori  = {_fmt_ov_eval_terminal(ate_ori.get('max'))} | max_pos  = {_fmt_ov_eval_terminal(ate_pos.get('max'))}")
    print(f"std_ori  = {_fmt_ov_eval_terminal(ate_ori.get('std'))} | std_pos  = {_fmt_ov_eval_terminal(ate_pos.get('std'))}")

    rpe_ori = _stats_block(rpe, "rotation_angle_deg")
    rpe_pos = _stats_block(rpe, rpe_pose_relation)
    if not rpe_pos:
        rpe_pos = _stats_block(rpe, "translation_part")
    rpe_config = pose_metrics.get("rpe_config", {})
    delta = rpe_config.get("delta", np.nan) if isinstance(rpe_config, dict) else np.nan
    delta_unit = str(rpe_config.get("delta_unit", "")) if isinstance(rpe_config, dict) else ""
    delta_label = "configured"
    delta_val = _finite_float(delta)
    if np.isfinite(delta_val):
        delta_label = f"{delta_val:g}{delta_unit}"

    time_ori = _stats_block(rpe_time, "rotation_angle_deg")
    time_pos = _stats_block(rpe_time, "translation_part")

    print("======================================")
    print("Relative Pose Error")
    print("======================================")
    print(
        f"{delta_label} - median_ori = {_fmt_ov_eval_terminal(rpe_ori.get('median'))} "
        f"| median_pos = {_fmt_ov_eval_terminal(rpe_pos.get('median'))} "
        f"({_int_or_zero(rpe.get('pair_count'))} samples)"
    )
    print(
        f"1s time - rmse_ori = {_fmt_ov_eval_terminal(time_ori.get('rmse'))} "
        f"| rmse_pos = {_fmt_ov_eval_terminal(time_pos.get('rmse'))} "
        f"({_int_or_zero(rpe_time.get('pair_count'))} samples)"
    )

    threshold = success.get("threshold", {}) if isinstance(success, dict) else {}
    threshold_m = _finite_float(threshold.get("threshold_m") if isinstance(threshold, dict) else np.nan)
    if not np.isfinite(threshold_m):
        threshold_m = _finite_float(success.get("threshold_m") if isinstance(success, dict) else np.nan)
    threshold_mode = str(threshold.get("mode", "") if isinstance(threshold, dict) else "")
    if not threshold_mode:
        threshold_mode = str(success.get("threshold_mode", "unknown") if isinstance(success, dict) else "unknown")
    total_distance_m = _finite_float(success.get("complete_total_distance_m") if isinstance(success, dict) else np.nan)
    duration_s = _finite_float(success.get("complete_total_time_s") if isinstance(success, dict) else np.nan)
    if not np.isfinite(total_distance_m):
        total_distance_m = _finite_float(cum_distance(pos_ref)[-1] if np.asarray(pos_ref).shape[0] > 0 else np.nan)
    if not np.isfinite(duration_s):
        tref = np.asarray(t_ref, dtype=float).reshape(-1)
        duration_s = _finite_float(tref[-1] - tref[0] if tref.size > 1 else np.nan)
    global_gate_m = _finite_float(success.get("global_gate_m") if isinstance(success, dict) else np.nan)
    global_gate_pct = _finite_float(success.get("global_gate_percentile") if isinstance(success, dict) else np.nan)
    gate_value = _finite_float(success.get("global_gate_value_m") if isinstance(success, dict) else np.nan)

    gate_desc = _fmt_ov_eval_terminal(global_gate_m, 2)
    if np.isfinite(global_gate_pct):
        gate_desc += f" / p{global_gate_pct:g}"
    print(
        f"SR config: GT path={_fmt_ov_eval_terminal(total_distance_m, 2)}m "
        f"| time={_fmt_ov_eval_terminal(duration_s, 2)}s "
        f"| threshold={_fmt_ov_eval_terminal(threshold_m, 2)}m({threshold_mode}) "
        f"| gate={gate_desc}m"
    )
    print(
        f"Coverage - path = {_fmt_ov_eval_terminal(_finite_float(success.get('path_coverage_ratio')) * 100.0, 2)}% "
        f"| time = {_fmt_ov_eval_terminal(_finite_float(success.get('temporal_coverage_ratio')) * 100.0, 2)}% "
        f"| status = {success.get('input_coverage_status', 'ok')}"
    )
    print(
        f"SR local@{_fmt_ov_eval_terminal(threshold_m, 1)}m - distance = "
        f"{_fmt_ov_eval_terminal(_finite_float(success.get('local_success_rate_distance')) * 100.0, 2)}% "
        f"| time = {_fmt_ov_eval_terminal(_finite_float(success.get('local_success_rate_time')) * 100.0, 2)}% "
        f"| valid_dist = {_fmt_ov_eval_terminal(success.get('valid_distance_m'))}/"
        f"{_fmt_ov_eval_terminal(success.get('local_total_distance_m'))}m"
    )
    print(
        f"SR complete@{_fmt_ov_eval_terminal(threshold_m, 1)}m - distance = "
        f"{_fmt_ov_eval_terminal(_finite_float(success.get('complete_success_rate_distance')) * 100.0, 2)}% "
        f"| time = {_fmt_ov_eval_terminal(_finite_float(success.get('complete_success_rate_time')) * 100.0, 2)}% "
        f"| valid_dist = {_fmt_ov_eval_terminal(success.get('valid_distance_m'))}/"
        f"{_fmt_ov_eval_terminal(success.get('complete_total_distance_m'))}m"
    )

    scale = _finite_float(eval_alignment.get("sim3_scale", eval_alignment.get("align_scale", np.nan)))
    reliable = bool(eval_alignment.get("sim3_reliable", True))
    if np.isfinite(scale):
        print(f"Sim3 scale = {_fmt_ov_eval_terminal(scale, 6)} | reliable = {str(reliable).lower()}")

    sr_status = str(success.get("sr_reliability_status", "ok") if isinstance(success, dict) else "ok")
    eval_reliable = bool(reliable and sr_status != "failed")
    print(f"Eval reliable = {str(eval_reliable).lower()}")
    warning = str(success.get("sr_warning_explanation", "") if isinstance(success, dict) else "")
    sim3_warning = str(eval_alignment.get("sim3_warning", "") if isinstance(eval_alignment, dict) else "")
    if not eval_reliable:
        warning_text = sim3_warning or warning
        if warning_text:
            print(f"[UNRELIABLE] {warning_text}")
    print("======================================")
    print(f"Aligned pairs: {_int_or_zero(eval_alignment.get('align_pair_count', ape.get('pair_count', 0)))}")
    if np.isfinite(gate_value):
        print(f"Global gate value = {_fmt_ov_eval_terminal(gate_value, 3)} m")
    solver = str(eval_alignment.get("sim3_solver", "") if isinstance(eval_alignment, dict) else "")
    if solver:
        print(f"Sim3 solver = {solver}")
    print(f"Eval source = {eval_source}")

    ate_rmse_ori = _finite_float(ate_ori.get("rmse"))
    ate_rmse_pos = _finite_float(ate_pos.get("rmse"))
    rpe_1s_rmse_ori = _finite_float(time_ori.get("rmse"))
    rpe_1s_rmse_pos = _finite_float(time_pos.get("rmse"))
    sr_dist_pct = _finite_float(success.get("success_rate_distance")) * 100.0
    sr_time_pct = _finite_float(success.get("success_rate_time")) * 100.0

    print("======================================")
    print("SINGLE-RUN LATEX TABLE")
    print("======================================")
    print(r"\begin{tabular}{lrrrrrl}")
    print(r"\hline")
    print(
        r"Mode & ATE rot. & ATE pos. & 1s RPE rot. & 1s RPE pos. & SR dist. & Reliable \\"
    )
    print(r"\hline")
    print(
        f"{_latex_escape_terminal(mode_label)} "
        f"& {_fmt_ov_eval_terminal(ate_rmse_ori)} "
        f"& {_fmt_ov_eval_terminal(ate_rmse_pos)} "
        f"& {_fmt_ov_eval_terminal(rpe_1s_rmse_ori)} "
        f"& {_fmt_ov_eval_terminal(rpe_1s_rmse_pos)} "
        f"& {_fmt_ov_eval_terminal(sr_dist_pct, 2)}\\% "
        f"& {str(eval_reliable).lower()} \\\\"
    )
    print(r"\hline")
    print(r"\end{tabular}")
    print(
        "% units: rotation=deg, position=m, SR=complete-reference drift-valid success rate; "
        f"SR time={_fmt_ov_eval_terminal(sr_time_pct, 2)}\\%, "
        f"scale={_fmt_ov_eval_terminal(scale, 6)}"
    )


def _apply_eval_alignment_from_info(pos: np.ndarray, quat: np.ndarray, info: dict) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(info, dict):
        raise ValueError("Alignment info is not available.")
    if "align_rotation_matrix" not in info or "align_translation" not in info:
        raise ValueError("Alignment transform is not stored in eval info.")
    scale = _finite_float(info.get("align_scale", 1.0), default=1.0)
    if not np.isfinite(scale):
        scale = 1.0
    r_eval = np.asarray(info["align_rotation_matrix"], dtype=float).reshape(3, 3)
    t_eval = np.asarray(info["align_translation"], dtype=float).reshape(3)
    q_est = np.asarray(quat, dtype=float)
    pos_new = scale * (r_eval @ np.asarray(pos, dtype=float).T).T + t_eval
    quat_new = (R.from_matrix(r_eval) * R.from_quat(q_est)).as_quat()
    norms = np.linalg.norm(quat_new, axis=1, keepdims=True)
    quat_new = quat_new / np.maximum(norms, 1e-12)
    return pos_new, quat_new


def _compute_orientation_diagnostics(pose_metrics: dict, stage: str = "step3") -> dict:
    ape_rot_rmse = _finite_float(
        pose_metrics.get("ape", {}).get(stage, {}).get("rotation_angle_deg", {}).get("rmse")
    )
    rpe_rot_rmse = _finite_float(
        pose_metrics.get("rpe", {}).get(stage, {}).get("rotation_angle_deg", {}).get("rmse")
    )
    rpe_time_rot_rmse = _finite_float(
        pose_metrics.get("rpe_time_1s", {}).get(stage, {}).get("rotation_angle_deg", {}).get("rmse")
    )
    success = pose_metrics.get("valid_segment", {}).get(stage, {}).get("success", {})
    sr_distance = _finite_float(success.get("success_rate_distance"))

    min_sr = 0.99
    ape_warning_threshold = 10.0
    ape_threshold = 45.0
    rpe_warning_threshold = 10.0
    rpe_threshold = 30.0
    rpe_time_warning_threshold = 10.0
    rpe_time_threshold = 30.0
    high_translation_sr = bool(np.isfinite(sr_distance) and sr_distance >= min_sr)
    rotation_bad = bool(
        (np.isfinite(ape_rot_rmse) and ape_rot_rmse >= ape_threshold)
        or (np.isfinite(rpe_rot_rmse) and rpe_rot_rmse >= rpe_threshold)
        or (np.isfinite(rpe_time_rot_rmse) and rpe_time_rot_rmse >= rpe_time_threshold)
    )
    rotation_warning = bool(
        (np.isfinite(ape_rot_rmse) and ape_rot_rmse >= ape_warning_threshold)
        or (np.isfinite(rpe_rot_rmse) and rpe_rot_rmse >= rpe_warning_threshold)
        or (np.isfinite(rpe_time_rot_rmse) and rpe_time_rot_rmse >= rpe_time_warning_threshold)
    )
    status = "critical" if rotation_bad else ("warning" if rotation_warning else "ok")
    inconsistent = high_translation_sr and rotation_bad
    unstable = rotation_bad
    warning = ""
    if inconsistent:
        warning = (
            "Translation SR is high but rotation error is unstable; "
            "SR is unchanged because it is translation-only."
        )
    elif rotation_warning:
        warning = "Rotation error exceeds the orientation reliability threshold."

    return {
        "orientation_unstable": bool(unstable),
        "orientation_status": status,
        "orientation_inconsistent_with_translation_sr": bool(inconsistent),
        "orientation_warning": warning,
        "orientation_stage": stage,
        "orientation_min_sr_for_warning": min_sr,
        "orientation_ape_rmse_deg": ape_rot_rmse,
        "orientation_ape_rmse_warning_deg": ape_warning_threshold,
        "orientation_ape_rmse_threshold_deg": ape_threshold,
        "orientation_rpe_rmse_deg": rpe_rot_rmse,
        "orientation_rpe_rmse_warning_deg": rpe_warning_threshold,
        "orientation_rpe_rmse_threshold_deg": rpe_threshold,
        "orientation_rpe_time_1s_rmse_deg": rpe_time_rot_rmse,
        "orientation_rpe_time_1s_rmse_warning_deg": rpe_time_warning_threshold,
        "orientation_rpe_time_1s_rmse_threshold_deg": rpe_time_threshold,
        "orientation_translation_sr_distance": sr_distance,
    }


def _interactive_view_metric_summary(
    pose_metrics: dict,
    *,
    stage: str,
    ape_relation: str,
    rpe_relation: str,
) -> dict:
    def finite_or_none(value) -> float | None:
        val = _finite_float(value)
        return float(val) if np.isfinite(val) else None

    success = pose_metrics.get("valid_segment", {}).get(stage, {}).get("success", {})
    eval_info = pose_metrics.get("eval_alignment", {}).get(stage, {})
    ape_trans_rmse_m = finite_or_none(
        pose_metrics.get("ape", {}).get(stage, {}).get(ape_relation, {}).get("rmse")
    )
    rpe_trans_rmse_m = finite_or_none(
        pose_metrics.get("rpe", {}).get(stage, {}).get(rpe_relation, {}).get("rmse")
    )
    rpe_time_1s_trans_rmse_m = finite_or_none(
        pose_metrics.get("rpe_time_1s", {}).get(stage, {}).get("translation_part", {}).get("rmse")
    )
    sim3_audit = _audit_sim3_full_trajectory(
        success=success,
        eval_info=eval_info,
        ape_trans_rmse_m=ape_trans_rmse_m,
        rpe_trans_rmse_m=rpe_trans_rmse_m,
        rpe_time_1s_trans_rmse_m=rpe_time_1s_trans_rmse_m,
    )
    return {
        "case_status": str(success.get("case_status", "")),
        "sr_distance": finite_or_none(success.get("success_rate_distance")),
        "sr_time": finite_or_none(success.get("success_rate_time")),
        "local_sr_distance": finite_or_none(success.get("local_success_rate_distance")),
        "local_sr_time": finite_or_none(success.get("local_success_rate_time")),
        "complete_sr_distance": finite_or_none(success.get("complete_success_rate_distance")),
        "complete_sr_time": finite_or_none(success.get("complete_success_rate_time")),
        "path_coverage_ratio": finite_or_none(success.get("path_coverage_ratio")),
        "temporal_coverage_ratio": finite_or_none(success.get("temporal_coverage_ratio")),
        "raw_sr_distance": finite_or_none(success.get("raw_success_rate_distance")),
        "raw_sr_time": finite_or_none(success.get("raw_success_rate_time")),
        "sr_reliability_status": str(success.get("sr_reliability_status", "")),
        "sr_warning_explanation": str(success.get("sr_warning_explanation", "")),
        "global_gate_failed": bool(success.get("global_gate_failed", False)),
        "global_gate_value_m": finite_or_none(success.get("global_gate_value_m")),
        "valid_distance_m": finite_or_none(success.get("valid_distance_m")),
        "total_distance_m": finite_or_none(success.get("total_distance_m")),
        "threshold_m": finite_or_none(success.get("threshold", {}).get("threshold_m")),
        "drift_threshold_mode": str(success.get("drift_threshold_mode", "")),
        "drift_rpe_1s_m": finite_or_none(success.get("drift_rpe_1s_m")),
        "drift_ape_slope_mps": finite_or_none(success.get("drift_ape_slope_mps")),
        "drift_ape_jump_m": finite_or_none(success.get("drift_ape_jump_m")),
        "ape_trans_rmse_m": ape_trans_rmse_m,
        "ape_rot_rmse_deg": finite_or_none(
            pose_metrics.get("ape", {}).get(stage, {}).get("rotation_angle_deg", {}).get("rmse")
        ),
        "rpe_trans_rmse_m": rpe_trans_rmse_m,
        "rpe_rot_rmse_deg": finite_or_none(
            pose_metrics.get("rpe", {}).get(stage, {}).get("rotation_angle_deg", {}).get("rmse")
        ),
        "rpe_time_1s_trans_rmse_m": rpe_time_1s_trans_rmse_m,
        "rpe_time_1s_rot_rmse_deg": finite_or_none(
            pose_metrics.get("rpe_time_1s", {}).get(stage, {}).get("rotation_angle_deg", {}).get("rmse")
        ),
        "align_scale": finite_or_none(eval_info.get("align_scale")),
        "sim3_scale": finite_or_none(eval_info.get("sim3_scale")),
        "sim3_scale_reliable": bool(eval_info.get("sim3_scale_reliable", True)),
        "sim3_scale_warning": bool(eval_info.get("sim3_scale_warning", False)),
        "sim3_scale_severe": bool(eval_info.get("sim3_scale_severe", False)),
        "sim3_warning_level": str(eval_info.get("sim3_warning_level", "")),
        "sim3_solver": str(eval_info.get("sim3_solver", "")),
        "sim3_anchor_samples": int(eval_info.get("sim3_anchor_samples", 0) or 0),
        "sim3_anchor_status": str(eval_info.get("sim3_anchor_status", "")),
        "sim3_consensus_status": str(eval_info.get("sim3_consensus_status", "")),
        "sim3_confidence": str(eval_info.get("sim3_confidence", "")),
        "sim3_consensus_count": int(eval_info.get("sim3_consensus_count", 0) or 0),
        "sim3_candidate_reliable_count": int(eval_info.get("sim3_candidate_reliable_count", 0) or 0),
        "sim3_stable_anchor_used": bool(eval_info.get("sim3_stable_anchor_used", False)),
        "sim3_fallback_used": bool(eval_info.get("sim3_robust_fallback_used", False)),
        "sim3_fallback_reason": str(eval_info.get("sim3_robust_fallback_reason", "")),
        "sim3_anchor_reliable": bool(eval_info.get("sim3_reliable", True)),
        "sim3_reliable": bool(sim3_audit.get("sim3_eval_reliable", True)),
        **sim3_audit,
    }


def _default_interactive_view(requested_eval_align_mode: str) -> str:
    requested = public_align_mode(requested_eval_align_mode, default="se3")
    if requested == "sim3":
        return "sim3"
    if requested == "posyaw":
        return "epa_posyaw"
    return "step3"


def _interactive_eval_view_specs(requested_eval_align_mode: str) -> tuple[tuple[str, str, str], ...]:
    requested = public_align_mode(requested_eval_align_mode, default="se3")
    if requested == "se3":
        return ()
    if requested == "posyaw":
        return (("epa_posyaw", "posyaw", "posyaw"),)
    if requested == "sim3":
        return (("sim3", "sim3", "sim3"),)
    return ()


def _public_eval_align_mode(raw_mode: str) -> str:
    mode = resolve_metric_eval_align_mode(
        raw_mode,
        default="none",
        collapse_sim3_aliases=True,
        legacy_origin=False,
    )
    if mode in {"", "none"}:
        return "se3"
    return public_align_mode(mode, default=mode)


def _audit_sim3_full_trajectory(
    *,
    success: dict,
    eval_info: dict,
    ape_trans_rmse_m: float | None,
    rpe_trans_rmse_m: float | None,
    rpe_time_1s_trans_rmse_m: float | None,
) -> dict:
    align_mode = str(eval_info.get("align_mode", "")).lower()
    solver = str(eval_info.get("sim3_solver", "")).lower()
    is_sim3 = bool("sim3" in align_mode or "sim3" in solver)
    if not is_sim3:
        return {
            "sim3_full_trajectory_status": "",
            "sim3_eval_reliable": True,
            "sim3_may_mask_failure": False,
            "sim3_failure_reason": "",
            "sim3_audit_tags": "",
        }

    def finite(value) -> float | None:
        val = _finite_float(value)
        return float(val) if np.isfinite(val) else None

    hard: list[str] = []
    soft: list[str] = []
    anchor_reliable = bool(eval_info.get("sim3_reliable", True))
    scale_severe = bool(eval_info.get("sim3_scale_severe", False))
    scale_warning = bool(eval_info.get("sim3_scale_warning", False))
    sr_distance = finite(success.get("success_rate_distance"))
    threshold_m = finite(success.get("threshold", {}).get("threshold_m"))
    drift_rpe_1s_m = finite(success.get("drift_rpe_1s_m"))
    global_gate_failed = bool(success.get("global_gate_failed", False))

    if not anchor_reliable:
        hard.append("anchor_unreliable")
    if scale_severe:
        hard.append("scale_severe")
    elif scale_warning:
        soft.append("scale_warning")
    if global_gate_failed:
        hard.append("global_gate_failed")

    if threshold_m is not None and ape_trans_rmse_m is not None:
        if ape_trans_rmse_m > max(4.0 * threshold_m, 20.0):
            hard.append("full_ape_extreme")
        elif ape_trans_rmse_m > threshold_m:
            soft.append("full_ape_above_threshold")
    if drift_rpe_1s_m is not None and rpe_time_1s_trans_rmse_m is not None:
        if rpe_time_1s_trans_rmse_m > max(4.0 * drift_rpe_1s_m, 5.0):
            hard.append("full_1s_rpe_extreme")
        elif rpe_time_1s_trans_rmse_m > 2.0 * drift_rpe_1s_m:
            soft.append("full_1s_rpe_high")
    if sr_distance is not None:
        if sr_distance < 0.8:
            hard.append("sr_distance_low")
        elif sr_distance < 0.9:
            soft.append("sr_distance_warn")

    hard_unique = list(dict.fromkeys(hard))
    soft_unique = list(dict.fromkeys(soft))
    status = "failed" if hard_unique else ("warning" if soft_unique else "ok")
    tags = hard_unique + soft_unique
    may_mask_failure = bool(scale_severe or ("scale_warning" in soft_unique and sr_distance is not None and sr_distance >= 0.75))
    return {
        "sim3_full_trajectory_status": status,
        "sim3_eval_reliable": status != "failed",
        "sim3_may_mask_failure": may_mask_failure,
        "sim3_failure_reason": "; ".join(hard_unique),
        "sim3_audit_tags": "; ".join(tags),
        "sim3_full_ape_threshold_m": threshold_m,
        "sim3_full_rpe_1s_threshold_m": drift_rpe_1s_m,
        "sim3_full_rpe_rmse_m": rpe_trans_rmse_m,
    }


def _annotate_sr_reliability(
    pose_metrics: dict,
    *,
    stage: str,
    orientation: dict,
    ape_relation: str,
    rpe_relation: str,
    input_coverage: dict | None = None,
) -> dict:
    success = pose_metrics.get("valid_segment", {}).get(stage, {}).get("success", {})
    if not isinstance(success, dict):
        return {}
    apply_input_coverage_to_success(success, input_coverage, set_primary=True)
    eval_info = pose_metrics.get("eval_alignment", {}).get(stage, {})
    ape_trans_rmse_m = _finite_float(
        pose_metrics.get("ape", {}).get(stage, {}).get(ape_relation, {}).get("rmse")
    )
    rpe_trans_rmse_m = _finite_float(
        pose_metrics.get("rpe", {}).get(stage, {}).get(rpe_relation, {}).get("rmse")
    )
    rpe_time_1s_trans_rmse_m = _finite_float(
        pose_metrics.get("rpe_time_1s", {}).get(stage, {}).get("translation_part", {}).get("rmse")
    )
    sim3_audit = _audit_sim3_full_trajectory(
        success=success,
        eval_info=eval_info if isinstance(eval_info, dict) else {},
        ape_trans_rmse_m=float(ape_trans_rmse_m) if np.isfinite(ape_trans_rmse_m) else None,
        rpe_trans_rmse_m=float(rpe_trans_rmse_m) if np.isfinite(rpe_trans_rmse_m) else None,
        rpe_time_1s_trans_rmse_m=(
            float(rpe_time_1s_trans_rmse_m) if np.isfinite(rpe_time_1s_trans_rmse_m) else None
        ),
    )

    hard_reasons: list[str] = []
    soft_reasons: list[str] = []

    input_coverage = input_coverage if isinstance(input_coverage, dict) else {}
    coverage_status = str(input_coverage.get("coverage_status", "ok"))
    if coverage_status == "failed":
        hard_reasons.append("input_coverage_failed")
    elif coverage_status == "warning":
        soft_reasons.append("input_coverage_warning")

    if bool(success.get("global_gate_failed", False)):
        hard_reasons.append("global_gate_failed")
    orientation_status = str(orientation.get("orientation_status", ""))
    if orientation_status == "critical" or bool(orientation.get("orientation_unstable", False)):
        hard_reasons.append("orientation_unstable")
    elif orientation_status == "warning":
        soft_reasons.append("orientation_warning")
    if not bool(sim3_audit.get("sim3_eval_reliable", True)):
        hard_reasons.append("sim3_unreliable")
    if bool(sim3_audit.get("sim3_may_mask_failure", False)):
        hard_reasons.append("sim3_may_mask_failure")
    sim3_status = str(sim3_audit.get("sim3_full_trajectory_status", "") or "")
    if sim3_status == "warning":
        soft_reasons.append("sim3_warning")

    hard_reasons = list(dict.fromkeys(hard_reasons))
    soft_reasons = [reason for reason in dict.fromkeys(soft_reasons) if reason not in hard_reasons]
    status = "failed" if hard_reasons else ("warning" if soft_reasons else "ok")

    explanation_map = {
        "global_gate_failed": "Global trajectory gate failed; local valid segments may not represent the whole run.",
        "input_coverage_failed": "The estimate does not cover enough of the complete reference trajectory.",
        "input_coverage_warning": "Reference support is incomplete; report temporal and path coverage with the metrics.",
        "orientation_unstable": "Translation SR is high while orientation is unstable; translation-only SR can be misleading.",
        "orientation_warning": "Rotation error is elevated even if translation-only SR remains high.",
        "sim3_unreliable": "Sim3 reliability audit failed; Sim3-aligned SR should not be trusted.",
        "sim3_may_mask_failure": "Sim3 may be masking a real trajectory failure by absorbing error through scale/alignment.",
        "sim3_warning": "Sim3 has a warning; inspect replay before treating SR as conclusive.",
    }
    explanations = [explanation_map[reason] for reason in hard_reasons + soft_reasons if reason in explanation_map]
    if not explanations:
        explanations = ["No SR reliability issue was detected."]

    update = {
        "sr_reliability_status": status,
        "sr_reliability_hard_reasons": hard_reasons,
        "sr_reliability_soft_reasons": soft_reasons,
        "sr_warning_explanation": " ".join(explanations),
        **sim3_audit,
    }
    success.update(update)
    return update
