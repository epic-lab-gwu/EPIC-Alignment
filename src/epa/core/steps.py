from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R

from .calibration import (
    build_translation_system,
    solve_extrinsic_rotation,
    solve_extrinsic_translation,
    solve_world_alignment,
)
from .evaluation import summarize_abs_errors
from .math_utils import normalize_quat_array, rmse
from .time_alignment import (
    compute_psr,
    get_angular_velocity_norm,
    interpolate_quat_linear,
    interpolate_quat_slerp,
    matching_time_indices,
)


def _rotation_error_rmse_deg(q_ref, q_est) -> float:
    err = (R.from_quat(q_est).inv() * R.from_quat(q_ref)).magnitude()
    return float(np.sqrt(np.mean(np.degrees(err) ** 2)))


def _alignment_rmse(pos_est: np.ndarray, pos_ref: np.ndarray) -> float:
    diff = np.asarray(pos_est, dtype=float) - np.asarray(pos_ref, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _solve_world_alignment_standard(P: np.ndarray, Q: np.ndarray) -> dict[str, object]:
    Rw, tw = solve_world_alignment(P, Q)
    pred = (Rw @ np.asarray(P, dtype=float).T).T + tw
    residuals = np.linalg.norm(pred - np.asarray(Q, dtype=float), axis=1)
    return {
        "R": Rw,
        "t": tw,
        "pred": pred,
        "residuals": residuals,
        "rmse_all_m": _alignment_rmse(pred, Q),
    }


def _robust_threshold(residuals: np.ndarray, *, min_inliers: int, max_rejection_ratio: float) -> float:
    finite = np.asarray(residuals, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    mad_sigma = 1.4826 * mad
    robust_threshold = median + 3.5 * max(mad_sigma, 1e-9)
    percentile_threshold = float(np.percentile(finite, 85.0))
    threshold = min(robust_threshold, percentile_threshold)

    n = int(finite.size)
    min_inliers = max(3, min(int(min_inliers), n))
    max_rejected = int(np.floor(float(max_rejection_ratio) * float(n)))
    max_rejected = max(0, min(max_rejected, n - min_inliers))
    min_keep = n - max_rejected
    sorted_res = np.sort(finite)
    threshold = max(threshold, float(sorted_res[min_keep - 1]))
    threshold = max(threshold, float(sorted_res[min_inliers - 1]))
    return float(threshold)


def _solve_world_alignment_robust_trimmed(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    max_iterations: int = 3,
    min_inlier_ratio: float = 0.35,
    max_rejection_ratio: float = 0.65,
) -> dict[str, object]:
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    standard = _solve_world_alignment_standard(P, Q)
    n = int(P.shape[0])
    min_inliers = max(6, int(np.ceil(float(min_inlier_ratio) * float(n))))
    if n < min_inliers:
        return {
            **standard,
            "mode": "standard",
            "selected_mask": np.ones(n, dtype=bool),
            "inlier_count": n,
            "rejected_count": 0,
            "rejection_ratio": 0.0,
            "outlier_threshold_m": float("inf"),
            "robust_available": False,
            "standard_rmse_all_m": float(standard["rmse_all_m"]),
            "standard_rmse_on_inliers_m": float(standard["rmse_all_m"]),
            "robust_rmse_inliers_m": float(standard["rmse_all_m"]),
            "robust_rmse_all_m": float(standard["rmse_all_m"]),
        }

    mask = np.ones(n, dtype=bool)
    threshold = float("inf")
    for _ in range(int(max_iterations)):
        fit = _solve_world_alignment_standard(P[mask], Q[mask])
        pred_all = (fit["R"] @ P.T).T + fit["t"]
        residuals = np.linalg.norm(pred_all - Q, axis=1)
        threshold = _robust_threshold(
            residuals,
            min_inliers=min_inliers,
            max_rejection_ratio=float(max_rejection_ratio),
        )
        new_mask = residuals <= threshold
        if int(np.count_nonzero(new_mask)) < min_inliers:
            keep = np.argsort(residuals)[:min_inliers]
            new_mask = np.zeros(n, dtype=bool)
            new_mask[keep] = True
            threshold = float(np.max(residuals[keep]))
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    robust = _solve_world_alignment_standard(P[mask], Q[mask])
    pred_all = (robust["R"] @ P.T).T + robust["t"]
    residuals_all = np.linalg.norm(pred_all - Q, axis=1)
    standard_res = np.asarray(standard["residuals"], dtype=float)
    standard_inlier_rmse = _alignment_rmse(np.asarray(standard["pred"])[mask], Q[mask])
    robust_inlier_rmse = _alignment_rmse(pred_all[mask], Q[mask])
    robust_all_rmse = _alignment_rmse(pred_all, Q)
    rejected = int(n - np.count_nonzero(mask))
    improvement = standard_inlier_rmse - robust_inlier_rmse
    enough_outliers = rejected >= max(3, int(np.ceil(0.05 * n)))
    better_on_inliers = (
        robust_inlier_rmse <= standard_inlier_rmse * 0.90
        or improvement >= 0.05
    )
    use_robust = bool(enough_outliers and better_on_inliers)
    selected = robust if use_robust else standard
    selected_mask = mask if use_robust else np.ones(n, dtype=bool)

    return {
        "R": selected["R"],
        "t": selected["t"],
        "pred": pred_all if use_robust else standard["pred"],
        "residuals": residuals_all if use_robust else standard_res,
        "rmse_all_m": robust_all_rmse if use_robust else float(standard["rmse_all_m"]),
        "mode": "robust_trimmed" if use_robust else "standard",
        "selected_mask": selected_mask,
        "inlier_count": int(np.count_nonzero(selected_mask)),
        "rejected_count": int(n - np.count_nonzero(selected_mask)),
        "rejection_ratio": float((n - np.count_nonzero(selected_mask)) / max(1, n)),
        "outlier_threshold_m": float(threshold if use_robust else float("inf")),
        "robust_available": True,
        "standard_rmse_all_m": float(standard["rmse_all_m"]),
        "standard_rmse_on_inliers_m": float(standard_inlier_rmse),
        "robust_rmse_inliers_m": float(robust_inlier_rmse),
        "robust_rmse_all_m": float(robust_all_rmse),
    }


def _select_step3_stable_solve_mask(
    pos_ref: np.ndarray,
    pos_est: np.ndarray,
    *,
    min_segment_len: int = 20,
    min_segment_ratio: float = 0.05,
    max_est_step_m: float = 5.0,
    max_step_ratio: float = 30.0,
) -> tuple[np.ndarray, dict[str, float]]:
    pos_ref = np.asarray(pos_ref, dtype=float)
    pos_est = np.asarray(pos_est, dtype=float)
    n = int(min(pos_ref.shape[0], pos_est.shape[0]))
    mask_all = np.ones(n, dtype=bool)
    if n < max(3, int(min_segment_len)):
        return mask_all, {
            "step3_stable_segment_used": 0.0,
            "step3_stable_segment_count": 1.0 if n > 0 else 0.0,
            "step3_stable_solve_count": float(n),
            "step3_stable_solve_ratio": 1.0 if n > 0 else 0.0,
            "step3_stable_segment_start_index": 0.0,
            "step3_stable_segment_end_index": float(max(0, n - 1)),
        }

    ref_step = np.linalg.norm(np.diff(pos_ref[:n], axis=0), axis=1)
    est_step = np.linalg.norm(np.diff(pos_est[:n], axis=0), axis=1)
    step_limit = np.maximum(float(max_est_step_m), float(max_step_ratio) * np.maximum(ref_step, 1e-3))
    stable_transition = np.isfinite(ref_step) & np.isfinite(est_step) & (est_step <= step_limit)

    segments: list[tuple[int, int]] = []
    i = 0
    while i < stable_transition.size:
        if not stable_transition[i]:
            i += 1
            continue
        start = i
        while i + 1 < stable_transition.size and stable_transition[i + 1]:
            i += 1
        end_exclusive = i + 2
        if end_exclusive - start >= int(min_segment_len):
            segments.append((start, end_exclusive))
        i += 1

    if not segments:
        return mask_all, {
            "step3_stable_segment_used": 0.0,
            "step3_stable_segment_count": 0.0,
            "step3_stable_solve_count": float(n),
            "step3_stable_solve_ratio": 1.0,
            "step3_stable_segment_start_index": 0.0,
            "step3_stable_segment_end_index": float(n - 1),
        }

    if segments[0][0] == 0:
        start, end = segments[0]
    else:
        start, end = max(segments, key=lambda item: item[1] - item[0])
    solve_count = end - start
    if solve_count >= n:
        return mask_all, {
            "step3_stable_segment_used": 0.0,
            "step3_stable_segment_count": float(len(segments)),
            "step3_stable_solve_count": float(n),
            "step3_stable_solve_ratio": 1.0,
            "step3_stable_segment_start_index": 0.0,
            "step3_stable_segment_end_index": float(n - 1),
        }

    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True
    if solve_count / max(1, n) < float(min_segment_ratio):
        return mask_all, {
            "step3_stable_segment_used": 0.0,
            "step3_stable_segment_count": float(len(segments)),
            "step3_stable_solve_count": float(n),
            "step3_stable_solve_ratio": 1.0,
            "step3_stable_segment_start_index": 0.0,
            "step3_stable_segment_end_index": float(n - 1),
        }
    return mask, {
        "step3_stable_segment_used": 1.0,
        "step3_stable_segment_count": float(len(segments)),
        "step3_stable_solve_count": float(solve_count),
        "step3_stable_solve_ratio": float(solve_count / max(1, n)),
        "step3_stable_segment_start_index": float(start),
        "step3_stable_segment_end_index": float(end - 1),
    }


def _limit_solve_mask(mask: np.ndarray, *, max_count: int = 100000) -> tuple[np.ndarray, float]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    selected = np.flatnonzero(mask)
    if selected.size <= int(max_count):
        return mask, float(selected.size)
    keep_pos = np.linspace(0, selected.size - 1, int(max_count), dtype=int)
    limited = np.zeros_like(mask)
    limited[selected[keep_pos]] = True
    return limited, float(np.count_nonzero(limited))


def _step3_success_rate_proxy(pos_ref: np.ndarray, errors_m: np.ndarray, *, threshold_m: float = 10.0) -> float:
    pos_ref = np.asarray(pos_ref, dtype=float)
    err = np.asarray(errors_m, dtype=float).reshape(-1)
    n = int(min(pos_ref.shape[0], err.size))
    if n < 2:
        return 0.0
    valid = np.isfinite(err[:n]) & (err[:n] <= float(threshold_m))
    valid_segment = valid[:-1] & valid[1:]
    seg_dist = np.linalg.norm(np.diff(pos_ref[:n], axis=0), axis=1)
    total = float(np.sum(seg_dist))
    if total <= 0.0:
        return 0.0
    return float(np.sum(seg_dist[valid_segment]) / total)


def _step3_gate_proxy(errors_m: np.ndarray, *, percentile: float = 5.0) -> float:
    finite = np.asarray(errors_m, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, float(percentile)))


def _select_step3_solve_variant(variants: list[dict]) -> dict:
    if not variants:
        raise ValueError("No Step3 solve variants were generated.")
    if len(variants) == 1:
        return variants[0]

    full = variants[0]
    stable = variants[1]
    full_sr = float(full["step3_sr_proxy"])
    stable_sr = float(stable["step3_sr_proxy"])
    full_gate = float(full["step3_gate_proxy_m"])
    stable_gate = float(stable["step3_gate_proxy_m"])
    full_anchor = float(full["step3_stable_anchor_rmse_m"])
    stable_anchor = float(stable["step3_stable_anchor_rmse_m"])
    stable_ratio = float(stable.get("step3_stable_solve_ratio", 1.0))

    if stable_anchor > 2.0 and stable_ratio >= 0.60:
        return full

    if stable_anchor <= 2.0 and full_anchor >= max(1.0, stable_anchor * 4.0, stable_anchor + 0.5):
        return stable

    if stable_sr >= full_sr + 0.02:
        return stable
    if (
        stable_anchor <= 2.0
        and stable_sr >= full_sr - 0.005
        and stable_gate <= min(full_gate * 0.5, full_gate - 1.0)
    ):
        return stable
    return full


def _step2_rotation_candidates(R_base: np.ndarray) -> list[tuple[str, np.ndarray]]:
    flips = [
        ("base", np.eye(3, dtype=float)),
        ("right_rx180", R.from_euler("x", 180.0, degrees=True).as_matrix()),
        ("right_ry180", R.from_euler("y", 180.0, degrees=True).as_matrix()),
        ("right_rz180", R.from_euler("z", 180.0, degrees=True).as_matrix()),
    ]
    return [(name, np.asarray(R_base, dtype=float) @ flip) for name, flip in flips]


def _solve_step2_step3_candidate(
    *,
    name: str,
    R_calc,
    pr_sync,
    qr_sync,
    pos_gt_solve,
    quat_gt_solve,
    pr_solve,
    qr_solve,
):
    Rr_mats = R.from_quat(qr_sync).as_matrix()
    Rr_mats_solve = R.from_quat(qr_solve).as_matrix()
    n_solve = int(np.asarray(pos_gt_solve).shape[0])
    full_mask = np.ones(n_solve, dtype=bool)
    stable_mask, stable_info = _select_step3_stable_solve_mask(pos_gt_solve, pr_solve)
    anchor_mask = stable_mask if float(stable_info.get("step3_stable_segment_used", 0.0)) == 1.0 else full_mask
    anchor_indices_all = np.flatnonzero(anchor_mask)
    anchor_count = max(3, min(int(np.ceil(0.2 * n_solve)), int(anchor_indices_all.size)))
    anchor_indices = anchor_indices_all[:anchor_count]

    def solve_variant(variant_name: str, base_mask: np.ndarray, base_info: dict[str, float]) -> dict[str, object]:
        solve_mask = np.asarray(base_mask, dtype=bool).reshape(-1)
        if int(np.count_nonzero(solve_mask)) < 3:
            solve_mask = full_mask.copy()
        solve_mask, limited_solve_count = _limit_solve_mask(solve_mask)
        info = dict(base_info)
        info["step3_stable_solve_count"] = limited_solve_count
        if float(info.get("step3_stable_segment_used", 0.0)) == 0.0:
            info["step3_stable_solve_ratio"] = 1.0
        else:
            info["step3_stable_solve_ratio"] = float(limited_solve_count / max(1, n_solve))

        t_variant = solve_extrinsic_translation(
            pos_gt_solve[solve_mask],
            quat_gt_solve[solve_mask],
            pr_solve[solve_mask],
            qr_solve[solve_mask],
            R_calc,
        )
        extrinsic_offset = R_calc.T @ np.asarray(t_variant, dtype=float)
        pr_corr = np.asarray(pr_sync, dtype=float) - np.einsum("nij,j->ni", Rr_mats, extrinsic_offset)
        pr_corr_solve = np.asarray(pr_solve, dtype=float) - np.einsum(
            "nij,j->ni",
            Rr_mats_solve,
            extrinsic_offset,
        )

        world_fit = _solve_world_alignment_robust_trimmed(pr_corr_solve[solve_mask], pos_gt_solve[solve_mask])
        Rw_variant = np.asarray(world_fit["R"], dtype=float)
        tw_variant = np.asarray(world_fit["t"], dtype=float)
        selected_mask_local = np.asarray(world_fit["selected_mask"], dtype=bool)
        selected_mask = np.zeros(n_solve, dtype=bool)
        selected_mask[np.flatnonzero(solve_mask)[selected_mask_local]] = True
        pred_step3_pairs = (Rw_variant @ pr_corr_solve.T).T + tw_variant
        selected_rmse_m = _alignment_rmse(pred_step3_pairs[selected_mask], pos_gt_solve[selected_mask])
        pr_final_variant = (Rw_variant @ pr_corr.T).T + tw_variant
        eval_errors = np.linalg.norm(pred_step3_pairs - pos_gt_solve, axis=1)
        if anchor_indices.size >= 3:
            anchor_rmse = _alignment_rmse(pr_final_variant[anchor_indices], pos_gt_solve[anchor_indices])
        else:
            anchor_rmse = float("inf")
        return {
            "step3_solve_variant": variant_name,
            "t_calc": t_variant,
            "Rw_calc": Rw_variant,
            "tw_calc": tw_variant,
            "pr_corrected": pr_corr,
            "pr_corrected_solve": pr_corr_solve,
            "pr_final": pr_final_variant,
            "step3_rmse_selected_m": selected_rmse_m,
            "step3_sr_proxy": _step3_success_rate_proxy(pos_gt_solve, eval_errors),
            "step3_gate_proxy_m": _step3_gate_proxy(eval_errors),
            "step3_stable_anchor_rmse_m": float(anchor_rmse),
            "step3_alignment_mode": str(world_fit["mode"]),
            "step3_standard_rmse_m": float(world_fit["standard_rmse_all_m"]),
            "step3_standard_rmse_on_inliers_m": float(world_fit["standard_rmse_on_inliers_m"]),
            "step3_robust_rmse_selected_m": float(world_fit["robust_rmse_inliers_m"]),
            "step3_robust_rmse_all_m": float(world_fit["robust_rmse_all_m"]),
            "step3_inlier_count": int(world_fit["inlier_count"]),
            "step3_rejected_count": int(world_fit["rejected_count"]),
            "step3_rejection_ratio": float(world_fit["rejection_ratio"]),
            "step3_outlier_threshold_m": float(world_fit["outlier_threshold_m"]),
            "step3_robust_available": bool(world_fit["robust_available"]),
            **info,
        }

    full_info = {
        "step3_stable_segment_used": 0.0,
        "step3_stable_segment_count": stable_info["step3_stable_segment_count"],
        "step3_stable_solve_count": float(n_solve),
        "step3_stable_solve_ratio": 1.0,
        "step3_stable_segment_start_index": 0.0,
        "step3_stable_segment_end_index": float(max(0, n_solve - 1)),
    }
    variants = [solve_variant("full", full_mask, full_info)]
    if float(stable_info.get("step3_stable_segment_used", 0.0)) == 1.0:
        variants.append(solve_variant("stable", stable_mask, stable_info))
    selected_variant = _select_step3_solve_variant(variants)

    t_calc = selected_variant["t_calc"]
    Rw_calc = selected_variant["Rw_calc"]
    tw_calc = selected_variant["tw_calc"]
    pr_corrected = selected_variant["pr_corrected"]
    pr_corrected_solve = selected_variant["pr_corrected_solve"]
    pr_final = selected_variant["pr_final"]

    R_step2_mats = np.einsum("nij,jk->nik", Rr_mats, R_calc.T)
    q_step2 = normalize_quat_array(R.from_matrix(R_step2_mats).as_quat())
    q_step3_from_step2 = normalize_quat_array((R.from_matrix(Rw_calc) * R.from_quat(q_step2)).as_quat())
    q_step3_from_raw = normalize_quat_array((R.from_matrix(Rw_calc) * R.from_quat(qr_sync)).as_quat())
    rot_rmse_from_step2 = _rotation_error_rmse_deg(quat_gt_solve, q_step3_from_step2[:n_solve])
    rot_rmse_from_raw = _rotation_error_rmse_deg(quat_gt_solve, q_step3_from_raw[:n_solve])
    if rot_rmse_from_raw < rot_rmse_from_step2:
        q_step3 = q_step3_from_raw
        orientation_mode = "world_raw"
        rot_rmse_deg = rot_rmse_from_raw
    else:
        q_step3 = q_step3_from_step2
        orientation_mode = "step2"
        rot_rmse_deg = rot_rmse_from_step2
    residual = _compute_step2_residual_metrics(
        pos_gt_solve=pos_gt_solve,
        quat_gt_solve=quat_gt_solve,
        pr_solve=pr_solve,
        qr_solve=qr_solve,
        R_calc=R_calc,
        t_calc=t_calc,
    )

    return {
        "candidate_name": str(name),
        "R_calc": R_calc,
        "t_calc": t_calc,
        "Rw_calc": Rw_calc,
        "tw_calc": tw_calc,
        "pr_corrected": pr_corrected,
        "pr_corrected_solve": pr_corrected_solve,
        "pr_final": pr_final,
        "pr_final_global": np.asarray(pr_final, dtype=float).copy(),
        "q_step2": q_step2,
        "q_step3": q_step3,
        "step3_rmse_selected_m": float(selected_variant["step3_rmse_selected_m"]),
        "rotation_ape_rmse_deg": rot_rmse_deg,
        "orientation_mode": orientation_mode,
        "rot_res_median_deg": float(residual["rot_res_median_deg"]),
        "rot_res_p95_deg": float(residual["rot_res_p95_deg"]),
        "step3_solve_variant": str(selected_variant["step3_solve_variant"]),
        "step3_solve_variant_code": float(1.0 if str(selected_variant["step3_solve_variant"]) == "stable" else 0.0),
        "step3_full_rmse_selected_m": float(variants[0]["step3_rmse_selected_m"]),
        "step3_full_anchor_rmse_m": float(variants[0]["step3_stable_anchor_rmse_m"]),
        "step3_candidate_full_sr_proxy": float(variants[0]["step3_sr_proxy"]),
        "step3_candidate_full_gate_proxy_m": float(variants[0]["step3_gate_proxy_m"]),
        "step3_candidate_full_anchor_rmse_m": float(variants[0]["step3_stable_anchor_rmse_m"]),
        "step3_candidate_stable_sr_proxy": float(variants[1]["step3_sr_proxy"]) if len(variants) > 1 else float("nan"),
        "step3_candidate_stable_gate_proxy_m": float(variants[1]["step3_gate_proxy_m"]) if len(variants) > 1 else float("nan"),
        "step3_candidate_stable_anchor_rmse_m": (
            float(variants[1]["step3_stable_anchor_rmse_m"]) if len(variants) > 1 else float("nan")
        ),
        "step3_alignment_mode": str(selected_variant["step3_alignment_mode"]),
        "step3_standard_rmse_m": float(selected_variant["step3_standard_rmse_m"]),
        "step3_standard_rmse_on_inliers_m": float(selected_variant["step3_standard_rmse_on_inliers_m"]),
        "step3_robust_rmse_selected_m": float(selected_variant["step3_robust_rmse_selected_m"]),
        "step3_robust_rmse_all_m": float(selected_variant["step3_robust_rmse_all_m"]),
        "step3_inlier_count": int(selected_variant["step3_inlier_count"]),
        "step3_rejected_count": int(selected_variant["step3_rejected_count"]),
        "step3_rejection_ratio": float(selected_variant["step3_rejection_ratio"]),
        "step3_outlier_threshold_m": float(selected_variant["step3_outlier_threshold_m"]),
        "step3_robust_available": bool(selected_variant["step3_robust_available"]),
        "step3_stable_segment_used": float(selected_variant["step3_stable_segment_used"]),
        "step3_stable_segment_count": float(selected_variant["step3_stable_segment_count"]),
        "step3_stable_solve_count": float(selected_variant["step3_stable_solve_count"]),
        "step3_stable_solve_ratio": float(selected_variant["step3_stable_solve_ratio"]),
        "step3_stable_segment_start_index": float(selected_variant["step3_stable_segment_start_index"]),
        "step3_stable_segment_end_index": float(selected_variant["step3_stable_segment_end_index"]),
    }


def _select_step2_step3_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("No Step2/Step3 candidates were generated.")
    base = candidates[0]
    best_trans = min(float(c.get("step3_full_rmse_selected_m", c["step3_rmse_selected_m"])) for c in candidates)
    best_rot = min(float(c["rotation_ape_rmse_deg"]) for c in candidates)
    base_rel = float(base["rot_res_median_deg"])
    best_sr = max(float(c.get("step3_candidate_full_sr_proxy", c.get("step3_sr_proxy", 0.0))) for c in candidates)
    trans_limit = best_trans * 1.05 + 0.02
    rel_limit = base_rel + 2.0
    rot_limit = best_rot + 20.0
    eligible = [
        c
        for c in candidates
        if float(c.get("step3_full_rmse_selected_m", c["step3_rmse_selected_m"])) <= trans_limit
        and float(c["rot_res_median_deg"]) <= rel_limit
        and float(c["rotation_ape_rmse_deg"]) <= rot_limit
        and float(c.get("step3_candidate_full_sr_proxy", c.get("step3_sr_proxy", 0.0))) >= best_sr - 0.02
    ]
    if not eligible:
        eligible = [base]
    return min(
        eligible,
        key=lambda c: (
            -float(c.get("step3_candidate_full_sr_proxy", c.get("step3_sr_proxy", 0.0))),
            float(c["rotation_ape_rmse_deg"]),
            float(c.get("step3_full_rmse_selected_m", c["step3_rmse_selected_m"])),
            float(c["rot_res_median_deg"]),
        ),
    )


def _downsample_by_max_hz(tvals, pos, quat, max_hz):
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    max_hz = float(max_hz)
    if max_hz <= 0.0 or tvals.size <= 2:
        return tvals, pos, quat, {
            "enabled": False,
            "max_hz": max_hz,
            "input_samples": int(tvals.size),
            "output_samples": int(tvals.size),
        }

    min_dt = 1.0 / max_hz
    keep = [0]
    last_t = float(tvals[0])
    for idx in range(1, tvals.size - 1):
        if float(tvals[idx]) - last_t >= min_dt * (1.0 - 1e-9):
            keep.append(idx)
            last_t = float(tvals[idx])
    if keep[-1] != tvals.size - 1:
        keep.append(tvals.size - 1)

    ids = np.asarray(keep, dtype=int)
    return tvals[ids], pos[ids], quat[ids], {
        "enabled": bool(ids.size < tvals.size),
        "max_hz": max_hz,
        "input_samples": int(tvals.size),
        "output_samples": int(ids.size),
    }


def _select_gt_overlap_window(tvals, pos, quat, t_est_sync, min_samples=10):
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    t_est_sync = np.asarray(t_est_sync, dtype=float).reshape(-1)

    finite_est = t_est_sync[np.isfinite(t_est_sync)]
    input_samples = int(tvals.size)
    info = {
        "mode": "full_gt_extrapolate_fallback",
        "fallback_used": True,
        "fallback_reason": "",
        "input_samples": input_samples,
        "selected_samples": input_samples,
        "overlap_samples": 0,
        "overlap_ratio": 0.0,
        "extrapolated_sample_ratio": 1.0,
        "est_sync_start_s": float("nan"),
        "est_sync_end_s": float("nan"),
        "min_samples": int(min_samples),
    }
    if input_samples == 0 or finite_est.size == 0:
        info["fallback_reason"] = "empty timestamp input"
        return tvals, pos, quat, info

    est_start = float(np.min(finite_est))
    est_end = float(np.max(finite_est))
    info["est_sync_start_s"] = est_start
    info["est_sync_end_s"] = est_end
    eps = max(1e-9, float(np.nanmedian(np.abs(np.diff(tvals)))) * 1e-6) if input_samples > 1 else 1e-9
    mask = (tvals >= est_start - eps) & (tvals <= est_end + eps)
    overlap_samples = int(np.sum(mask))
    overlap_ratio = float(overlap_samples / max(1, input_samples))
    info["overlap_samples"] = overlap_samples
    info["overlap_ratio"] = overlap_ratio
    info["extrapolated_sample_ratio"] = float(1.0 - overlap_ratio)

    if overlap_samples >= int(min_samples):
        info.update(
            {
                "mode": "overlap",
                "fallback_used": False,
                "fallback_reason": "",
                "selected_samples": overlap_samples,
                "extrapolated_sample_ratio": 0.0,
            }
        )
        return tvals[mask], pos[mask], quat[mask], info

    info["fallback_reason"] = (
        f"overlap_samples={overlap_samples} < min_samples={int(min_samples)}"
    )
    return tvals, pos, quat, info


def _offset_grid(min_offset_s: float, max_offset_s: float, step_s: float):
    if step_s <= 0.0:
        raise ValueError("offset step must be > 0.")
    values = np.arange(min_offset_s, max_offset_s + 0.5 * step_s, step_s)
    return [float(v) for v in values]


def _match_nearest_timestamps(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
):
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1) + float(offset_est_s)
    idx_ref = []
    idx_est = []
    for i, t_ref in enumerate(s_ref):
        diffs = np.abs(s_est - t_ref)
        j = int(np.argmin(diffs))
        if float(diffs[j]) <= float(max_diff_s):
            idx_ref.append(int(i))
            idx_est.append(int(j))
    return np.asarray(idx_ref, dtype=int), np.asarray(idx_est, dtype=int)


def _associate_gt_est(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
):
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1)

    if s_ref.size == 0 or s_est.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    est_longer = s_est.size > s_ref.size
    if est_longer:
        ids_short, ids_long = _match_nearest_timestamps(
            s_ref,
            s_est,
            max_diff_s=float(max_diff_s),
            offset_est_s=float(offset_est_s),
        )
        ids_ref = ids_short
        ids_est = ids_long
    else:
        ids_short, ids_long = _match_nearest_timestamps(
            s_est,
            s_ref,
            max_diff_s=float(max_diff_s),
            offset_est_s=-float(offset_est_s),
        )
        ids_ref = ids_long
        ids_est = ids_short

    return np.asarray(ids_ref, dtype=int), np.asarray(ids_est, dtype=int)


def _offset_match_diagnostics(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
    overlap_gate_min_pairs: int | None = None,
):
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1)
    pair_cap_global = max(1, min(s_ref.size, s_est.size))
    if overlap_gate_min_pairs is None:
        overlap_gate_min_pairs = max(80, int(0.1 * float(pair_cap_global)))
    overlap_gate_min_pairs = max(1, int(overlap_gate_min_pairs))

    ids_ref, ids_est = matching_time_indices(
        s_ref,
        s_est,
        max_diff=float(max_diff_s),
        offset_2=float(offset_est_s),
    )
    match_count = int(len(ids_ref))
    ratio_global = min(1.0, float(match_count) / float(pair_cap_global))

    s_est_shift = s_est + float(offset_est_s)
    overlap_start = max(float(s_ref[0]), float(s_est_shift[0]))
    overlap_end = min(float(s_ref[-1]), float(s_est_shift[-1]))
    overlap_cap = 0
    ratio_overlap = ratio_global
    if overlap_end >= overlap_start:
        n_ref_overlap = int(np.sum((s_ref >= overlap_start) & (s_ref <= overlap_end)))
        n_est_overlap = int(np.sum((s_est_shift >= overlap_start) & (s_est_shift <= overlap_end)))
        overlap_cap = int(min(n_ref_overlap, n_est_overlap))
        if overlap_cap > 0:
            ratio_overlap = min(1.0, float(match_count) / float(overlap_cap))

    ratio_gate = ratio_global
    if overlap_cap >= overlap_gate_min_pairs:
        ratio_gate = max(ratio_global, ratio_overlap)

    return {
        "match_count": match_count,
        "ratio_global": float(ratio_global),
        "ratio_overlap": float(ratio_overlap),
        "ratio_gate": float(ratio_gate),
        "pair_cap_global": int(pair_cap_global),
        "pair_cap_overlap": int(overlap_cap),
        "overlap_gate_min_pairs": int(overlap_gate_min_pairs),
        "ids_ref": ids_ref,
        "ids_est": ids_est,
    }


def _map_est_to_ref_nearest(
    *,
    t_ref,
    t_est,
    pos_est,
    quat_est,
    offset_est_s: float,
):
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    t_est_shift = np.asarray(t_est, dtype=float).reshape(-1) - float(offset_est_s)
    pos_est = np.asarray(pos_est, dtype=float)
    quat_est = np.asarray(quat_est, dtype=float)
    if t_est_shift.size == 0:
        raise ValueError("Empty estimation timestamps for nearest mapping.")

    idx = np.empty(t_ref.size, dtype=int)
    j = np.searchsorted(t_est_shift, t_ref, side="left")
    for i, k in enumerate(j):
        if k <= 0:
            idx[i] = 0
            continue
        if k >= t_est_shift.size:
            idx[i] = t_est_shift.size - 1
            continue
        d0 = abs(t_ref[i] - t_est_shift[k - 1])
        d1 = abs(t_ref[i] - t_est_shift[k])
        idx[i] = (k - 1) if d0 <= d1 else k

    return pos_est[idx], quat_est[idx]


def _prefer_offset_candidate(curr, cand):
    if curr is None:
        return True
    curr_valid, curr_rmse, curr_matches, curr_abs = curr
    cand_valid, cand_rmse, cand_matches, cand_abs = cand
    if cand_valid and not curr_valid:
        return True
    if curr_valid and not cand_valid:
        return False
    if cand_rmse < curr_rmse - 1e-12:
        return True
    if np.isclose(cand_rmse, curr_rmse, atol=1e-12, rtol=0.0):
        if cand_matches > curr_matches:
            return True
        if cand_matches == curr_matches and cand_abs < curr_abs - 1e-12:
            return True
    return False


def _search_direct_offset_from_matched_pairs(
    *,
    t_ref,
    pos_ref,
    t_est,
    pos_est,
    initial_offset_s: float,
    max_diff_s: float,
    min_match_ratio: float,
    coarse_min_s: float = -20.0,
    coarse_max_s: float = 20.0,
    coarse_step_s: float = 0.2,
    refine_window_s: float = 0.5,
    refine_step_s: float = 0.01,
):
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    t_est = np.asarray(t_est, dtype=float).reshape(-1)
    pref = np.asarray(pos_ref, dtype=float)
    pest = np.asarray(pos_est, dtype=float)
    if t_ref.size != pref.shape[0] or t_est.size != pest.shape[0]:
        raise ValueError("Timestamp and position array lengths are inconsistent.")

    pair_cap = max(1, min(t_ref.size, t_est.size))
    min_required = max(3, int(float(min_match_ratio) * float(pair_cap)))

    best_key = None
    best = None

    def _eval_offset(offset_s: float):
        ids_ref, ids_est = _associate_gt_est(
            t_ref,
            t_est,
            max_diff_s=float(max_diff_s),
            offset_est_s=-float(offset_s),
        )
        n = len(ids_ref)
        if n < 3:
            return None
        pref_sel = pref[np.asarray(ids_ref, dtype=int)]
        pest_sel = pest[np.asarray(ids_est, dtype=int)]
        Rw, tw = solve_world_alignment(pest_sel, pref_sel)
        pred = (Rw @ pest_sel.T).T + tw
        rmse = float(np.sqrt(np.mean(np.sum((pred - pref_sel) ** 2, axis=1))))
        valid = bool(n >= min_required)
        key = (valid, rmse, int(n), abs(float(offset_s)))
        return {
            "offset_s": float(offset_s),
            "matches": int(n),
            "valid": valid,
            "rmse_pairs_m": rmse,
            "ids_ref": np.asarray(ids_ref, dtype=int),
            "ids_est": np.asarray(ids_est, dtype=int),
            "_key": key,
        }

    for off in _offset_grid(coarse_min_s, coarse_max_s, coarse_step_s):
        candidate = _eval_offset(off)
        if candidate is None:
            continue
        if _prefer_offset_candidate(best_key, candidate["_key"]):
            best_key = candidate["_key"]
            best = candidate

    init_candidate = _eval_offset(float(initial_offset_s))
    if init_candidate is not None and _prefer_offset_candidate(best_key, init_candidate["_key"]):
        best_key = init_candidate["_key"]
        best = init_candidate

    if best is None:
        return None

    refine_start = max(float(coarse_min_s), float(best["offset_s"]) - float(refine_window_s))
    refine_end = min(float(coarse_max_s), float(best["offset_s"]) + float(refine_window_s))
    for off in _offset_grid(refine_start, refine_end, refine_step_s):
        candidate = _eval_offset(off)
        if candidate is None:
            continue
        if _prefer_offset_candidate(best_key, candidate["_key"]):
            best_key = candidate["_key"]
            best = candidate

    best.pop("_key", None)
    best["required_matches"] = int(min_required)
    return best


def _run_time_alignment(
    *,
    t_gt,
    quat_gt,
    t_est,
    quat_est,
    dt_resample: float,
    offset_search_window_s: float,
    offset_min_match_ratio: float,
    evo_match_max_diff_s: float,
    artificial_offset_s: float | None = None,
):
    t_gt_mid, om_gt = get_angular_velocity_norm(t_gt, quat_gt)
    t_est_mid, om_est = get_angular_velocity_norm(t_est, quat_est)

    dt_resample = float(dt_resample)
    if dt_resample <= 0.0:
        raise ValueError("--dt-resample must be > 0.")
    offset_search_window_s = float(offset_search_window_s)
    offset_min_match_ratio = float(offset_min_match_ratio)
    if not (0.0 <= offset_min_match_ratio <= 1.0):
        raise ValueError("--offset-min-match-ratio must be in [0, 1].")

    t_min = max(float(np.min(t_gt_mid)), float(np.min(t_est_mid)))
    t_max = min(float(np.max(t_gt_mid)), float(np.max(t_est_mid)))
    if t_max - t_min < 100 * dt_resample:
        raise ValueError("Not enough overlap between GT and estimation for robust time alignment.")

    eps = max(1e-9, abs(dt_resample) * 1e-6)
    safe_min = float(np.nextafter(t_min + eps, np.inf))
    safe_max = float(np.nextafter(t_max - eps, -np.inf))
    if safe_max - safe_min < 100 * dt_resample:
        raise ValueError("Not enough valid overlap after applying interpolation safety margins.")
    sample_count = int(np.floor((safe_max - safe_min) / dt_resample)) + 1
    t_uniform = safe_min + dt_resample * np.arange(sample_count, dtype=float)
    t_uniform = np.clip(t_uniform, safe_min, safe_max)
    if t_uniform.size < 100:
        raise ValueError("Not enough valid resampled points after applying interpolation safety margins.")

    sig_gt = interp1d(t_gt_mid, om_gt, kind="linear")(t_uniform)
    sig_est = interp1d(t_est_mid, om_est, kind="linear")(t_uniform)
    sig_gt -= np.mean(sig_gt)
    sig_est -= np.mean(sig_est)
    if (not np.all(np.isfinite(sig_gt))) or (not np.all(np.isfinite(sig_est))):
        raise ValueError(
            "Step-1 signals contain non-finite values. Check timestamps for duplicates/non-monotonic samples."
        )

    corr = correlate(sig_est, sig_gt, mode="full")
    if not np.all(np.isfinite(corr)):
        raise ValueError(
            "Cross-correlation contains non-finite values (possible invalid timestamp deltas or signal values)."
        )
    lags = np.arange(-len(sig_gt) + 1, len(sig_est))
    offsets_s = lags.astype(float) * dt_resample

    search_mask = np.ones_like(offsets_s, dtype=bool)
    if offset_search_window_s > 0.0:
        search_mask = np.abs(offsets_s) <= offset_search_window_s
        if not np.any(search_mask):
            raise ValueError("Offset search window has no valid lag candidates. Increase --offset-search-window-s.")

    search_indices = np.flatnonzero(search_mask)
    peak_idx = int(search_indices[int(np.argmax(corr[search_indices]))])
    calculated_offset = float(offsets_s[peak_idx])

    def _count_matches_for_offset(offset_s: float) -> dict[str, object]:
        return _offset_match_diagnostics(
            t_gt,
            t_est,
            max_diff_s=evo_match_max_diff_s,
            offset_est_s=-float(offset_s),
        )

    def _best_near_zero_offset(window_s: float) -> tuple[float, int]:
        zero_mask = np.abs(offsets_s) <= float(window_s)
        if offset_search_window_s > 0.0:
            zero_mask &= search_mask
        if np.any(zero_mask):
            zero_indices = np.flatnonzero(zero_mask)
            zero_peak_idx = int(zero_indices[int(np.argmax(corr[zero_indices]))])
        else:
            zero_peak_idx = int(np.argmin(np.abs(offsets_s)))
        return float(offsets_s[zero_peak_idx]), zero_peak_idx

    def _omega_rmse_after_offset(offset_s: float) -> float:
        sig_est_shifted_for_offset = interp1d(
            t_uniform - float(offset_s),
            sig_est,
            kind="linear",
            fill_value="extrapolate",
        )(t_uniform)
        return rmse(sig_est_shifted_for_offset - sig_gt)

    match_diag = _count_matches_for_offset(calculated_offset)
    match_ratio_global = float(match_diag["ratio_global"])
    match_ratio_overlap = float(match_diag["ratio_overlap"])
    match_ratio_gate = float(match_diag["ratio_gate"])
    overlap_pair_cap = int(match_diag["pair_cap_overlap"])
    overlap_gate_min_pairs = int(match_diag["overlap_gate_min_pairs"])
    fallback_used = 0.0
    fallback_offset = float("nan")
    fallback_ratio_global = float("nan")
    fallback_ratio_overlap = float("nan")
    fallback_ratio_gate = float("nan")
    step1_forced_candidate = False
    step1_force_reason = ""
    near_zero_window_s = 1.0
    near_zero_offset, near_zero_peak_idx = _best_near_zero_offset(near_zero_window_s)
    near_zero_diag = _count_matches_for_offset(near_zero_offset)
    near_zero_ratio_global = float(near_zero_diag["ratio_global"])
    near_zero_ratio_overlap = float(near_zero_diag["ratio_overlap"])
    near_zero_ratio_gate = float(near_zero_diag["ratio_gate"])
    omega_rmse_after_peak = _omega_rmse_after_offset(calculated_offset)
    omega_rmse_after_near_zero = _omega_rmse_after_offset(near_zero_offset)
    zero_offset = 0.0
    zero_peak_idx = int(np.argmin(np.abs(offsets_s)))
    zero_diag = _count_matches_for_offset(zero_offset)
    zero_ratio_gate = float(zero_diag["ratio_gate"])
    omega_rmse_after_zero = _omega_rmse_after_offset(zero_offset)
    small_offset_improve_ratio = (
        (omega_rmse_after_zero - omega_rmse_after_near_zero)
        / (omega_rmse_after_zero + 1e-12)
    )
    if (
        abs(near_zero_offset) <= near_zero_window_s
        and abs(near_zero_offset) > 0.0
        and zero_ratio_gate >= offset_min_match_ratio
        and small_offset_improve_ratio < 0.01
    ):
        near_zero_offset = zero_offset
        near_zero_peak_idx = zero_peak_idx
        near_zero_diag = zero_diag
        near_zero_ratio_global = float(near_zero_diag["ratio_global"])
        near_zero_ratio_overlap = float(near_zero_diag["ratio_overlap"])
        near_zero_ratio_gate = float(near_zero_diag["ratio_gate"])
        omega_rmse_after_near_zero = omega_rmse_after_zero

    near_zero_preferred = (
        offset_search_window_s <= 0.0
        and abs(calculated_offset) > near_zero_window_s
        and near_zero_ratio_gate >= offset_min_match_ratio
        and omega_rmse_after_peak >= omega_rmse_after_near_zero * 0.95
    )
    # A large correlation peak is ignored when timestamp matching and angular
    # velocity residuals show that the near-zero candidate is equally plausible.
    if near_zero_preferred:
        calculated_offset = near_zero_offset
        peak_idx = near_zero_peak_idx
        match_ratio_global = near_zero_ratio_global
        match_ratio_overlap = near_zero_ratio_overlap
        match_ratio_gate = near_zero_ratio_gate
        overlap_pair_cap = int(near_zero_diag["pair_cap_overlap"])
        overlap_gate_min_pairs = int(near_zero_diag["overlap_gate_min_pairs"])

    if match_ratio_gate < offset_min_match_ratio:
        fallback_offset = near_zero_offset
        zero_peak_idx = near_zero_peak_idx

        fallback_diag = _count_matches_for_offset(fallback_offset)
        fallback_ratio_global = float(fallback_diag["ratio_global"])
        fallback_ratio_overlap = float(fallback_diag["ratio_overlap"])
        fallback_ratio_gate = float(fallback_diag["ratio_gate"])
        if fallback_ratio_gate >= offset_min_match_ratio:
            calculated_offset = fallback_offset
            peak_idx = zero_peak_idx
            match_ratio_global = fallback_ratio_global
            match_ratio_overlap = fallback_ratio_overlap
            match_ratio_gate = fallback_ratio_gate
            overlap_pair_cap = int(fallback_diag["pair_cap_overlap"])
            overlap_gate_min_pairs = int(fallback_diag["overlap_gate_min_pairs"])
            fallback_used = 1.0
        else:
            step1_force_reason = (
                "Unreliable step-1 offset estimate: "
                f"best_match_ratio_global={match_ratio_global:.3f}, "
                f"best_match_ratio_overlap={match_ratio_overlap:.3f}, "
                f"best_match_ratio_gate={match_ratio_gate:.3f}, "
                f"fallback_match_ratio_global={fallback_ratio_global:.3f}, "
                f"fallback_match_ratio_overlap={fallback_ratio_overlap:.3f}, "
                f"fallback_match_ratio_gate={fallback_ratio_gate:.3f}, "
                f"required>={offset_min_match_ratio:.3f}. "
                "Check timestamp quality (duplicates/non-monotonic) or adjust offset search settings."
            )
            step1_forced_candidate = True
            print("--- STEP 1 FALLBACK ---")
            print(step1_force_reason)
            print(
                f"Continuing with highest-confidence candidate offset: {calculated_offset:.4f} s"
            )

    sig_est_shifted = interp1d(
        t_uniform - calculated_offset,
        sig_est,
        kind="linear",
        fill_value="extrapolate",
    )(t_uniform)

    corr_norm_peak = corr[peak_idx] / (np.linalg.norm(sig_gt) * np.linalg.norm(sig_est) + 1e-12)
    psr = compute_psr(corr, peak_idx, guard_bins=max(1, int(0.02 / dt_resample)))
    time_metrics = {
        "offset_est_s": calculated_offset,
        "offset_err_ms": np.nan,
        "xcorr_peak_normalized": corr_norm_peak,
        "xcorr_psr": psr,
        "omega_rmse_before": rmse(sig_est - sig_gt),
        "omega_rmse_after": rmse(sig_est_shifted - sig_gt),
        "offset_search_window_s": offset_search_window_s,
        "offset_min_match_ratio": offset_min_match_ratio,
        "offset_fallback_used": fallback_used,
        "offset_match_ratio_global": match_ratio_global,
        "offset_match_ratio_overlap": match_ratio_overlap,
        "offset_match_ratio_gate": match_ratio_gate,
        "offset_overlap_pair_cap": float(overlap_pair_cap),
        "offset_overlap_gate_min_pairs": float(overlap_gate_min_pairs),
        "fallback_offset_s": float(fallback_offset),
        "fallback_match_ratio_global": float(fallback_ratio_global),
        "fallback_match_ratio_overlap": float(fallback_ratio_overlap),
        "fallback_match_ratio_gate": float(fallback_ratio_gate),
        "step1_forced_candidate_code": 1.0 if step1_forced_candidate else 0.0,
        "near_zero_offset_s": float(near_zero_offset),
        "near_zero_match_ratio_global": float(near_zero_ratio_global),
        "near_zero_match_ratio_overlap": float(near_zero_ratio_overlap),
        "near_zero_match_ratio_gate": float(near_zero_ratio_gate),
        "near_zero_omega_rmse_after": float(omega_rmse_after_near_zero),
        "zero_omega_rmse_after": float(omega_rmse_after_zero),
        "near_zero_improve_ratio": float(small_offset_improve_ratio),
        "near_zero_preferred_code": 1.0 if near_zero_preferred else 0.0,
    }

    evo_t_offset_used_s = -calculated_offset
    match_ids_ref, _ = matching_time_indices(
        t_gt, t_est, max_diff=evo_match_max_diff_s, offset_2=evo_t_offset_used_s
    )
    time_metrics["evo_t_offset_used_s"] = evo_t_offset_used_s
    time_metrics["evo_match_max_diff_s"] = evo_match_max_diff_s
    time_metrics["evo_matches_equivalent"] = float(len(match_ids_ref))
    time_metrics["evo_matches_ratio_equivalent"] = float(match_ratio_global)
    time_metrics["evo_matches_ratio_overlap_aware"] = float(match_ratio_overlap)
    time_metrics["evo_matches_ratio_for_gate"] = float(match_ratio_gate)
    if artificial_offset_s is not None:
        time_metrics["offset_err_ms"] = abs(calculated_offset - float(artificial_offset_s)) * 1e3

    time_metrics["omega_rmse_improve_pct"] = (
        (time_metrics["omega_rmse_before"] - time_metrics["omega_rmse_after"])
        / (time_metrics["omega_rmse_before"] + 1e-12)
        * 100.0
    )

    return {
        "calculated_offset": calculated_offset,
        "time_metrics": time_metrics,
        "step1_forced_candidate": step1_forced_candidate,
        "step1_force_reason": step1_force_reason,
        "t_uniform": t_uniform,
        "sig_gt": sig_gt,
        "sig_est": sig_est,
        "corr": corr,
        "lags": lags,
    }


def _prepare_solve_eval_trajectories(
    *,
    t_gt,
    pos_gt,
    quat_gt,
    t_est,
    pos_est,
    quat_est,
    calculated_offset: float,
    downsample_hz: float,
    quat_interp: str,
):
    t_est_sync = np.asarray(t_est, dtype=float) - float(calculated_offset)
    t_gt_full = np.asarray(t_gt, dtype=float)
    pos_gt_full = np.asarray(pos_gt, dtype=float)
    quat_gt_full = np.asarray(quat_gt, dtype=float)
    t_gt_eval, pos_gt_eval, quat_gt_eval, overlap_info = _select_gt_overlap_window(
        t_gt_full,
        pos_gt_full,
        quat_gt_full,
        t_est_sync,
        min_samples=10,
    )

    downsample_hz = float(downsample_hz)
    if downsample_hz < 0.0:
        raise ValueError("--downsample-hz must be >= 0.")
    t_gt_out, pos_gt_out, quat_gt_out, downsample_info = _downsample_by_max_hz(
        t_gt_eval,
        pos_gt_eval,
        quat_gt_eval,
        downsample_hz,
    )

    interp_p = interp1d(t_est_sync, pos_est, axis=0, fill_value="extrapolate")
    pr_sync = interp_p(t_gt_out)
    if str(quat_interp) == "slerp":
        qr_sync = interpolate_quat_slerp(t_est_sync, quat_est, t_gt_out)
    else:
        qr_sync = interpolate_quat_linear(t_est_sync, quat_est, t_gt_out)

    return {
        "t_est_sync": t_est_sync,
        "t_gt": t_gt_out,
        "pos_gt": pos_gt_out,
        "quat_gt": quat_gt_out,
        "pr_sync": pr_sync,
        "qr_sync": qr_sync,
        "pos_gt_solve": np.asarray(pos_gt_out, dtype=float),
        "quat_gt_solve": np.asarray(quat_gt_out, dtype=float),
        "pr_solve": np.asarray(pr_sync, dtype=float),
        "qr_solve": np.asarray(qr_sync, dtype=float),
        "overlap_info": overlap_info,
        "downsample_info": downsample_info,
    }


def _solve_step2_step3(
    *,
    pr_sync,
    qr_sync,
    pos_gt_solve,
    quat_gt_solve,
    pr_solve,
    qr_solve,
):
    R_base = solve_extrinsic_rotation(quat_gt_solve, qr_solve)
    candidates = []
    for candidate_name, R_candidate in _step2_rotation_candidates(R_base):
        candidates.append(
            _solve_step2_step3_candidate(
                name=candidate_name,
                R_calc=R_candidate,
                pr_sync=pr_sync,
                qr_sync=qr_sync,
                pos_gt_solve=pos_gt_solve,
                quat_gt_solve=quat_gt_solve,
                pr_solve=pr_solve,
                qr_solve=qr_solve,
            )
        )
    selected = _select_step2_step3_candidate(candidates)
    ambiguity_detected = (
        str(selected["candidate_name"]) != "base"
        or float(candidates[0]["rotation_ape_rmse_deg"]) > float(selected["rotation_ape_rmse_deg"]) + 5.0
    )

    return {
        "R_calc": selected["R_calc"],
        "t_calc": selected["t_calc"],
        "Rw_calc": selected["Rw_calc"],
        "tw_calc": selected["tw_calc"],
        "pr_corrected": selected["pr_corrected"],
        "pr_corrected_solve": selected["pr_corrected_solve"],
        "pr_final": selected["pr_final"],
        "pr_final_global": selected["pr_final_global"],
        "q_step2": selected["q_step2"],
        "q_step3": selected["q_step3"],
        "step3_choice": {
            "step3_rmse_selected_m": float(selected["step3_rmse_selected_m"]),
            "orientation_candidate_count": float(len(candidates)),
            "orientation_candidate_selected_code": float(
                [c["candidate_name"] for c in candidates].index(selected["candidate_name"])
            ),
            "orientation_mode_code": float(1.0 if str(selected["orientation_mode"]) == "world_raw" else 0.0),
            "orientation_candidate_rot_rmse_deg": float(selected["rotation_ape_rmse_deg"]),
            "orientation_candidate_trans_rmse_m": float(selected["step3_rmse_selected_m"]),
            "orientation_candidate_rel_rot_median_deg": float(selected["rot_res_median_deg"]),
            "orientation_ambiguity_detected": float(bool(ambiguity_detected)),
            "step3_solve_variant": str(selected["step3_solve_variant"]),
            "step3_solve_variant_code": float(selected["step3_solve_variant_code"]),
            "step3_candidate_full_sr_proxy": float(selected["step3_candidate_full_sr_proxy"]),
            "step3_candidate_full_gate_proxy_m": float(selected["step3_candidate_full_gate_proxy_m"]),
            "step3_candidate_full_anchor_rmse_m": float(selected["step3_candidate_full_anchor_rmse_m"]),
            "step3_candidate_stable_sr_proxy": float(selected["step3_candidate_stable_sr_proxy"]),
            "step3_candidate_stable_gate_proxy_m": float(selected["step3_candidate_stable_gate_proxy_m"]),
            "step3_candidate_stable_anchor_rmse_m": float(selected["step3_candidate_stable_anchor_rmse_m"]),
            "step3_alignment_mode": str(selected["step3_alignment_mode"]),
            "step3_alignment_mode_code": float(
                1.0 if str(selected["step3_alignment_mode"]) == "robust_trimmed" else 0.0
            ),
            "step3_standard_rmse_m": float(selected["step3_standard_rmse_m"]),
            "step3_standard_rmse_on_inliers_m": float(selected["step3_standard_rmse_on_inliers_m"]),
            "step3_robust_rmse_selected_m": float(selected["step3_robust_rmse_selected_m"]),
            "step3_robust_rmse_all_m": float(selected["step3_robust_rmse_all_m"]),
            "step3_inlier_count": float(selected["step3_inlier_count"]),
            "step3_rejected_count": float(selected["step3_rejected_count"]),
            "step3_rejection_ratio": float(selected["step3_rejection_ratio"]),
            "step3_outlier_threshold_m": float(selected["step3_outlier_threshold_m"]),
            "step3_robust_available": float(bool(selected["step3_robust_available"])),
            "step3_stable_segment_used": float(selected["step3_stable_segment_used"]),
            "step3_stable_segment_count": float(selected["step3_stable_segment_count"]),
            "step3_stable_solve_count": float(selected["step3_stable_solve_count"]),
            "step3_stable_solve_ratio": float(selected["step3_stable_solve_ratio"]),
            "step3_stable_segment_start_index": float(selected["step3_stable_segment_start_index"]),
            "step3_stable_segment_end_index": float(selected["step3_stable_segment_end_index"]),
        },
    }


def _compute_step2_residual_metrics(
    *,
    pos_gt_solve,
    quat_gt_solve,
    pr_solve,
    qr_solve,
    R_calc,
    t_calc,
):
    Rv_rot = R.from_quat(quat_gt_solve)
    Rr_rot = R.from_quat(qr_solve)
    dA = Rv_rot[:-1].inv() * Rv_rot[1:]
    dB = Rr_rot[:-1].inv() * Rr_rot[1:]
    Rext_rot = R.from_matrix(R_calc)
    dA_pred = Rext_rot * dB * Rext_rot.inv()
    rot_res_deg = np.degrees((dA.inv() * dA_pred).magnitude())

    C_mat, d_vec, num_constraints = build_translation_system(
        pos_gt_solve, quat_gt_solve, pr_solve, qr_solve, R_calc
    )
    trans_res = C_mat @ t_calc - d_vec
    trans_res_norm = np.linalg.norm(trans_res.reshape(-1, 3), axis=1)

    return {
        "rot_res_mean_deg": np.mean(rot_res_deg),
        "rot_res_median_deg": np.median(rot_res_deg),
        "rot_res_p95_deg": np.percentile(rot_res_deg, 95),
        "trans_eq_rmse_m": rmse(trans_res_norm),
        "trans_eq_p95_m": np.percentile(trans_res_norm, 95),
        "translation_system_cond": np.linalg.cond(C_mat),
        "translation_constraints": float(num_constraints),
    }


def _compute_trajectory_metrics(*, pos_gt, pr_sync, pr_corrected, pr_final):
    raw_err = np.linalg.norm(pr_sync - pos_gt, axis=1)
    step2_err = np.linalg.norm(pr_corrected - pos_gt, axis=1)
    step3_err = np.linalg.norm(pr_final - pos_gt, axis=1)
    raw_stats = summarize_abs_errors(raw_err)
    step2_stats = summarize_abs_errors(step2_err)
    step3_stats = summarize_abs_errors(step3_err)

    traj_metrics = {
        "ate_rmse_raw_m": raw_stats["rmse"],
        "ate_rmse_step2_m": step2_stats["rmse"],
        "ate_rmse_step3_m": step3_stats["rmse"],
        "ate_p95_raw_m": raw_stats["p95"],
        "ate_p95_step2_m": step2_stats["p95"],
        "ate_p95_step3_m": step3_stats["p95"],
        "ate_rmse_improve_raw_to_step3_pct": (
            (raw_stats["rmse"] - step3_stats["rmse"]) / (raw_stats["rmse"] + 1e-12) * 100.0
        ),
        "ate_rmse_improve_step2_to_step3_pct": (
            (step2_stats["rmse"] - step3_stats["rmse"]) / (step2_stats["rmse"] + 1e-12) * 100.0
        ),
    }

    return {
        "raw_err": raw_err,
        "step2_err": step2_err,
        "step3_err": step3_err,
        "raw_stats": raw_stats,
        "step2_stats": step2_stats,
        "step3_stats": step3_stats,
        "traj_metrics": traj_metrics,
    }
