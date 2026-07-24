from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .calibration import (
    build_translation_system,
    solve_extrinsic_rotation,
    solve_extrinsic_translation,
    solve_world_alignment,
)
from .evaluation import summarize_abs_errors
from .math_utils import normalize_quat_array, rmse


def _rotation_error_rmse_deg(q_ref, q_est) -> float:
    err = (R.from_quat(q_est).inv() * R.from_quat(q_ref)).magnitude()
    return float(np.sqrt(np.mean(np.degrees(err) ** 2)))


def _alignment_rmse(pos_est: np.ndarray, pos_ref: np.ndarray) -> float:
    diff = np.asarray(pos_est, dtype=float) - np.asarray(pos_ref, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _rot_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _solve_world_alignment_posyaw_standard(P: np.ndarray, Q: np.ndarray) -> dict[str, object]:
    src = np.asarray(P, dtype=float)
    dst = np.asarray(Q, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 2:
        raise ValueError("PosYaw world alignment requires at least 2 paired 3D points.")

    src_xy = src[:, :2]
    dst_xy = dst[:, :2]
    mu_src_xy = np.mean(src_xy, axis=0)
    mu_dst_xy = np.mean(dst_xy, axis=0)
    src0 = src_xy - mu_src_xy
    dst0 = dst_xy - mu_dst_xy
    cov = dst0.T @ src0
    yaw = float(np.arctan2(cov[1, 0] - cov[0, 1], cov[0, 0] + cov[1, 1]))
    Rw = _rot_z(yaw)
    tw = np.mean(dst, axis=0) - Rw @ np.mean(src, axis=0)
    pred = (Rw @ src.T).T + tw
    residuals = np.linalg.norm(pred - dst, axis=1)
    return {
        "R": Rw,
        "t": tw,
        "pred": pred,
        "residuals": residuals,
        "rmse_all_m": _alignment_rmse(pred, dst),
        "yaw_deg": float(np.degrees(yaw)),
    }


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


def _solve_world_alignment_posyaw_robust_trimmed(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    max_iterations: int = 3,
    min_inlier_ratio: float = 0.35,
    max_rejection_ratio: float = 0.65,
) -> dict[str, object]:
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    standard = _solve_world_alignment_posyaw_standard(P, Q)
    n = int(P.shape[0])
    min_inliers = max(6, int(np.ceil(float(min_inlier_ratio) * float(n))))
    if n < min_inliers:
        return {
            **standard,
            "mode": "posyaw_standard",
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
        fit = _solve_world_alignment_posyaw_standard(P[mask], Q[mask])
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

    robust = _solve_world_alignment_posyaw_standard(P[mask], Q[mask])
    pred_all = (robust["R"] @ P.T).T + robust["t"]
    residuals_all = np.linalg.norm(pred_all - Q, axis=1)
    standard_res = np.asarray(standard["residuals"], dtype=float)
    standard_inlier_rmse = _alignment_rmse(np.asarray(standard["pred"])[mask], Q[mask])
    robust_inlier_rmse = _alignment_rmse(pred_all[mask], Q[mask])
    robust_all_rmse = _alignment_rmse(pred_all, Q)
    rejected = int(n - np.count_nonzero(mask))
    improvement = standard_inlier_rmse - robust_inlier_rmse
    enough_outliers = rejected >= max(3, int(np.ceil(0.05 * n)))
    better_on_inliers = robust_inlier_rmse <= standard_inlier_rmse * 0.90 or improvement >= 0.05
    use_robust = bool(enough_outliers and better_on_inliers)
    selected = robust if use_robust else standard
    selected_mask = mask if use_robust else np.ones(n, dtype=bool)

    return {
        "R": selected["R"],
        "t": selected["t"],
        "pred": pred_all if use_robust else standard["pred"],
        "residuals": residuals_all if use_robust else standard_res,
        "rmse_all_m": robust_all_rmse if use_robust else float(standard["rmse_all_m"]),
        "mode": "posyaw_robust_trimmed" if use_robust else "posyaw_standard",
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
        "yaw_deg": float(selected["yaw_deg"]),
    }


def _robust_threshold(
    residuals: np.ndarray, *, min_inliers: int, max_rejection_ratio: float
) -> float:
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
    better_on_inliers = robust_inlier_rmse <= standard_inlier_rmse * 0.90 or improvement >= 0.05
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


def _select_stable_alignment_window(
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
    step_limit = np.maximum(
        float(max_est_step_m), float(max_step_ratio) * np.maximum(ref_step, 1e-3)
    )
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


def _select_motion_stable_prefix_window(
    pos_ref: np.ndarray,
    pos_est: np.ndarray,
    *,
    min_prefix_len: int = 80,
    min_prefix_ratio: float = 0.10,
    window_len: int = 80,
    max_bad_fraction: float = 0.15,
    max_est_step_m: float = 10.0,
    max_step_ratio: float = 80.0,
    max_window_step_m: float = 3.0,
    max_window_ratio: float = 20.0,
) -> tuple[np.ndarray, dict[str, float]]:
    pos_ref = np.asarray(pos_ref, dtype=float)
    pos_est = np.asarray(pos_est, dtype=float)
    n = int(min(pos_ref.shape[0], pos_est.shape[0]))
    mask_all = np.ones(n, dtype=bool)
    base_info = {
        "step3_stable_segment_used": 0.0,
        "step3_stable_segment_count": 1.0 if n > 0 else 0.0,
        "step3_stable_solve_count": float(n),
        "step3_stable_solve_ratio": 1.0 if n > 0 else 0.0,
        "step3_stable_segment_start_index": 0.0,
        "step3_stable_segment_end_index": float(max(0, n - 1)),
    }
    if n < max(3, int(min_prefix_len)):
        return mask_all, base_info

    ref_step = np.linalg.norm(np.diff(pos_ref[:n], axis=0), axis=1)
    est_step = np.linalg.norm(np.diff(pos_est[:n], axis=0), axis=1)
    finite = np.isfinite(ref_step) & np.isfinite(est_step)
    transition_limit = np.maximum(
        float(max_est_step_m),
        float(max_step_ratio) * np.maximum(ref_step, 1e-3),
    )
    bad_transition = (~finite) | (est_step > transition_limit)
    trans_n = int(bad_transition.size)
    if trans_n < 2:
        return mask_all, base_info

    win = max(5, min(int(window_len), trans_n))
    min_prefix = max(int(min_prefix_len), int(np.ceil(float(min_prefix_ratio) * float(n))))
    start_scan = min(max(0, min_prefix - 1), max(0, trans_n - win))
    cut_transition = None
    for start in range(start_scan, trans_n - win + 1):
        stop = start + win
        ref_win = ref_step[start:stop]
        est_win = est_step[start:stop]
        finite_win = finite[start:stop]
        bad_fraction = float(np.mean(bad_transition[start:stop]))
        if np.any(finite_win):
            ref_med = float(np.median(ref_win[finite_win]))
            est_med = float(np.median(est_win[finite_win]))
        else:
            ref_med = 0.0
            est_med = float("inf")
        window_limit = max(
            float(max_window_step_m),
            float(max_window_ratio) * max(ref_med, 1e-3),
        )
        if bad_fraction >= float(max_bad_fraction) or est_med > window_limit:
            cut_transition = start
            break

    if cut_transition is None:
        return mask_all, base_info

    end = int(cut_transition + 1)
    if end < min_prefix or end >= int(0.95 * n):
        return mask_all, base_info

    mask = np.zeros(n, dtype=bool)
    mask[:end] = True
    return mask, {
        "step3_stable_segment_used": 1.0,
        "step3_stable_segment_count": 1.0,
        "step3_stable_solve_count": float(end),
        "step3_stable_solve_ratio": float(end / max(1, n)),
        "step3_stable_segment_start_index": 0.0,
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


def _alignment_success_rate_proxy(
    pos_ref: np.ndarray, errors_m: np.ndarray, *, threshold_m: float = 10.0
) -> float:
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


def _alignment_gate_proxy(errors_m: np.ndarray, *, percentile: float = 5.0) -> float:
    finite = np.asarray(errors_m, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, float(percentile)))


def _select_world_alignment_variant(variants: list[dict]) -> dict:
    if not variants:
        raise ValueError("No Step3 solve variants were generated.")
    if len(variants) == 1:
        return variants[0]

    full = variants[0]
    original_stable = next(
        (item for item in variants[1:] if item["step3_solve_variant"] == "stable"), None
    )
    motion_stable = next(
        (item for item in variants[1:] if item["step3_solve_variant"] == "motion_stable_prefix"),
        None,
    )
    stable = original_stable if original_stable is not None else motion_stable
    if stable is None:
        return full

    def should_use_motion_prefix(candidate: dict) -> bool:
        candidate_sr = float(candidate["step3_sr_proxy"])
        candidate_gate = float(candidate["step3_gate_proxy_m"])
        candidate_anchor = float(candidate["step3_stable_anchor_rmse_m"])
        candidate_ratio = float(candidate.get("step3_stable_solve_ratio", 1.0))
        return bool(
            full_sr <= 0.02
            and full_gate >= 30.0
            and full_anchor >= 100.0
            and 0.12 <= candidate_ratio <= 0.25
            and candidate_sr >= full_sr + 0.02
            and candidate_gate <= min(full_gate * 0.10, full_gate - 5.0)
            and 2.0 <= candidate_anchor <= min(20.0, full_anchor * 0.10)
        )

    best_motion = min(
        [item for item in variants[1:] if item["step3_solve_variant"] == "motion_stable_prefix"],
        key=lambda item: (
            -float(item["step3_sr_proxy"]),
            float(item["step3_gate_proxy_m"]),
            float(item["step3_stable_anchor_rmse_m"]),
        ),
        default=None,
    )
    full_sr = float(full["step3_sr_proxy"])
    full_gate = float(full["step3_gate_proxy_m"])
    full_anchor = float(full["step3_stable_anchor_rmse_m"])

    if best_motion is not None and should_use_motion_prefix(best_motion):
        return best_motion

    if original_stable is None:
        return full

    stable = original_stable
    stable_sr = float(stable["step3_sr_proxy"])
    stable_gate = float(stable["step3_gate_proxy_m"])
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


def _extrinsic_rotation_candidates(R_base: np.ndarray) -> list[tuple[str, np.ndarray]]:
    flips = [
        ("base", np.eye(3, dtype=float)),
        ("right_rx180", R.from_euler("x", 180.0, degrees=True).as_matrix()),
        ("right_ry180", R.from_euler("y", 180.0, degrees=True).as_matrix()),
        ("right_rz180", R.from_euler("z", 180.0, degrees=True).as_matrix()),
    ]
    return [(name, np.asarray(R_base, dtype=float) @ flip) for name, flip in flips]


def _solve_extrinsic_world_candidate(
    *,
    name: str,
    R_calc,
    pr_sync,
    qr_sync,
    pos_gt_solve,
    quat_gt_solve,
    pr_solve,
    qr_solve,
    global_align_mode: str = "se3",
):
    Rr_mats = R.from_quat(qr_sync).as_matrix()
    Rr_mats_solve = R.from_quat(qr_solve).as_matrix()
    n_solve = int(np.asarray(pos_gt_solve).shape[0])
    full_mask = np.ones(n_solve, dtype=bool)
    stable_mask, stable_info = _select_stable_alignment_window(pos_gt_solve, pr_solve)
    motion_mask, motion_info = _select_motion_stable_prefix_window(pos_gt_solve, pr_solve)
    anchor_mask = (
        stable_mask
        if float(stable_info.get("step3_stable_segment_used", 0.0)) == 1.0
        else full_mask
    )
    anchor_indices_all = np.flatnonzero(anchor_mask)
    anchor_count = max(3, min(int(np.ceil(0.2 * n_solve)), int(anchor_indices_all.size)))
    anchor_indices = anchor_indices_all[:anchor_count]

    def solve_variant(
        variant_name: str, base_mask: np.ndarray, base_info: dict[str, float]
    ) -> dict[str, object]:
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
        pr_corr = np.asarray(pr_sync, dtype=float) - np.einsum(
            "nij,j->ni", Rr_mats, extrinsic_offset
        )
        pr_corr_solve = np.asarray(pr_solve, dtype=float) - np.einsum(
            "nij,j->ni",
            Rr_mats_solve,
            extrinsic_offset,
        )

        if str(global_align_mode).lower() in {"posyaw", "epa_posyaw"}:
            world_fit = _solve_world_alignment_posyaw_robust_trimmed(
                pr_corr_solve[solve_mask],
                pos_gt_solve[solve_mask],
            )
        else:
            world_fit = _solve_world_alignment_robust_trimmed(
                pr_corr_solve[solve_mask], pos_gt_solve[solve_mask]
            )
        Rw_variant = np.asarray(world_fit["R"], dtype=float)
        tw_variant = np.asarray(world_fit["t"], dtype=float)
        selected_mask_local = np.asarray(world_fit["selected_mask"], dtype=bool)
        selected_mask = np.zeros(n_solve, dtype=bool)
        selected_mask[np.flatnonzero(solve_mask)[selected_mask_local]] = True
        pred_step3_pairs = (Rw_variant @ pr_corr_solve.T).T + tw_variant
        selected_rmse_m = _alignment_rmse(
            pred_step3_pairs[selected_mask], pos_gt_solve[selected_mask]
        )
        pr_final_variant = (Rw_variant @ pr_corr.T).T + tw_variant
        eval_errors = np.linalg.norm(pred_step3_pairs - pos_gt_solve, axis=1)
        if anchor_indices.size >= 3:
            anchor_rmse = _alignment_rmse(
                pr_final_variant[anchor_indices], pos_gt_solve[anchor_indices]
            )
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
            "step3_selected_mask": selected_mask,
            "step3_rmse_selected_m": selected_rmse_m,
            "step3_sr_proxy": _alignment_success_rate_proxy(pos_gt_solve, eval_errors),
            "step3_gate_proxy_m": _alignment_gate_proxy(eval_errors),
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
    if float(motion_info.get("step3_stable_segment_used", 0.0)) == 1.0:
        variants.append(solve_variant("motion_stable_prefix", motion_mask, motion_info))
    selected_variant = _select_world_alignment_variant(variants)

    t_calc = selected_variant["t_calc"]
    Rw_calc = selected_variant["Rw_calc"]
    tw_calc = selected_variant["tw_calc"]
    pr_corrected = selected_variant["pr_corrected"]
    pr_corrected_solve = selected_variant["pr_corrected_solve"]
    pr_final = selected_variant["pr_final"]
    step3_selected_mask = np.asarray(selected_variant["step3_selected_mask"], dtype=bool)

    R_step2_mats = np.einsum("nij,jk->nik", Rr_mats, R_calc.T)
    q_step2 = normalize_quat_array(R.from_matrix(R_step2_mats).as_quat())
    q_step3_from_step2 = normalize_quat_array(
        (R.from_matrix(Rw_calc) * R.from_quat(q_step2)).as_quat()
    )
    q_step3_from_raw = normalize_quat_array(
        (R.from_matrix(Rw_calc) * R.from_quat(qr_sync)).as_quat()
    )
    rot_rmse_from_step2 = _rotation_error_rmse_deg(quat_gt_solve, q_step3_from_step2[:n_solve])
    rot_rmse_from_raw = _rotation_error_rmse_deg(quat_gt_solve, q_step3_from_raw[:n_solve])
    if str(global_align_mode).lower() in {"posyaw", "epa_posyaw"}:
        q_step3 = q_step3_from_step2
        orientation_mode = "step2_posyaw"
        rot_rmse_deg = rot_rmse_from_step2
    elif rot_rmse_from_raw < rot_rmse_from_step2:
        q_step3 = q_step3_from_raw
        orientation_mode = "world_raw"
        rot_rmse_deg = rot_rmse_from_raw
    else:
        q_step3 = q_step3_from_step2
        orientation_mode = "step2"
        rot_rmse_deg = rot_rmse_from_step2
    residual = _compute_extrinsic_residual_metrics(
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
        "step3_selected_mask": step3_selected_mask,
        "step3_rmse_selected_m": float(selected_variant["step3_rmse_selected_m"]),
        "rotation_ape_rmse_deg": rot_rmse_deg,
        "orientation_mode": orientation_mode,
        "rot_res_median_deg": float(residual["rot_res_median_deg"]),
        "rot_res_p95_deg": float(residual["rot_res_p95_deg"]),
        "step3_solve_variant": str(selected_variant["step3_solve_variant"]),
        "step3_solve_variant_code": float(
            {
                "full": 0.0,
                "stable": 1.0,
                "motion_stable_prefix": 2.0,
            }.get(str(selected_variant["step3_solve_variant"]), -1.0)
        ),
        "step3_full_rmse_selected_m": float(variants[0]["step3_rmse_selected_m"]),
        "step3_full_anchor_rmse_m": float(variants[0]["step3_stable_anchor_rmse_m"]),
        "step3_candidate_full_sr_proxy": float(variants[0]["step3_sr_proxy"]),
        "step3_candidate_full_gate_proxy_m": float(variants[0]["step3_gate_proxy_m"]),
        "step3_candidate_full_anchor_rmse_m": float(variants[0]["step3_stable_anchor_rmse_m"]),
        "step3_candidate_stable_sr_proxy": (
            max(float(v["step3_sr_proxy"]) for v in variants[1:])
            if len(variants) > 1
            else float("nan")
        ),
        "step3_candidate_stable_gate_proxy_m": (
            min(float(v["step3_gate_proxy_m"]) for v in variants[1:])
            if len(variants) > 1
            else float("nan")
        ),
        "step3_candidate_stable_anchor_rmse_m": (
            min(float(v["step3_stable_anchor_rmse_m"]) for v in variants[1:])
            if len(variants) > 1
            else float("nan")
        ),
        "step3_alignment_mode": str(selected_variant["step3_alignment_mode"]),
        "step3_standard_rmse_m": float(selected_variant["step3_standard_rmse_m"]),
        "step3_standard_rmse_on_inliers_m": float(
            selected_variant["step3_standard_rmse_on_inliers_m"]
        ),
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
        "step3_stable_segment_start_index": float(
            selected_variant["step3_stable_segment_start_index"]
        ),
        "step3_stable_segment_end_index": float(selected_variant["step3_stable_segment_end_index"]),
    }


def _select_extrinsic_world_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("No Step2/Step3 candidates were generated.")
    base = candidates[0]
    best_trans = min(
        float(c.get("step3_full_rmse_selected_m", c["step3_rmse_selected_m"])) for c in candidates
    )
    best_rot = min(float(c["rotation_ape_rmse_deg"]) for c in candidates)
    base_rel = float(base["rot_res_median_deg"])
    best_sr = max(
        float(c.get("step3_candidate_full_sr_proxy", c.get("step3_sr_proxy", 0.0)))
        for c in candidates
    )
    trans_limit = best_trans * 1.05 + 0.02
    rel_limit = base_rel + 2.0
    rot_limit = best_rot + 20.0
    eligible = [
        c
        for c in candidates
        if float(c.get("step3_full_rmse_selected_m", c["step3_rmse_selected_m"])) <= trans_limit
        and float(c["rot_res_median_deg"]) <= rel_limit
        and float(c["rotation_ape_rmse_deg"]) <= rot_limit
        and float(c.get("step3_candidate_full_sr_proxy", c.get("step3_sr_proxy", 0.0)))
        >= best_sr - 0.02
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


def _solve_extrinsic_and_world_alignment(
    *,
    pr_sync,
    qr_sync,
    pos_gt_solve,
    quat_gt_solve,
    pr_solve,
    qr_solve,
    global_align_mode: str = "se3",
):
    R_base = solve_extrinsic_rotation(quat_gt_solve, qr_solve)
    candidates = []
    for candidate_name, R_candidate in _extrinsic_rotation_candidates(R_base):
        candidates.append(
            _solve_extrinsic_world_candidate(
                name=candidate_name,
                R_calc=R_candidate,
                pr_sync=pr_sync,
                qr_sync=qr_sync,
                pos_gt_solve=pos_gt_solve,
                quat_gt_solve=quat_gt_solve,
                pr_solve=pr_solve,
                qr_solve=qr_solve,
                global_align_mode=global_align_mode,
            )
        )
    selected = _select_extrinsic_world_candidate(candidates)
    ambiguity_detected = (
        str(selected["candidate_name"]) != "base"
        or float(candidates[0]["rotation_ape_rmse_deg"])
        > float(selected["rotation_ape_rmse_deg"]) + 5.0
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
        "step3_selected_mask": selected["step3_selected_mask"],
        "step3_choice": {
            "step3_rmse_selected_m": float(selected["step3_rmse_selected_m"]),
            "orientation_candidate_count": float(len(candidates)),
            "orientation_candidate_selected_code": float(
                [c["candidate_name"] for c in candidates].index(selected["candidate_name"])
            ),
            "orientation_mode_code": float(
                1.0 if str(selected["orientation_mode"]) == "world_raw" else 0.0
            ),
            "orientation_candidate_rot_rmse_deg": float(selected["rotation_ape_rmse_deg"]),
            "orientation_candidate_trans_rmse_m": float(selected["step3_rmse_selected_m"]),
            "orientation_candidate_rel_rot_median_deg": float(selected["rot_res_median_deg"]),
            "orientation_ambiguity_detected": float(bool(ambiguity_detected)),
            "step3_solve_variant": str(selected["step3_solve_variant"]),
            "step3_solve_variant_code": float(selected["step3_solve_variant_code"]),
            "step3_candidate_full_sr_proxy": float(selected["step3_candidate_full_sr_proxy"]),
            "step3_candidate_full_gate_proxy_m": float(
                selected["step3_candidate_full_gate_proxy_m"]
            ),
            "step3_candidate_full_anchor_rmse_m": float(
                selected["step3_candidate_full_anchor_rmse_m"]
            ),
            "step3_candidate_stable_sr_proxy": float(selected["step3_candidate_stable_sr_proxy"]),
            "step3_candidate_stable_gate_proxy_m": float(
                selected["step3_candidate_stable_gate_proxy_m"]
            ),
            "step3_candidate_stable_anchor_rmse_m": float(
                selected["step3_candidate_stable_anchor_rmse_m"]
            ),
            "step3_alignment_mode": str(selected["step3_alignment_mode"]),
            "step3_alignment_mode_code": float(
                1.0
                if str(selected["step3_alignment_mode"])
                in {"robust_trimmed", "posyaw_robust_trimmed"}
                else 0.0
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


def _compute_extrinsic_residual_metrics(
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


def _compute_alignment_error_metrics(*, pos_gt, pr_sync, pr_corrected, pr_final):
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
