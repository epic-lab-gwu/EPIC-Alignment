from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .math_utils import normalize_quat_array


def _validate_pose_pairs(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_ref = np.asarray(pos_ref, dtype=float)
    p_est = np.asarray(pos_est, dtype=float)
    q_ref = normalize_quat_array(np.asarray(quat_ref, dtype=float))
    q_est = normalize_quat_array(np.asarray(quat_est, dtype=float))

    if p_ref.shape != p_est.shape:
        raise ValueError("EPICA Sim3 requires paired position arrays.")
    if q_ref.shape != q_est.shape:
        raise ValueError("EPICA Sim3 requires paired quaternion arrays.")
    if p_ref.shape[0] != q_ref.shape[0]:
        raise ValueError("EPICA Sim3 position and quaternion counts must match.")
    if p_ref.shape[0] < 3:
        raise ValueError("EPICA Sim3 requires at least 3 paired samples.")
    return p_ref, q_ref, p_est, q_est


def _umeyama_transform(
    pos_est: np.ndarray,
    pos_ref: np.ndarray,
    *,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(pos_est, dtype=float)
    dst = np.asarray(pos_ref, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Umeyama alignment requires at least 3 paired 3D points.")

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    r_fit = u @ s_mat @ vt

    scale = 1.0
    if with_scale:
        var_src = np.mean(np.sum(src_centered**2, axis=1))
        if var_src < 1e-12:
            raise ValueError("Degenerate source trajectory for Sim3 scale solve.")
        scale = float(np.trace(np.diag(d) @ s_mat) / var_src)
    t_fit = mu_dst - scale * (r_fit @ mu_src)
    return float(scale), r_fit, t_fit


def _solve_scale_translation_fixed_rotation(
    pos_ref: np.ndarray,
    pos_est: np.ndarray,
    r_fit: np.ndarray,
) -> tuple[float, np.ndarray]:
    p_ref = np.asarray(pos_ref, dtype=float)
    p_rot = (np.asarray(r_fit, dtype=float) @ np.asarray(pos_est, dtype=float).T).T
    mu_est = np.mean(p_rot, axis=0)
    mu_ref = np.mean(p_ref, axis=0)
    x = p_rot - mu_est
    y = p_ref - mu_ref
    denom = float(np.sum(x * x))
    if denom < 1e-12:
        raise ValueError("Degenerate source trajectory for EPICA Sim3 scale solve.")
    scale = float(np.sum(x * y) / denom)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"EPICA Sim3 estimated a non-positive scale: {scale}.")
    t_fit = mu_ref - scale * mu_est
    return scale, t_fit


def _apply_similarity(
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    scale: float,
    r_fit: np.ndarray,
    t_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_new = float(scale) * (np.asarray(r_fit, dtype=float) @ np.asarray(pos_est, dtype=float).T).T + np.asarray(
        t_fit, dtype=float
    )
    quat_new = normalize_quat_array(
        (R.from_matrix(np.asarray(r_fit, dtype=float)) * R.from_quat(np.asarray(quat_est, dtype=float))).as_quat()
    )
    return pos_new, quat_new


def _path_length(pos: np.ndarray) -> float:
    p = np.asarray(pos, dtype=float)
    if p.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _bbox_diag(pos: np.ndarray) -> float:
    p = np.asarray(pos, dtype=float)
    if p.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(np.max(p, axis=0) - np.min(p, axis=0)))


def _rank_info(pos: np.ndarray) -> tuple[int, float]:
    p = np.asarray(pos, dtype=float)
    if p.shape[0] < 3:
        return 0, 0.0
    centered = p - np.mean(p, axis=0)
    svals = np.linalg.svd(centered, compute_uv=False)
    if svals.size == 0 or float(svals[0]) <= 1e-12:
        return 0, 0.0
    ratios = svals / float(svals[0])
    rank = int(np.count_nonzero(ratios > 1e-3))
    second_ratio = float(ratios[1]) if ratios.size > 1 else 0.0
    return rank, second_ratio


def _orientation_span_deg(quat: np.ndarray) -> float:
    q = normalize_quat_array(np.asarray(quat, dtype=float))
    if q.shape[0] < 2:
        return 0.0
    rel = R.from_quat(q[0]).inv() * R.from_quat(q)
    return float(np.degrees(np.percentile(rel.magnitude(), 95)))


def _diagnostics(
    *,
    solver: str,
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    scale: float,
    r_fit: np.ndarray,
    t_fit: np.ndarray,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    pos_aligned, quat_aligned = _apply_similarity(pos_est, quat_est, scale, r_fit, t_fit)
    pos_residual_m = np.linalg.norm(pos_aligned - pos_ref, axis=1)
    rot_residual_deg = np.degrees((R.from_quat(quat_aligned).inv() * R.from_quat(quat_ref)).magnitude())
    info: dict[str, object] = {
        "sim3_solver": solver,
        "sim3_pair_count": int(np.asarray(pos_ref).shape[0]),
        "sim3_orientation_rmse_deg": float(np.sqrt(np.mean(rot_residual_deg * rot_residual_deg))),
        "sim3_orientation_mean_deg": float(np.mean(rot_residual_deg)),
        "sim3_position_rmse_m": float(np.sqrt(np.mean(pos_residual_m * pos_residual_m))),
        "sim3_position_mean_m": float(np.mean(pos_residual_m)),
    }
    if extra:
        info.update(extra)
    return info


def _candidate_windows(
    n: int,
    *,
    min_window_samples: int,
    fractions: tuple[float, ...],
    stride_fraction: float = 0.5,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    stride_fraction = float(np.clip(stride_fraction, 0.05, 1.0))
    for frac in fractions:
        size = max(int(min_window_samples), int(np.ceil(float(frac) * n)))
        size = min(n, max(3, size))
        stride = max(1, int(size * stride_fraction))
        starts = list(range(0, max(1, n - size + 1), stride))
        if starts[-1] != n - size:
            starts.append(n - size)
        for start in starts:
            item = (int(start), int(start + size))
            if item not in seen:
                windows.append(item)
                seen.add(item)
    if n >= max(3, min_window_samples):
        item = (0, int(n))
        if item not in seen:
            windows.append(item)
    return windows


def solve_epa_sim3_v1(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    *,
    min_window_samples: int = 30,
    window_fractions: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40),
    min_path_length_m: float = 1.0,
    min_bbox_diag_m: float = 0.25,
    max_anchor_rmse_ratio: float = 0.08,
    max_anchor_rmse_m: float = 2.0,
    max_orientation_rmse_deg: float = 90.0,
    min_scale: float = 1e-6,
    max_scale: float = 1e6,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Robust window-anchor Sim3 from estimate to reference.

    Candidate windows across the whole trajectory estimate position-only
    Sim3 transforms. Windows are scored only by their own alignment quality
    and observability; the rest of the trajectory is left for evaluation so
    drift outside the selected window is not averaged away.
    """

    p_ref, q_ref, p_est, q_est = _validate_pose_pairs(pos_ref, quat_ref, pos_est, quat_est)
    n = int(p_ref.shape[0])
    windows = _candidate_windows(n, min_window_samples=int(min_window_samples), fractions=tuple(window_fractions))
    best: dict[str, object] | None = None
    rejected: list[str] = []

    for start, end in windows:
        pr = p_ref[start:end]
        pe = p_est[start:end]
        qr = q_ref[start:end]
        qe = q_est[start:end]
        count = int(end - start)
        ref_path = _path_length(pr)
        est_path = _path_length(pe)
        ref_bbox = _bbox_diag(pr)
        est_bbox = _bbox_diag(pe)
        ref_rank, ref_rank_ratio = _rank_info(pr)
        est_rank, est_rank_ratio = _rank_info(pe)
        ori_span = max(_orientation_span_deg(qr), _orientation_span_deg(qe))

        reject_reason = ""
        if count < max(3, int(min_window_samples)):
            reject_reason = "too_few_samples"
        elif ref_path < float(min_path_length_m) or est_path <= 1e-9:
            reject_reason = "insufficient_path_length"
        elif ref_bbox < float(min_bbox_diag_m) or est_bbox <= 1e-9:
            reject_reason = "insufficient_spatial_extent"
        elif min(ref_rank, est_rank) < 2 and ori_span < 5.0:
            reject_reason = "degenerate_motion"
        if reject_reason:
            rejected.append(f"{start}:{end}:{reject_reason}")
            continue

        try:
            scale, r_fit, t_fit = _umeyama_transform(pe, pr, with_scale=True)
        except Exception as exc:
            rejected.append(f"{start}:{end}:solve_failed:{exc}")
            continue
        if not np.isfinite(scale) or scale <= 0.0 or scale < float(min_scale) or scale > float(max_scale):
            rejected.append(f"{start}:{end}:scale_out_of_range:{scale}")
            continue

        pos_win, quat_win = _apply_similarity(pe, qe, scale, r_fit, t_fit)
        pos_res = np.linalg.norm(pos_win - pr, axis=1)
        rot_res = np.degrees((R.from_quat(quat_win).inv() * R.from_quat(qr)).magnitude())
        pos_rmse = float(np.sqrt(np.mean(pos_res * pos_res)))
        rot_rmse = float(np.sqrt(np.mean(rot_res * rot_res)))
        rmse_ratio = float(pos_rmse / max(ref_path, 1e-9))
        reliable = bool(
            pos_rmse <= max(float(max_anchor_rmse_m), float(max_anchor_rmse_ratio) * ref_path)
            and rot_rmse <= float(max_orientation_rmse_deg)
        )

        scale_penalty = 0.02 * float(abs(np.log10(float(scale)))) if scale > 0.0 else 1e6
        rank_penalty = 0.1 if min(ref_rank, est_rank) < 2 else 0.0
        observability_penalty = 0.05 / max(min(ref_rank_ratio, est_rank_ratio, 1.0), 0.05)
        length_bonus = 0.01 * float(np.log1p(count))
        score = rmse_ratio + 0.003 * rot_rmse + scale_penalty + rank_penalty + observability_penalty - length_bonus
        candidate = {
            "score": float(score),
            "reliable": reliable,
            "scale": float(scale),
            "r_fit": r_fit,
            "t_fit": t_fit,
            "start": int(start),
            "end": int(end),
            "count": int(count),
            "pos_rmse": float(pos_rmse),
            "rot_rmse": float(rot_rmse),
            "rmse_ratio": float(rmse_ratio),
            "ref_path": float(ref_path),
            "est_path": float(est_path),
            "ref_bbox": float(ref_bbox),
            "est_bbox": float(est_bbox),
            "ref_rank": int(ref_rank),
            "est_rank": int(est_rank),
            "ref_rank_ratio": float(ref_rank_ratio),
            "est_rank_ratio": float(est_rank_ratio),
            "orientation_span_deg": float(ori_span),
        }
        if best is None or (bool(candidate["reliable"]) and not bool(best["reliable"])) or (
            bool(candidate["reliable"]) == bool(best["reliable"]) and float(candidate["score"]) < float(best["score"])
        ):
            best = candidate

    if best is None or not bool(best["reliable"]):
        info = _diagnostics(
            solver="epa_sim3_v1",
            pos_ref=p_ref,
            quat_ref=q_ref,
            pos_est=p_est,
            quat_est=q_est,
            scale=1.0,
            r_fit=np.eye(3),
            t_fit=np.zeros(3),
            extra={
                "sim3_anchor_status": "no_reliable_anchor",
                "sim3_reliable": False,
                "sim3_candidate_count": int(len(windows)),
                "sim3_candidate_rejected_count": int(len(rejected)),
                "sim3_anchor_failures": "; ".join(rejected[:20]),
            },
        )
        return 1.0, np.eye(3), np.zeros(3), info

    info = _diagnostics(
        solver="epa_sim3_v1",
        pos_ref=p_ref,
        quat_ref=q_ref,
        pos_est=p_est,
        quat_est=q_est,
        scale=float(best["scale"]),
        r_fit=np.asarray(best["r_fit"], dtype=float),
        t_fit=np.asarray(best["t_fit"], dtype=float),
        extra={
            "sim3_anchor_status": "ok",
            "sim3_reliable": True,
            "sim3_candidate_count": int(len(windows)),
            "sim3_candidate_rejected_count": int(len(rejected)),
            "sim3_anchor_start_index": int(best["start"]),
            "sim3_anchor_end_index": int(best["end"]),
            "sim3_anchor_samples": int(best["count"]),
            "sim3_anchor_sample_ratio": float(best["count"] / max(1, n)),
            "sim3_anchor_score": float(best["score"]),
            "sim3_anchor_position_rmse_m": float(best["pos_rmse"]),
            "sim3_anchor_orientation_rmse_deg": float(best["rot_rmse"]),
            "sim3_anchor_rmse_path_ratio": float(best["rmse_ratio"]),
            "sim3_anchor_ref_path_m": float(best["ref_path"]),
            "sim3_anchor_est_path_m": float(best["est_path"]),
            "sim3_anchor_ref_bbox_m": float(best["ref_bbox"]),
            "sim3_anchor_est_bbox_m": float(best["est_bbox"]),
            "sim3_anchor_ref_rank": int(best["ref_rank"]),
            "sim3_anchor_est_rank": int(best["est_rank"]),
            "sim3_anchor_ref_rank_ratio": float(best["ref_rank_ratio"]),
            "sim3_anchor_est_rank_ratio": float(best["est_rank_ratio"]),
            "sim3_anchor_orientation_span_deg": float(best["orientation_span_deg"]),
            "sim3_anchor_failures": "; ".join(rejected[:20]),
        },
    )
    return float(best["scale"]), np.asarray(best["r_fit"], dtype=float), np.asarray(best["t_fit"], dtype=float), info


def _candidate_cross_rmse_ratio(
    candidate: dict[str, object],
    target: dict[str, object],
    p_ref: np.ndarray,
    p_est: np.ndarray,
) -> tuple[float, float]:
    start = int(target["start"])
    end = int(target["end"])
    scale = float(candidate["scale"])
    r_fit = np.asarray(candidate["r_fit"], dtype=float)
    t_fit = np.asarray(candidate["t_fit"], dtype=float)
    pos_new = scale * (r_fit @ p_est[start:end].T).T + t_fit
    residual = np.linalg.norm(pos_new - p_ref[start:end], axis=1)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    ratio = float(rmse / max(float(target["ref_path"]), 1e-9))
    return rmse, ratio


def _rotation_gap_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    return float(np.degrees((R.from_matrix(r_a) * R.from_matrix(r_b).inv()).magnitude()))


def _compatible_epa_candidates(
    a: dict[str, object],
    b: dict[str, object],
    p_ref: np.ndarray,
    p_est: np.ndarray,
    *,
    max_scale_log_gap: float,
    max_rotation_gap_deg: float,
    max_cross_rmse_ratio: float,
    max_cross_rmse_m: float,
) -> bool:
    scale_a = max(float(a["scale"]), 1e-12)
    scale_b = max(float(b["scale"]), 1e-12)
    scale_gap = abs(float(np.log(scale_a / scale_b)))
    if scale_gap > float(max_scale_log_gap):
        return False
    rot_gap = _rotation_gap_deg(np.asarray(a["r_fit"], dtype=float), np.asarray(b["r_fit"], dtype=float))
    if rot_gap > float(max_rotation_gap_deg):
        return False

    rmse_ab, ratio_ab = _candidate_cross_rmse_ratio(a, b, p_ref, p_est)
    rmse_ba, ratio_ba = _candidate_cross_rmse_ratio(b, a, p_ref, p_est)
    limit_ab = max(float(max_cross_rmse_m), float(max_cross_rmse_ratio) * float(b["ref_path"]))
    limit_ba = max(float(max_cross_rmse_m), float(max_cross_rmse_ratio) * float(a["ref_path"]))
    return bool(rmse_ab <= limit_ab and rmse_ba <= limit_ba and ratio_ab <= max_cross_rmse_ratio and ratio_ba <= max_cross_rmse_ratio)


def _connected_components(neighbors: list[set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for i in range(len(neighbors)):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in neighbors[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def _motion_health_profile(
    p_ref: np.ndarray,
    p_est: np.ndarray,
    *,
    timestamps_s: np.ndarray | None = None,
    max_step_scale_deviation: float,
    max_step_motion_deviation: float,
    max_abs_est_speed_mps: float,
) -> dict[str, object]:
    n = int(np.asarray(p_ref).shape[0])
    if n < 2:
        return {
            "sample_healthy": np.ones(n, dtype=bool),
            "step_bad": np.zeros(max(0, n - 1), dtype=bool),
            "step_scale_ratio_median": float("nan"),
            "est_step_median_m": float("nan"),
            "bad_step_ratio": 0.0,
            "healthy_sample_ratio": 1.0,
        }

    ref_step = np.linalg.norm(np.diff(np.asarray(p_ref, dtype=float), axis=0), axis=1)
    est_step = np.linalg.norm(np.diff(np.asarray(p_est, dtype=float), axis=0), axis=1)
    valid_ratio = (ref_step > 1e-6) & (est_step > 1e-9) & np.isfinite(ref_step) & np.isfinite(est_step)
    if np.any(valid_ratio):
        ratios = est_step[valid_ratio] / np.maximum(ref_step[valid_ratio], 1e-9)
        ratio_med = float(np.median(ratios[ratios > 0.0])) if np.any(ratios > 0.0) else 1.0
    else:
        ratio_med = 1.0
    ratio_med = max(ratio_med, 1e-12)

    positive_est = est_step[np.isfinite(est_step) & (est_step > 1e-9)]
    est_step_med = float(np.median(positive_est)) if positive_est.size else 1.0
    est_step_med = max(est_step_med, 1e-12)
    q75 = float(np.percentile(positive_est, 75)) if positive_est.size else est_step_med
    q25 = float(np.percentile(positive_est, 25)) if positive_est.size else est_step_med
    iqr = max(q75 - q25, 1e-12)

    step_bad = np.zeros(n - 1, dtype=bool)
    if np.any(valid_ratio):
        ratio = np.ones(n - 1, dtype=float)
        ratio[valid_ratio] = est_step[valid_ratio] / np.maximum(ref_step[valid_ratio], 1e-9)
        ratio_dev = np.maximum(ratio / ratio_med, ratio_med / np.maximum(ratio, 1e-12))
        step_bad |= ratio_dev > float(max_step_scale_deviation)
    motion_dev = est_step / est_step_med
    step_bad |= motion_dev > float(max_step_motion_deviation)
    step_bad |= est_step > (q75 + 8.0 * iqr)
    speed_bad_ratio = 0.0
    speed_p95 = float("nan")
    if timestamps_s is not None:
        stamps = np.asarray(timestamps_s, dtype=float).reshape(-1)
        if stamps.size == n:
            dt = np.diff(stamps)
            valid_dt = np.isfinite(dt) & (dt > 1e-6)
            speed = np.full(n - 1, np.nan, dtype=float)
            speed[valid_dt] = est_step[valid_dt] / dt[valid_dt]
            finite_speed = speed[np.isfinite(speed)]
            if finite_speed.size:
                speed_p95 = float(np.percentile(finite_speed, 95))
                speed_bad = speed > float(max_abs_est_speed_mps)
                step_bad |= np.where(np.isfinite(speed_bad), speed_bad, False)
                speed_bad_ratio = float(np.mean(speed_bad[np.isfinite(speed)]))
    step_bad &= np.isfinite(est_step)

    sample_healthy = np.ones(n, dtype=bool)
    bad_ids = np.flatnonzero(step_bad)
    if bad_ids.size:
        sample_healthy[bad_ids] = False
        sample_healthy[np.minimum(bad_ids + 1, n - 1)] = False

    return {
        "sample_healthy": sample_healthy,
        "step_bad": step_bad,
        "step_scale_ratio_median": float(ratio_med),
        "est_step_median_m": float(est_step_med),
        "est_speed_p95_mps": speed_p95,
        "est_speed_bad_ratio": speed_bad_ratio,
        "bad_step_ratio": float(np.mean(step_bad)) if step_bad.size else 0.0,
        "healthy_sample_ratio": float(np.mean(sample_healthy)) if sample_healthy.size else 0.0,
    }


def _window_health(
    start: int,
    end: int,
    health: dict[str, object],
) -> tuple[float, float]:
    sample = np.asarray(health["sample_healthy"], dtype=bool)
    step_bad = np.asarray(health["step_bad"], dtype=bool)
    sample_ratio = float(np.mean(sample[start:end])) if end > start else 0.0
    if end - start > 1:
        bad_ratio = float(np.mean(step_bad[start : end - 1]))
    else:
        bad_ratio = 0.0
    return sample_ratio, bad_ratio


def _support_segment_count(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return 0
    starts = values & np.concatenate(([True], ~values[:-1]))
    return int(np.count_nonzero(starts))


def _candidate_global_support(
    candidate: dict[str, object],
    targets: list[dict[str, object]],
    p_ref: np.ndarray,
    q_ref: np.ndarray,
    p_est: np.ndarray,
    q_est: np.ndarray,
    *,
    max_anchor_rmse_ratio: float,
    max_anchor_rmse_m: float,
    max_orientation_rmse_deg: float,
) -> dict[str, object]:
    n = int(p_ref.shape[0])
    support_sample = np.zeros(n, dtype=bool)
    support_rmses: list[float] = []
    support_windows = 0
    scale = float(candidate["scale"])
    r_fit = np.asarray(candidate["r_fit"], dtype=float)
    t_fit = np.asarray(candidate["t_fit"], dtype=float)

    for target in targets:
        start = int(target["start"])
        end = int(target["end"])
        pos_new, quat_new = _apply_similarity(p_est[start:end], q_est[start:end], scale, r_fit, t_fit)
        pos_res = np.linalg.norm(pos_new - p_ref[start:end], axis=1)
        rot_res = np.degrees((R.from_quat(quat_new).inv() * R.from_quat(q_ref[start:end])).magnitude())
        pos_rmse = float(np.sqrt(np.mean(pos_res * pos_res)))
        rot_rmse = float(np.sqrt(np.mean(rot_res * rot_res)))
        limit = max(float(max_anchor_rmse_m), float(max_anchor_rmse_ratio) * float(target["ref_path"]))
        if pos_rmse <= limit and rot_rmse <= float(max_orientation_rmse_deg):
            support_sample[start:end] = True
            support_rmses.append(pos_rmse)
            support_windows += 1

    step_lengths = np.linalg.norm(np.diff(p_ref, axis=0), axis=1) if n > 1 else np.zeros(0, dtype=float)
    total_distance = float(np.sum(step_lengths))
    if step_lengths.size:
        support_step = support_sample[:-1] & support_sample[1:]
        support_distance = float(np.sum(step_lengths[support_step]))
    else:
        support_distance = 0.0
    support_ids = np.flatnonzero(support_sample)
    support_span_ratio = (
        float((int(support_ids[-1]) - int(support_ids[0]) + 1) / max(1, n))
        if support_ids.size
        else 0.0
    )
    rmses = np.asarray(support_rmses, dtype=float)
    return {
        "support_window_count": int(support_windows),
        "support_distance_m": support_distance,
        "support_distance_ratio": float(support_distance / max(total_distance, 1e-9)),
        "support_time_ratio": float(np.mean(support_sample)) if support_sample.size else 0.0,
        "support_span_ratio": support_span_ratio,
        "support_segment_count": _support_segment_count(support_sample),
        "support_rmse_median_m": float(np.median(rmses)) if rmses.size else float("nan"),
        "support_rmse_mean_m": float(np.mean(rmses)) if rmses.size else float("nan"),
    }


def solve_epa_sim3_v2(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    *,
    min_window_samples: int = 30,
    window_fractions: tuple[float, ...] = (0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70),
    window_stride_fraction: float = 0.25,
    min_path_length_m: float = 1.0,
    min_bbox_diag_m: float = 0.25,
    max_anchor_rmse_ratio: float = 0.08,
    max_anchor_rmse_m: float = 2.0,
    max_orientation_rmse_deg: float = 90.0,
    min_scale: float = 1e-6,
    max_scale: float = 1e6,
    top_k_consensus: int = 16,
    max_consensus_scale_gap: float = float(np.log(1.35)),
    max_consensus_rotation_gap_deg: float = 12.0,
    max_consensus_cross_rmse_ratio: float = 0.10,
    max_consensus_cross_rmse_m: float = 3.0,
    min_consensus_span_ratio: float = 0.25,
    max_step_scale_deviation: float = 8.0,
    max_step_motion_deviation: float = 20.0,
    min_window_health_sample_ratio: float = 0.75,
    max_window_bad_step_ratio: float = 0.25,
    min_global_support_distance_ratio: float = 0.35,
    min_global_support_time_ratio: float = 0.25,
    min_global_support_window_count: int = 2,
    min_global_health_sample_ratio: float = 0.80,
    max_global_bad_step_ratio: float = 0.15,
    max_global_speed_bad_ratio: float = 0.05,
    min_reliable_scale: float = 0.1,
    max_reliable_scale: float = 10.0,
    timestamps_s: np.ndarray | None = None,
    max_abs_est_speed_mps: float = 100.0,
    max_solver_samples: int = 2500,
    max_support_candidates: int = 48,
    max_support_targets: int = 96,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Consensus window-anchor Sim3 from estimate to reference.

    V2 keeps V1's full-trajectory window search, but a single low-residual
    window is no longer enough to call the transform reliable. Candidate
    anchors must agree with other anchors on held-out anchor windows before
    receiving high or medium confidence.
    """

    p_ref_full, q_ref_full, p_est_full, q_est_full = _validate_pose_pairs(pos_ref, quat_ref, pos_est, quat_est)
    n_full = int(p_ref_full.shape[0])
    sample_idx: np.ndarray | None = None
    if int(max_solver_samples) > 0 and n_full > int(max_solver_samples):
        sample_idx = np.unique(np.linspace(0, n_full - 1, int(max_solver_samples), dtype=int))
        p_ref = p_ref_full[sample_idx]
        q_ref = q_ref_full[sample_idx]
        p_est = p_est_full[sample_idx]
        q_est = q_est_full[sample_idx]
        ts_work = None if timestamps_s is None else np.asarray(timestamps_s, dtype=float).reshape(-1)[sample_idx]
    else:
        p_ref = p_ref_full
        q_ref = q_ref_full
        p_est = p_est_full
        q_est = q_est_full
        ts_work = timestamps_s
    n = int(p_ref.shape[0])
    health = _motion_health_profile(
        p_ref,
        p_est,
        timestamps_s=ts_work,
        max_step_scale_deviation=float(max_step_scale_deviation),
        max_step_motion_deviation=float(max_step_motion_deviation),
        max_abs_est_speed_mps=float(max_abs_est_speed_mps),
    )
    windows = _candidate_windows(
        n,
        min_window_samples=int(min_window_samples),
        fractions=tuple(window_fractions),
        stride_fraction=float(window_stride_fraction),
    )
    candidates: list[dict[str, object]] = []
    rejected: list[str] = []

    for start, end in windows:
        pr = p_ref[start:end]
        pe = p_est[start:end]
        qr = q_ref[start:end]
        qe = q_est[start:end]
        count = int(end - start)
        ref_path = _path_length(pr)
        est_path = _path_length(pe)
        ref_bbox = _bbox_diag(pr)
        est_bbox = _bbox_diag(pe)
        ref_rank, ref_rank_ratio = _rank_info(pr)
        est_rank, est_rank_ratio = _rank_info(pe)
        ori_span = max(_orientation_span_deg(qr), _orientation_span_deg(qe))
        health_sample_ratio, bad_step_ratio = _window_health(start, end, health)

        reject_reason = ""
        if count < max(3, int(min_window_samples)):
            reject_reason = "too_few_samples"
        elif health_sample_ratio < float(min_window_health_sample_ratio) or bad_step_ratio > float(max_window_bad_step_ratio):
            reject_reason = "unhealthy_motion"
        elif ref_path < float(min_path_length_m) or est_path <= 1e-9:
            reject_reason = "insufficient_path_length"
        elif ref_bbox < float(min_bbox_diag_m) or est_bbox <= 1e-9:
            reject_reason = "insufficient_spatial_extent"
        elif min(ref_rank, est_rank) < 2 and ori_span < 5.0:
            reject_reason = "degenerate_motion"
        if reject_reason:
            rejected.append(f"{start}:{end}:{reject_reason}")
            continue

        try:
            scale, r_fit, t_fit = _umeyama_transform(pe, pr, with_scale=True)
        except Exception as exc:
            rejected.append(f"{start}:{end}:solve_failed:{exc}")
            continue
        if not np.isfinite(scale) or scale <= 0.0 or scale < float(min_scale) or scale > float(max_scale):
            rejected.append(f"{start}:{end}:scale_out_of_range:{scale}")
            continue

        pos_win, quat_win = _apply_similarity(pe, qe, scale, r_fit, t_fit)
        pos_res = np.linalg.norm(pos_win - pr, axis=1)
        rot_res = np.degrees((R.from_quat(quat_win).inv() * R.from_quat(qr)).magnitude())
        pos_rmse = float(np.sqrt(np.mean(pos_res * pos_res)))
        rot_rmse = float(np.sqrt(np.mean(rot_res * rot_res)))
        rmse_ratio = float(pos_rmse / max(ref_path, 1e-9))
        reliable = bool(
            pos_rmse <= max(float(max_anchor_rmse_m), float(max_anchor_rmse_ratio) * ref_path)
            and rot_rmse <= float(max_orientation_rmse_deg)
        )

        scale_penalty = 0.02 * float(abs(np.log10(float(scale)))) if scale > 0.0 else 1e6
        rank_penalty = 0.1 if min(ref_rank, est_rank) < 2 else 0.0
        observability_penalty = 0.05 / max(min(ref_rank_ratio, est_rank_ratio, 1.0), 0.05)
        length_bonus = 0.01 * float(np.log1p(count))
        score = rmse_ratio + 0.003 * rot_rmse + scale_penalty + rank_penalty + observability_penalty - length_bonus
        candidates.append(
            {
                "score": float(score),
                "reliable": reliable,
                "scale": float(scale),
                "r_fit": r_fit,
                "t_fit": t_fit,
                "start": int(start),
                "end": int(end),
                "count": int(count),
                "pos_rmse": float(pos_rmse),
                "rot_rmse": float(rot_rmse),
                "rmse_ratio": float(rmse_ratio),
                "ref_path": float(ref_path),
                "est_path": float(est_path),
                "ref_bbox": float(ref_bbox),
                "est_bbox": float(est_bbox),
                "ref_rank": int(ref_rank),
                "est_rank": int(est_rank),
                "ref_rank_ratio": float(ref_rank_ratio),
                "est_rank_ratio": float(est_rank_ratio),
                "orientation_span_deg": float(ori_span),
                "health_sample_ratio": float(health_sample_ratio),
                "bad_step_ratio": float(bad_step_ratio),
                "anchor_allowed": bool(not (int(start) == 0 and int(end) == n)),
            }
        )

    reliable_candidates = [c for c in candidates if bool(c["reliable"]) and bool(c.get("anchor_allowed", True))]
    reliable_candidates.sort(key=lambda item: float(item["score"]))
    if not reliable_candidates:
        info = _diagnostics(
            solver="epa_sim3_v2",
            pos_ref=p_ref_full,
            quat_ref=q_ref_full,
            pos_est=p_est_full,
            quat_est=q_est_full,
            scale=1.0,
            r_fit=np.eye(3),
            t_fit=np.zeros(3),
            extra={
                "sim3_version": "v2",
                "sim3_anchor_status": "no_reliable_anchor",
                "sim3_consensus_status": "no_anchor",
                "sim3_confidence": "no_anchor",
                "sim3_reliable": False,
                "sim3_candidate_count": int(len(windows)),
                "sim3_candidate_reliable_count": 0,
                "sim3_candidate_rejected_count": int(len(rejected)),
                "sim3_solver_sample_count": int(n),
                "sim3_original_pair_count": int(n_full),
                "sim3_solver_subsampled": bool(sample_idx is not None),
                "sim3_max_solver_samples": int(max_solver_samples),
                "sim3_window_stride_fraction": float(window_stride_fraction),
                "sim3_window_fractions": ",".join(str(float(x)) for x in window_fractions),
                "sim3_health_sample_ratio": float(health["healthy_sample_ratio"]),
                "sim3_health_bad_step_ratio": float(health["bad_step_ratio"]),
                "sim3_health_step_scale_ratio_median": float(health["step_scale_ratio_median"]),
                "sim3_health_est_step_median_m": float(health["est_step_median_m"]),
                "sim3_health_est_speed_p95_mps": float(health["est_speed_p95_mps"]),
                "sim3_health_est_speed_bad_ratio": float(health["est_speed_bad_ratio"]),
                "sim3_health_max_abs_est_speed_mps": float(max_abs_est_speed_mps),
                "sim3_health_min_global_sample_ratio": float(min_global_health_sample_ratio),
                "sim3_health_max_global_bad_step_ratio": float(max_global_bad_step_ratio),
                "sim3_health_max_global_speed_bad_ratio": float(max_global_speed_bad_ratio),
                "sim3_failure_reason": "no_reliable_anchor",
                "sim3_anchor_failures": "; ".join(rejected[:20]),
            },
        )
        return 1.0, np.eye(3), np.zeros(3), info

    global_health_ok = bool(
        float(health["healthy_sample_ratio"]) >= float(min_global_health_sample_ratio)
        and float(health["bad_step_ratio"]) <= float(max_global_bad_step_ratio)
        and float(health["est_speed_bad_ratio"]) <= float(max_global_speed_bad_ratio)
    )
    support_targets = sorted(candidates, key=lambda item: float(item["score"]))[: max(1, int(max_support_targets))]
    support_eval_candidates = reliable_candidates[: max(1, int(max_support_candidates))]
    for item in support_eval_candidates:
        item.update(
            _candidate_global_support(
                item,
                support_targets,
                p_ref,
                q_ref,
                p_est,
                q_est,
                max_anchor_rmse_ratio=float(max_anchor_rmse_ratio),
                max_anchor_rmse_m=float(max_anchor_rmse_m),
                max_orientation_rmse_deg=float(max_orientation_rmse_deg),
            )
        )
        support_distance_ratio_i = float(item.get("support_distance_ratio", 0.0))
        support_time_ratio_i = float(item.get("support_time_ratio", 0.0))
        support_span_ratio_i = float(item.get("support_span_ratio", 0.0))
        support_window_count_i = int(item.get("support_window_count", 0) or 0)
        scale_i = float(item.get("scale", float("nan")))
        scale_ok_i = bool(
            np.isfinite(scale_i)
            and scale_i >= float(min_reliable_scale)
            and scale_i <= float(max_reliable_scale)
        )
        item["support_gate_ok"] = bool(
            scale_ok_i
            and global_health_ok
            and support_distance_ratio_i >= float(min_global_support_distance_ratio)
            and support_time_ratio_i >= float(min_global_support_time_ratio)
            and support_span_ratio_i >= float(min_consensus_span_ratio)
            and support_window_count_i >= int(min_global_support_window_count)
        )
        item["support_scale_ok"] = scale_ok_i

    reliable_candidates = support_eval_candidates
    reliable_candidates.sort(
        key=lambda item: (
            not bool(item.get("support_gate_ok", False)),
            -float(item.get("support_distance_ratio", 0.0)),
            -float(item.get("support_time_ratio", 0.0)),
            -float(item.get("support_span_ratio", 0.0)),
            float(item.get("support_rmse_median_m", np.inf)),
            float(item["score"]),
        )
    )
    top = reliable_candidates[: max(1, int(top_k_consensus))]
    neighbors = [set([i]) for i in range(len(top))]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if _compatible_epa_candidates(
                top[i],
                top[j],
                p_ref,
                p_est,
                max_scale_log_gap=float(max_consensus_scale_gap),
                max_rotation_gap_deg=float(max_consensus_rotation_gap_deg),
                max_cross_rmse_ratio=float(max_consensus_cross_rmse_ratio),
                max_cross_rmse_m=float(max_consensus_cross_rmse_m),
            ):
                neighbors[i].add(j)
                neighbors[j].add(i)

    selected = reliable_candidates[0]
    components = _connected_components(neighbors)
    selected_top_idx = 0
    selected_comp = next((comp for comp in components if selected_top_idx in comp), [selected_top_idx])
    cluster = [top[i] for i in selected_comp]

    scales = np.array([float(c["scale"]) for c in cluster], dtype=float)
    rotations = [np.asarray(c["r_fit"], dtype=float) for c in cluster]
    scale_spread = float(np.max(np.log(scales)) - np.min(np.log(scales))) if scales.size > 1 else 0.0
    rot_spread = 0.0
    if len(rotations) > 1:
        rot_spread = float(max(_rotation_gap_deg(rotations[i], rotations[j]) for i in range(len(rotations)) for j in range(i + 1, len(rotations))))
    span_ratio = float((max(int(c["end"]) for c in cluster) - min(int(c["start"]) for c in cluster)) / max(1, n))
    consensus_count = int(len(cluster))
    consensus_ratio = float(consensus_count / max(1, len(top)))
    min_span = float(min_consensus_span_ratio)
    support_distance_ratio = float(selected.get("support_distance_ratio", 0.0))
    support_time_ratio = float(selected.get("support_time_ratio", 0.0))
    support_span_ratio = float(selected.get("support_span_ratio", 0.0))
    support_window_count = int(selected.get("support_window_count", 0) or 0)
    support_ok = bool(selected.get("support_gate_ok", False))
    if support_ok and support_distance_ratio >= 0.60 and support_span_ratio >= 0.50 and support_window_count >= 3:
        confidence = "high"
        consensus_status = "ok"
    elif support_ok:
        confidence = "medium"
        consensus_status = "ok"
    elif support_ok and consensus_count == 1 and len(reliable_candidates) == 1:
        confidence = "low"
        consensus_status = "single_anchor"
    else:
        confidence = "conflicting_anchors"
        consensus_status = "conflicting"
    is_reliable = bool(confidence in {"high", "medium"})
    applied_scale = float(selected["scale"]) if is_reliable else 1.0
    applied_r = np.asarray(selected["r_fit"], dtype=float) if is_reliable else np.eye(3)
    applied_t = np.asarray(selected["t_fit"], dtype=float) if is_reliable else np.zeros(3)
    if is_reliable:
        failure_reason = ""
    elif not global_health_ok:
        failure_reason = "global_motion_unhealthy"
    else:
        failure_reason = "no_dominant_global_support"

    info = _diagnostics(
        solver="epa_sim3_v2",
        pos_ref=p_ref_full,
        quat_ref=q_ref_full,
        pos_est=p_est_full,
        quat_est=q_est_full,
        scale=applied_scale,
        r_fit=applied_r,
        t_fit=applied_t,
        extra={
            "sim3_version": "v2",
            "sim3_anchor_status": "ok",
            "sim3_consensus_status": consensus_status,
            "sim3_confidence": confidence,
            "sim3_reliable": is_reliable,
            "sim3_transform_applied": bool(is_reliable),
            "sim3_failure_reason": failure_reason,
            "sim3_candidate_count": int(len(windows)),
            "sim3_candidate_fit_count": int(len(candidates)),
            "sim3_candidate_reliable_count": int(len(reliable_candidates)),
            "sim3_candidate_rejected_count": int(len(rejected)),
            "sim3_solver_sample_count": int(n),
            "sim3_original_pair_count": int(n_full),
            "sim3_solver_subsampled": bool(sample_idx is not None),
            "sim3_max_solver_samples": int(max_solver_samples),
            "sim3_max_support_candidates": int(max_support_candidates),
            "sim3_max_support_targets": int(max_support_targets),
            "sim3_window_stride_fraction": float(window_stride_fraction),
            "sim3_window_fractions": ",".join(str(float(x)) for x in window_fractions),
            "sim3_health_sample_ratio": float(health["healthy_sample_ratio"]),
            "sim3_health_bad_step_ratio": float(health["bad_step_ratio"]),
            "sim3_health_step_scale_ratio_median": float(health["step_scale_ratio_median"]),
            "sim3_health_est_step_median_m": float(health["est_step_median_m"]),
            "sim3_health_est_speed_p95_mps": float(health["est_speed_p95_mps"]),
            "sim3_health_est_speed_bad_ratio": float(health["est_speed_bad_ratio"]),
            "sim3_health_max_abs_est_speed_mps": float(max_abs_est_speed_mps),
            "sim3_health_global_ok": bool(global_health_ok),
            "sim3_health_min_global_sample_ratio": float(min_global_health_sample_ratio),
            "sim3_health_max_global_bad_step_ratio": float(max_global_bad_step_ratio),
            "sim3_health_max_global_speed_bad_ratio": float(max_global_speed_bad_ratio),
            "sim3_consensus_count": consensus_count,
            "sim3_consensus_ratio": consensus_ratio,
            "sim3_consensus_scale_spread_log": scale_spread,
            "sim3_consensus_rotation_spread_deg": rot_spread,
            "sim3_consensus_anchor_span_ratio": span_ratio,
            "sim3_consensus_min_span_ratio": min_span,
            "sim3_global_support_distance_m": float(selected.get("support_distance_m", 0.0)),
            "sim3_global_support_distance_ratio": support_distance_ratio,
            "sim3_global_support_time_ratio": support_time_ratio,
            "sim3_global_support_span_ratio": support_span_ratio,
            "sim3_global_support_segment_count": int(selected.get("support_segment_count", 0) or 0),
            "sim3_global_support_window_count": support_window_count,
            "sim3_global_support_rmse_median_m": float(selected.get("support_rmse_median_m", np.nan)),
            "sim3_global_support_rmse_mean_m": float(selected.get("support_rmse_mean_m", np.nan)),
            "sim3_global_support_min_distance_ratio": float(min_global_support_distance_ratio),
            "sim3_global_support_min_time_ratio": float(min_global_support_time_ratio),
            "sim3_global_support_min_window_count": int(min_global_support_window_count),
            "sim3_global_support_scale_ok": bool(selected.get("support_scale_ok", False)),
            "sim3_global_support_min_reliable_scale": float(min_reliable_scale),
            "sim3_global_support_max_reliable_scale": float(max_reliable_scale),
            "sim3_selected_scale": float(selected["scale"]),
            "sim3_anchor_start_index": int(selected["start"]),
            "sim3_anchor_end_index": int(selected["end"]),
            "sim3_anchor_samples": int(selected["count"]),
            "sim3_anchor_sample_ratio": float(selected["count"] / max(1, n)),
            "sim3_anchor_score": float(selected["score"]),
            "sim3_anchor_position_rmse_m": float(selected["pos_rmse"]),
            "sim3_anchor_orientation_rmse_deg": float(selected["rot_rmse"]),
            "sim3_anchor_rmse_path_ratio": float(selected["rmse_ratio"]),
            "sim3_anchor_ref_path_m": float(selected["ref_path"]),
            "sim3_anchor_est_path_m": float(selected["est_path"]),
            "sim3_anchor_ref_bbox_m": float(selected["ref_bbox"]),
            "sim3_anchor_est_bbox_m": float(selected["est_bbox"]),
            "sim3_anchor_ref_rank": int(selected["ref_rank"]),
            "sim3_anchor_est_rank": int(selected["est_rank"]),
            "sim3_anchor_ref_rank_ratio": float(selected["ref_rank_ratio"]),
            "sim3_anchor_est_rank_ratio": float(selected["est_rank_ratio"]),
            "sim3_anchor_orientation_span_deg": float(selected["orientation_span_deg"]),
            "sim3_anchor_failures": "; ".join(rejected[:20]),
        },
    )
    return applied_scale, applied_r, applied_t, info


def solve_epa_sim3(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    **kwargs,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    return solve_epa_sim3_v2(pos_ref, quat_ref, pos_est, quat_est, **kwargs)


def _orientation_mean_rotation(quat_ref: np.ndarray, quat_est: np.ndarray) -> R:
    rel_rot = R.from_quat(quat_ref) * R.from_quat(quat_est).inv()
    return rel_rot.mean()


def solve_orientation_consistent_sim3(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Estimate a pose-aware Sim3 transform from estimate to reference.

    Unlike position-only Umeyama alignment, this solver first estimates the
    global rotation from the orientation pairs, then solves scale and
    translation with that rotation fixed. This keeps the Sim3 transform
    consistent with the full 6DoF pose sequence instead of allowing position
    fitting alone to choose the rotation.
    """

    p_ref, q_ref, p_est, q_est = _validate_pose_pairs(pos_ref, quat_ref, pos_est, quat_est)
    r_fit_obj = _orientation_mean_rotation(q_ref, q_est)
    r_fit = r_fit_obj.as_matrix()
    scale, t_fit = _solve_scale_translation_fixed_rotation(p_ref, p_est, r_fit)
    info = _diagnostics(
        solver="epica_orientation_consistent",
        pos_ref=p_ref,
        quat_ref=q_ref,
        pos_est=p_est,
        quat_est=q_est,
        scale=scale,
        r_fit=r_fit,
        t_fit=t_fit,
    )
    return float(scale), r_fit, t_fit, info


def solve_joint_grid_sim3(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    *,
    orientation_weight: float = 0.01,
    samples: int = 41,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Search between position-optimal and orientation-optimal Sim3 rotations."""

    p_ref, q_ref, p_est, q_est = _validate_pose_pairs(pos_ref, quat_ref, pos_est, quat_est)
    _, r_pos, _ = _umeyama_transform(p_est, p_ref, with_scale=True)
    r_ori = _orientation_mean_rotation(q_ref, q_est).as_matrix()

    pos_obj = R.from_matrix(r_pos)
    ori_obj = R.from_matrix(r_ori)
    delta = ori_obj * pos_obj.inv()
    delta_vec = delta.as_rotvec()
    best: tuple[float, float, np.ndarray, np.ndarray, float, float] | None = None
    sample_count = max(3, int(samples))
    for alpha in np.linspace(0.0, 1.0, sample_count):
        r_fit = (R.from_rotvec(float(alpha) * delta_vec) * pos_obj).as_matrix()
        try:
            scale, t_fit = _solve_scale_translation_fixed_rotation(p_ref, p_est, r_fit)
        except ValueError:
            continue
        pos_new, quat_new = _apply_similarity(p_est, q_est, scale, r_fit, t_fit)
        pos_res = np.linalg.norm(pos_new - p_ref, axis=1)
        rot_res = np.degrees((R.from_quat(quat_new).inv() * R.from_quat(q_ref)).magnitude())
        pos_rmse = float(np.sqrt(np.mean(pos_res * pos_res)))
        rot_rmse = float(np.sqrt(np.mean(rot_res * rot_res)))
        objective = pos_rmse + float(orientation_weight) * rot_rmse
        if best is None or objective < best[0]:
            best = (objective, scale, r_fit, t_fit, pos_rmse, rot_rmse)
    if best is None:
        raise ValueError("EPICA joint-grid Sim3 could not find a positive-scale candidate.")

    objective, scale, r_fit, t_fit, pos_rmse, rot_rmse = best
    info = _diagnostics(
        solver="epica_joint_grid",
        pos_ref=p_ref,
        quat_ref=q_ref,
        pos_est=p_est,
        quat_est=q_est,
        scale=scale,
        r_fit=r_fit,
        t_fit=t_fit,
        extra={
            "sim3_joint_objective": float(objective),
            "sim3_joint_orientation_weight_m_per_deg": float(orientation_weight),
            "sim3_joint_grid_samples": int(sample_count),
            "sim3_joint_selected_position_rmse_m": float(pos_rmse),
            "sim3_joint_selected_orientation_rmse_deg": float(rot_rmse),
        },
    )
    return float(scale), r_fit, t_fit, info


def solve_trimmed_umeyama_sim3(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    *,
    trim_fraction: float = 0.8,
    iterations: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Robust position Sim3 using iterative trimmed Umeyama."""

    p_ref, q_ref, p_est, q_est = _validate_pose_pairs(pos_ref, quat_ref, pos_est, quat_est)
    n = int(p_ref.shape[0])
    keep_count = max(3, min(n, int(np.ceil(float(trim_fraction) * n))))
    mask = np.ones(n, dtype=bool)
    scale = 1.0
    r_fit = np.eye(3)
    t_fit = np.zeros(3)
    for _ in range(max(1, int(iterations))):
        scale, r_fit, t_fit = _umeyama_transform(p_est[mask], p_ref[mask], with_scale=True)
        pos_new = scale * (r_fit @ p_est.T).T + t_fit
        residual = np.linalg.norm(pos_new - p_ref, axis=1)
        keep = np.argsort(np.where(np.isfinite(residual), residual, np.inf))[:keep_count]
        new_mask = np.zeros(n, dtype=bool)
        new_mask[keep] = True
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    scale, r_fit, t_fit = _umeyama_transform(p_est[mask], p_ref[mask], with_scale=True)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"EPICA trimmed Sim3 estimated a non-positive scale: {scale}.")
    info = _diagnostics(
        solver="epica_trimmed_umeyama",
        pos_ref=p_ref,
        quat_ref=q_ref,
        pos_est=p_est,
        quat_est=q_est,
        scale=scale,
        r_fit=r_fit,
        t_fit=t_fit,
        extra={
            "sim3_trim_fraction": float(trim_fraction),
            "sim3_trim_inlier_count": int(np.count_nonzero(mask)),
            "sim3_trim_rejected_count": int(n - np.count_nonzero(mask)),
        },
    )
    return float(scale), r_fit, t_fit, info


def solve_epica_sim3_variant(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    method: str,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    method_l = str(method).lower()
    if method_l in {"epica_sim3", "epica_sim3_orientation", "orientation"}:
        return solve_orientation_consistent_sim3(pos_ref, quat_ref, pos_est, quat_est)
    if method_l in {"epica_sim3_joint", "joint"}:
        return solve_joint_grid_sim3(pos_ref, quat_ref, pos_est, quat_est)
    if method_l in {"epica_sim3_trimmed", "trimmed"}:
        return solve_trimmed_umeyama_sim3(pos_ref, quat_ref, pos_est, quat_est)
    if method_l in {"epa_sim3_v1", "epa_sim3_legacy"}:
        return solve_epa_sim3_v1(pos_ref, quat_ref, pos_est, quat_est)
    if method_l in {"epa_sim3", "epa_sim3_v2", "robust_anchor", "robust_anchor_sim3"}:
        return solve_epa_sim3(pos_ref, quat_ref, pos_est, quat_est)
    raise ValueError(f"Unsupported EPICA Sim3 variant: {method}")


def solve_anchor_sim3(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    *,
    anchor_fraction: float = 0.2,
    min_anchor_samples: int = 30,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, object]]:
    """Estimate Sim3 on the beginning anchor segment only.

    The returned transform is intended to be applied to the full trajectory,
    so later drift cannot be pulled back by a global full-trajectory fit.
    """

    p_ref, q_ref, p_est, q_est = _validate_pose_pairs(pos_ref, quat_ref, pos_est, quat_est)
    n = int(p_ref.shape[0])
    n_anchor = max(int(min_anchor_samples), int(np.ceil(float(anchor_fraction) * n)))
    n_anchor = min(n, max(3, n_anchor))

    failures: list[str] = []
    for method in ("epica_sim3_joint", "epica_sim3_trimmed"):
        try:
            scale, r_fit, t_fit, anchor_info = solve_epica_sim3_variant(
                pos_ref=p_ref[:n_anchor],
                quat_ref=q_ref[:n_anchor],
                pos_est=p_est[:n_anchor],
                quat_est=q_est[:n_anchor],
                method=method,
            )
            info = _diagnostics(
                solver="epica_anchor_sim3",
                pos_ref=p_ref,
                quat_ref=q_ref,
                pos_est=p_est,
                quat_est=q_est,
                scale=scale,
                r_fit=r_fit,
                t_fit=t_fit,
                extra={
                    "sim3_anchor_solver": str(anchor_info.get("sim3_solver", method)),
                    "sim3_anchor_fraction": float(anchor_fraction),
                    "sim3_anchor_samples": int(n_anchor),
                    "sim3_anchor_sample_ratio": float(n_anchor / max(1, n)),
                    "sim3_anchor_failures": "; ".join(failures),
                },
            )
            return float(scale), r_fit, t_fit, info
        except Exception as exc:
            failures.append(f"{method}: {exc}")
    raise ValueError("EPICA anchor Sim3 failed on anchor segment: " + "; ".join(failures))


def apply_orientation_consistent_sim3(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scale, r_fit, t_fit, info = solve_orientation_consistent_sim3(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
    )
    pos_new, quat_new = _apply_similarity(pos_est, quat_est, scale, r_fit, t_fit)
    info["align_scale"] = float(scale)
    return pos_new, quat_new, info
