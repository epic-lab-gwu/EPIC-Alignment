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


