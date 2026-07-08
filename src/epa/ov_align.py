from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.math_utils import normalize_quat_array


def associate_est_gt(
    t_est: np.ndarray,
    p_est: np.ndarray,
    q_est: np.ndarray,
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    q_gt: np.ndarray,
    max_diff: float,
    offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i_est: list[int] = []
    i_gt: list[int] = []
    gt_ptr = 0

    for i, te in enumerate(np.asarray(t_est, dtype=float).reshape(-1)):
        tgt = te + float(offset)
        best_diff = float(max_diff)
        best_gt = -1

        while gt_ptr < int(t_gt.size) and t_gt[gt_ptr] < tgt and abs(float(t_gt[gt_ptr] - tgt)) > float(max_diff):
            gt_ptr += 1

        while gt_ptr < int(t_gt.size) and abs(float(t_gt[gt_ptr] - tgt)) <= float(max_diff):
            cur = abs(float(t_gt[gt_ptr] - tgt))
            if cur >= best_diff:
                break
            best_diff = cur
            best_gt = int(gt_ptr)
            gt_ptr += 1

        if best_gt != -1:
            i_est.append(int(i))
            i_gt.append(best_gt)

    if len(i_est) < 3:
        raise ValueError(
            "Unable to associate enough timestamps between estimate and ground truth. "
            f"matches={len(i_est)}, max_diff={max_diff}, offset={offset}."
        )

    ie = np.asarray(i_est, dtype=int)
    ig = np.asarray(i_gt, dtype=int)
    t_match = np.asarray(t_gt, dtype=float)[ig]
    return (
        t_match,
        np.asarray(p_est, dtype=float)[ie],
        np.asarray(q_est, dtype=float)[ie],
        t_match,
        np.asarray(p_gt, dtype=float)[ig],
        np.asarray(q_gt, dtype=float)[ig],
        ie,
    )


def rot_z(theta: float) -> np.ndarray:
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def best_yaw(c_mat: np.ndarray) -> float:
    return float(math.atan2(float(c_mat[0, 1] - c_mat[1, 0]), float(c_mat[0, 0] + c_mat[1, 1])))


def umeyama(
    data_xyz: np.ndarray,
    model_xyz: np.ndarray,
    known_scale: bool,
    yaw_only: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    data = np.asarray(data_xyz, dtype=float)
    model = np.asarray(model_xyz, dtype=float)
    if data.shape != model.shape or data.shape[0] < 2:
        raise ValueError("Alignment requires paired points with count >= 2.")

    mu_m = np.mean(model, axis=0)
    mu_d = np.mean(data, axis=0)
    m0 = model - mu_m
    d0 = data - mu_d

    n = float(data.shape[0])
    c_mat = (m0.T @ d0) / n
    sigma2 = float(np.mean(np.sum(d0**2, axis=1)))

    u, svals, vt = np.linalg.svd(c_mat)
    s_mat = np.eye(3, dtype=float)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0

    r_fit = rot_z(best_yaw(n * c_mat.T)) if yaw_only else u @ s_mat @ vt

    if known_scale:
        scale = 1.0
    else:
        if sigma2 < 1e-12:
            raise ValueError("Degenerate variance for Sim3 alignment.")
        scale = float(np.trace(np.diag(svals) @ s_mat) / sigma2)

    t_fit = mu_m - scale * (r_fit @ mu_d)
    return scale, r_fit, t_fit


def apply_similarity(
    p_est: np.ndarray,
    q_est: np.ndarray,
    scale: float,
    r_fit: np.ndarray,
    t_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_new = float(scale) * (np.asarray(r_fit, dtype=float) @ np.asarray(p_est, dtype=float).T).T + np.asarray(
        t_fit,
        dtype=float,
    )
    q_new = normalize_quat_array(
        (R.from_matrix(np.asarray(r_fit, dtype=float)) * R.from_quat(np.asarray(q_est, dtype=float))).as_quat()
    )
    return p_new, q_new
