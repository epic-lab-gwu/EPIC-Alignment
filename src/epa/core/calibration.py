import numpy as np
from scipy.spatial.transform import Rotation as R


def robust_scale(residuals, *, floor: float = 1e-6) -> float:
    """Return a MAD-based residual scale for robust weighting."""
    values = np.asarray(residuals, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(floor)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= float(floor):
        centered = np.abs(values - median)
        nonzero = centered[centered > float(floor)]
        if nonzero.size:
            scale = float(np.median(nonzero)) / 0.6745
    return float(max(scale, float(floor)))


def huber_weights(residuals, delta: float) -> np.ndarray:
    """Return Huber IRLS weights for non-negative residual magnitudes."""
    values = np.asarray(residuals, dtype=float)
    delta = max(float(delta), 1e-12)
    weights = np.ones_like(values, dtype=float)
    large = np.isfinite(values) & (values > delta)
    weights[large] = delta / values[large]
    weights[~np.isfinite(values)] = 0.0
    return weights


def cauchy_weights(residuals, delta: float) -> np.ndarray:
    """Return smooth Cauchy M-estimator weights for residual magnitudes."""
    values = np.asarray(residuals, dtype=float)
    delta = max(float(delta), 1e-12)
    scaled = values / delta
    weights = 1.0 / (1.0 + scaled * scaled)
    weights[~np.isfinite(values)] = 0.0
    return weights


def robust_weights(residuals, delta: float, kernel: str) -> np.ndarray:
    """Dispatch the supported soft robust weighting functions."""
    mode = str(kernel or "huber").strip().lower()
    if mode == "huber":
        return huber_weights(residuals, delta)
    if mode == "cauchy":
        return cauchy_weights(residuals, delta)
    raise ValueError(f"Unsupported soft robust kernel: {kernel}")


def build_translation_system(pv, qv, pr, qr, Rext):
    pv = np.asarray(pv, dtype=float)
    pr = np.asarray(pr, dtype=float)
    Rext = np.asarray(Rext, dtype=float)
    Rv = R.from_quat(qv).as_matrix()
    Rr = R.from_quat(qr).as_matrix()
    if pv.shape[0] < 2:
        raise ValueError("No valid translation constraints for extrinsic translation solve.")

    dpv = pv[1:] - pv[:-1]
    dpr = pr[1:] - pr[:-1]
    Rv_i_t = np.swapaxes(Rv[:-1], 1, 2)
    Rr_i_t = np.swapaxes(Rr[:-1], 1, 2)
    ta = np.einsum("nij,nj->ni", Rv_i_t, dpv)
    tb = np.einsum("nij,nj->ni", Rr_i_t, dpr)
    mask = np.linalg.norm(ta, axis=1) >= 1e-3
    if not np.any(mask):
        raise ValueError("No valid translation constraints for extrinsic translation solve.")

    C = np.einsum("nij,njk->nik", Rv_i_t[mask], Rv[1:][mask]) - np.eye(3)
    d = np.einsum("ij,nj->ni", Rext, tb[mask]) - ta[mask]
    return C.reshape(-1, 3), d.reshape(-1), int(np.count_nonzero(mask))


def solve_extrinsic_rotation(qv, qr):
    Rv = R.from_quat(qv)
    Rr = R.from_quat(qr)
    vA = (Rv[:-1].inv() * Rv[1:]).as_rotvec()
    vB = (Rr[:-1].inv() * Rr[1:]).as_rotvec()
    mask = np.linalg.norm(vA, axis=1) > 1e-3
    if np.sum(mask) < 3:
        raise ValueError("Not enough rotational excitation to estimate extrinsic rotation.")
    H = vB[mask].T @ vA[mask]
    U, _, Vt = np.linalg.svd(H)
    X = Vt.T @ U.T
    if np.linalg.det(X) < 0:
        Vt[2, :] *= -1
        X = Vt.T @ U.T
    return X


def solve_extrinsic_translation(
    pv,
    qv,
    pr,
    qr,
    Rext,
    *,
    robust_kernel: str = "none",
    robust_delta_m: float | None = None,
    robust_max_iterations: int = 3,
    return_info: bool = False,
):
    C, d, _ = build_translation_system(pv, qv, pr, qr, Rext)
    kernel = str(robust_kernel or "none").strip().lower()
    if kernel not in {"standard", "none", "huber", "cauchy"}:
        raise ValueError(f"Unsupported robust kernel: {robust_kernel}")

    estimate, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
    residual_vec = (C @ estimate - d).reshape(-1, 3)
    residuals = np.linalg.norm(residual_vec, axis=1)
    standard_rmse = float(np.sqrt(np.mean(residuals * residuals)))
    info = {
        "kernel": kernel,
        "iterations": 0,
        "delta_m": float("nan"),
        "scale_m": float("nan"),
        "downweighted_count": 0,
        "effective_weight_fraction": 1.0,
        "standard_rmse_m": standard_rmse,
        "robust_rmse_m": standard_rmse,
    }
    if kernel in {"standard", "none"}:
        return (estimate, info) if return_info else estimate

    for iteration in range(max(1, int(robust_max_iterations))):
        scale = robust_scale(residuals, floor=1e-6)
        delta = (
            float(robust_delta_m)
            if robust_delta_m is not None
            else 1.345 * max(scale, 1e-3)
        )
        weights = robust_weights(residuals, delta, kernel)
        row_scale = np.repeat(np.sqrt(weights), 3)
        weighted_C = C * row_scale[:, None]
        weighted_d = d * row_scale
        updated, _, _, _ = np.linalg.lstsq(weighted_C, weighted_d, rcond=None)
        updated_residuals = np.linalg.norm((C @ updated - d).reshape(-1, 3), axis=1)
        info.update(
            {
                "iterations": int(iteration + 1),
                "delta_m": float(delta),
                "scale_m": float(scale),
                "downweighted_count": int(np.count_nonzero(weights < 0.999999)),
                "effective_weight_fraction": float(np.mean(weights)),
                "robust_rmse_m": float(np.sqrt(np.mean(updated_residuals**2))),
            }
        )
        if np.linalg.norm(updated - estimate) <= 1e-9 * (1.0 + np.linalg.norm(estimate)):
            estimate = updated
            residuals = updated_residuals
            break
        estimate = updated
        residuals = updated_residuals
    return (estimate, info) if return_info else estimate


def solve_world_alignment(P, Q):
    cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    Rw = Vt.T @ U.T
    if np.linalg.det(Rw) < 0:
        Vt[2, :] *= -1
        Rw = Vt.T @ U.T
    return Rw, cQ - Rw @ cP


def solve_rotation_first_alignment(P, Q, qP, qQ, weights=None):
    """Solve a rigid transform with rotation fixed by orientation pairs.

    The returned transform maps source poses ``(P, qP)`` into reference poses
    ``(Q, qQ)``.  Unlike position-only SE(3), position residuals cannot rotate
    the trajectory; they determine translation only after the orientation mean.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    qP = np.asarray(qP, dtype=float)
    qQ = np.asarray(qQ, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Rotation-first alignment requires paired Nx3 positions.")
    if qP.shape != qQ.shape or qP.shape != (P.shape[0], 4):
        raise ValueError("Rotation-first alignment requires paired Nx4 quaternions.")
    if P.shape[0] < 2:
        raise ValueError("Rotation-first alignment requires at least 2 pose pairs.")

    mean_weights = None
    if weights is not None:
        mean_weights = np.asarray(weights, dtype=float).reshape(-1)
        if mean_weights.size != P.shape[0]:
            raise ValueError("Rotation-first weights must match the pose-pair count.")
        valid = np.isfinite(mean_weights) & (mean_weights > 0.0)
        if int(np.count_nonzero(valid)) < 2:
            raise ValueError("Rotation-first alignment requires at least 2 positive weights.")
        P = P[valid]
        Q = Q[valid]
        qP = qP[valid]
        qQ = qQ[valid]
        mean_weights = mean_weights[valid]

    relative = R.from_quat(qQ) * R.from_quat(qP).inv()
    Rw = relative.mean(weights=mean_weights).as_matrix()
    offsets = Q - (Rw @ P.T).T
    if mean_weights is None:
        tw = np.mean(offsets, axis=0)
    else:
        tw = np.sum(offsets * mean_weights[:, None], axis=0) / float(
            np.sum(mean_weights)
        )
    return np.asarray(Rw, dtype=float), np.asarray(tw, dtype=float)


def solve_world_alignment_weighted(P, Q, weights):
    """Solve a weighted rigid alignment with the same convention as solve_world_alignment."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Weighted world alignment requires paired Nx3 arrays.")
    if weights.size != P.shape[0]:
        raise ValueError("Weighted world alignment weights must match the pair count.")
    valid = np.isfinite(weights) & (weights > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError("Weighted world alignment requires at least 2 positive-weight pairs.")
    P = P[valid]
    Q = Q[valid]
    weights = weights[valid]
    weight_sum = float(np.sum(weights))
    cP = np.sum(P * weights[:, None], axis=0) / weight_sum
    cQ = np.sum(Q * weights[:, None], axis=0) / weight_sum
    H = ((P - cP) * weights[:, None]).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    Rw = Vt.T @ U.T
    if np.linalg.det(Rw) < 0:
        Vt[2, :] *= -1
        Rw = Vt.T @ U.T
    return Rw, cQ - Rw @ cP


def solve_world_alignment_robust(
    P,
    Q,
    *,
    robust_kernel: str = "none",
    robust_delta_m: float | None = None,
    robust_max_iterations: int = 3,
    return_info: bool = False,
):
    """Solve rigid alignment with optional Huber IRLS weighting."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    kernel = str(robust_kernel or "none").strip().lower()
    if kernel not in {"standard", "none", "huber", "cauchy"}:
        raise ValueError(f"Unsupported robust kernel: {robust_kernel}")
    Rw, tw = solve_world_alignment(P, Q)
    pred = (Rw @ P.T).T + tw
    residuals = np.linalg.norm(pred - Q, axis=1)
    standard_rmse = float(np.sqrt(np.mean(residuals**2)))
    info = {
        "kernel": kernel,
        "iterations": 0,
        "delta_m": float("nan"),
        "scale_m": float("nan"),
        "downweighted_count": 0,
        "effective_weight_fraction": 1.0,
        "standard_rmse_m": standard_rmse,
        "robust_rmse_m": standard_rmse,
    }
    if kernel in {"standard", "none"}:
        return (Rw, tw, info) if return_info else (Rw, tw)

    for iteration in range(max(1, int(robust_max_iterations))):
        scale = robust_scale(residuals, floor=1e-6)
        delta = (
            float(robust_delta_m)
            if robust_delta_m is not None
            else 1.345 * max(scale, 1e-3)
        )
        weights = robust_weights(residuals, delta, kernel)
        updated_R, updated_t = solve_world_alignment_weighted(P, Q, weights)
        updated_pred = (updated_R @ P.T).T + updated_t
        updated_residuals = np.linalg.norm(updated_pred - Q, axis=1)
        info.update(
            {
                "iterations": int(iteration + 1),
                "delta_m": float(delta),
                "scale_m": float(scale),
                "downweighted_count": int(np.count_nonzero(weights < 0.999999)),
                "effective_weight_fraction": float(np.mean(weights)),
                "robust_rmse_m": float(np.sqrt(np.mean(updated_residuals**2))),
            }
        )
        rotation_delta = R.from_matrix(updated_R @ Rw.T).magnitude()
        translation_delta = np.linalg.norm(updated_t - tw)
        Rw, tw = updated_R, updated_t
        residuals = updated_residuals
        if rotation_delta <= 1e-9 and translation_delta <= 1e-9 * (1.0 + np.linalg.norm(tw)):
            break
    return (Rw, tw, info) if return_info else (Rw, tw)
