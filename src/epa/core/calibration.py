import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


class InsufficientRotationExcitationError(ValueError):
    """Valid pose input does not provide enough usable calibration pairs."""


def _rotation_observability(vA, vB, rotation, weights, *, min_information_ratio=0.03):
    """Check directional excitation and curvature of the matched rotation fit.

    For a scatter matrix S, trace(S) I - S is rotational information. Unlike
    requiring rank-three scatter/cross-covariance, this correctly permits two
    nonparallel rotation axes. Matched-fit curvature also checks whether the
    two trajectories agree in the directions that appear excited separately.
    Ratios are conditioning heuristics, not statistical confidence bounds.
    """
    threshold = float(min_information_ratio)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("Rotation information ratio must be in (0, 1].")
    a, b = np.asarray(vA, dtype=float), np.asarray(vB, dtype=float)
    w = np.asarray(weights, dtype=float)
    scatter_a = (a * w[:, None]).T @ a
    scatter_b = (b * w[:, None]).T @ b
    cross = np.asarray(rotation, dtype=float) @ ((b * w[:, None]).T @ a)
    matched = 0.5 * (cross + cross.T)
    ratios = {}
    for name, scatter in (
        ("reference", scatter_a), ("estimate", scatter_b), ("matched", matched)
    ):
        information = np.trace(scatter) * np.eye(3) - scatter
        eigenvalues = np.linalg.eigvalsh(information)
        ratio = (
            float(max(0.0, eigenvalues[0]) / eigenvalues[-1])
            if np.all(np.isfinite(eigenvalues)) and eigenvalues[-1] > 1e-15
            else 0.0
        )
        ratios[f"{name}_information_ratio"] = ratio
    weakest = min(ratios.values())
    return {
        **ratios,
        "information_ratio": weakest,
        "min_information_ratio": threshold,
        "observable": bool(weakest >= threshold),
    }


def _constrain_weak_rotation(vA, vB, rotation, weights, observability):
    """Refit log(X) in the supported reference-frame information subspace.

    Freeze the original inliers, weights and local information directions.
    Zero weak components are an identity-centered prior, not measured values.
    Keep the original fit on numerical failure; callers must gate its use.
    """
    a, b = np.asarray(vA, dtype=float), np.asarray(vB, dtype=float)
    x, w = np.asarray(rotation, dtype=float), np.asarray(weights, dtype=float)
    sa, sb = (a * w[:, None]).T @ a, (b * w[:, None]).T @ b
    cross = x @ ((b * w[:, None]).T @ a)
    matched = 0.5 * (cross + cross.T)
    matrices = {
        "reference": np.trace(sa) * np.eye(3) - sa,
        "estimate": x @ (np.trace(sb) * np.eye(3) - sb) @ x.T,
        "matched": np.trace(matched) * np.eye(3) - matched,
    }
    source = min(matrices, key=lambda key: observability[f"{key}_information_ratio"])
    eigenvalues, axes = np.linalg.eigh(matrices[source])
    weak = eigenvalues < float(observability["min_information_ratio"]) * eigenvalues[-1]
    if eigenvalues[-1] <= 1e-15:
        weak[:] = True
    basis, weak_axes = axes[:, ~weak], axes[:, weak]
    info = {
        "constraint_applied": True, "constraint_success": False,
        "constraint_dimension": int(np.count_nonzero(weak)),
        "constraint_information_source": source,
        "constraint_weak_axes": weak_axes.T.tolist(),
        "constraint_nfev": 0,
    }
    if basis.shape[1] == 0 or not np.any(weak):
        return x, info
    root_weight = np.sqrt(w)[:, None]

    def residual(theta):
        return ((R.from_rotvec(basis @ theta).apply(b) - a) * root_weight).ravel()

    starts = [np.zeros(basis.shape[1]), basis.T @ R.from_matrix(x).as_rotvec()]
    fits = [least_squares(
        residual, start, method="trf", ftol=1e-11, xtol=1e-11, gtol=1e-11,
        max_nfev=4000,
    ) for start in starts]
    valid = [fit for fit in fits if fit.success and np.all(np.isfinite(fit.fun))]
    if not valid:
        return x, info
    fit = min(valid, key=lambda value: float(value.fun @ value.fun))
    constrained = R.from_rotvec(basis @ fit.x)
    identity_residual = residual(starts[0])
    identity_cost = float(identity_residual @ identity_residual)
    cost = float(fit.fun @ fit.fun)
    success = (
        cost <= identity_cost + max(1e-12, 1e-9 * identity_cost)
        and np.linalg.norm(weak_axes.T @ constrained.as_rotvec()) <= 1e-9
    )
    info.update(
        constraint_success=bool(success), constraint_nfev=int(fit.nfev),
        constraint_identity_cost=identity_cost, constraint_final_cost=cost,
    )
    return (constrained.as_matrix() if success else x), info


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


def _safe_quaternion_array(quaternions) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(quaternions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Rotation trajectories must contain Nx4 quaternions.")
    valid = np.all(np.isfinite(values), axis=1) & (np.linalg.norm(values, axis=1) > 1e-12)
    safe = values.copy()
    safe[~valid] = np.asarray([0.0, 0.0, 0.0, 1.0])
    return safe, valid


def build_translation_system(
    pv,
    qv,
    pr,
    qr,
    Rext,
    *,
    transition_mask: np.ndarray | None = None,
    pair_indices: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
):
    pv = np.asarray(pv, dtype=float)
    pr = np.asarray(pr, dtype=float)
    Rext = np.asarray(Rext, dtype=float)
    qv_safe, _ = _safe_quaternion_array(qv)
    qr_safe, _ = _safe_quaternion_array(qr)
    Rv = R.from_quat(qv_safe).as_matrix()
    Rr = R.from_quat(qr_safe).as_matrix()
    if pv.shape[0] < 2:
        raise ValueError("No valid translation constraints for extrinsic translation solve.")

    if pair_indices is None:
        pair_ids = np.column_stack(
            [np.arange(pv.shape[0] - 1), np.arange(1, pv.shape[0])]
        )
    else:
        pair_ids = np.asarray(pair_indices, dtype=int).reshape(-1, 2)
        if pair_ids.size == 0:
            raise ValueError("No pose pairs were provided for extrinsic translation solve.")
        if np.any(pair_ids < 0) or np.any(pair_ids >= pv.shape[0]):
            raise ValueError("Translation pair indices are outside the pose trajectory.")
        if np.any(pair_ids[:, 1] <= pair_ids[:, 0]):
            raise ValueError("Translation pose pairs must have increasing indices.")

    pair_i = pair_ids[:, 0]
    pair_j = pair_ids[:, 1]
    dpv = pv[pair_j] - pv[pair_i]
    dpr = pr[pair_j] - pr[pair_i]
    Rv_i_t = np.swapaxes(Rv[pair_i], 1, 2)
    Rr_i_t = np.swapaxes(Rr[pair_i], 1, 2)
    ta = np.einsum("nij,nj->ni", Rv_i_t, dpv)
    tb = np.einsum("nij,nj->ni", Rr_i_t, dpr)
    mask = np.linalg.norm(ta, axis=1) >= 1e-3
    if transition_mask is not None:
        if pair_indices is not None:
            raise ValueError("Use pair_mask with pair_indices; transition_mask is sequential-only.")
        requested_mask = np.asarray(transition_mask, dtype=bool).reshape(-1)
        if requested_mask.size != mask.size:
            raise ValueError(
                "Translation transition mask must match the number of pose transitions."
            )
        mask &= requested_mask
    if pair_mask is not None:
        requested_pair_mask = np.asarray(pair_mask, dtype=bool).reshape(-1)
        if requested_pair_mask.size != mask.size:
            raise ValueError("Translation pair mask must match the number of pose pairs.")
        mask &= requested_pair_mask
    if not np.any(mask):
        raise ValueError("No valid translation constraints for extrinsic translation solve.")

    C = np.einsum("nij,njk->nik", Rv_i_t[mask], Rv[pair_j][mask]) - np.eye(3)
    d = np.einsum("ij,nj->ni", Rext, tb[mask]) - ta[mask]
    return C.reshape(-1, 3), d.reshape(-1), int(np.count_nonzero(mask))


def extrinsic_rotation_transition_mask(
    qv,
    qr,
    *,
    min_excitation_rad: float = 1e-3,
    angle_abs_threshold_deg: float = 0.5,
    angle_rel_threshold: float = 0.2,
    relative_angle_floor_deg: float = 1.0,
):
    """Select sequential rotations whose magnitudes are compatible."""
    qv_safe, qv_valid = _safe_quaternion_array(qv)
    qr_safe, qr_valid = _safe_quaternion_array(qr)
    Rv = R.from_quat(qv_safe)
    Rr = R.from_quat(qr_safe)
    if len(Rv) != len(Rr):
        raise ValueError("Extrinsic rotation trajectories must have the same length.")
    if len(Rv) < 2:
        raise ValueError("Extrinsic rotation calibration requires at least 2 poses.")
    vA = (Rv[:-1].inv() * Rv[1:]).as_rotvec()
    vB = (Rr[:-1].inv() * Rr[1:]).as_rotvec()
    norm_a = np.linalg.norm(vA, axis=1)
    norm_b = np.linalg.norm(vB, axis=1)
    consistent, _, _ = _rotation_angle_consistency(
        norm_a,
        norm_b,
        angle_abs_threshold_deg=angle_abs_threshold_deg,
        angle_rel_threshold=angle_rel_threshold,
        relative_angle_floor_deg=relative_angle_floor_deg,
    )
    finite = (
        qv_valid[:-1]
        & qv_valid[1:]
        & qr_valid[:-1]
        & qr_valid[1:]
        & np.all(np.isfinite(vA), axis=1)
        & np.all(np.isfinite(vB), axis=1)
    )
    excited = norm_a > max(float(min_excitation_rad), 0.0)
    mask = finite & excited & consistent
    info = {
        "transition_count": int(mask.size),
        "accepted_count": int(np.count_nonzero(mask)),
        "rejected_nonfinite_count": int(np.count_nonzero(~finite)),
        "rejected_low_excitation_count": int(np.count_nonzero(finite & ~excited)),
        "rejected_angle_count": int(np.count_nonzero(finite & excited & ~consistent)),
        "angle_abs_threshold_deg": float(angle_abs_threshold_deg),
        "angle_rel_threshold": float(angle_rel_threshold),
        "relative_angle_floor_deg": float(relative_angle_floor_deg),
    }
    return mask, vA, vB, info


def solve_extrinsic_rotation(
    qv,
    qr,
    *,
    angle_abs_threshold_deg: float = 0.5,
    angle_rel_threshold: float = 0.2,
    relative_angle_floor_deg: float = 1.0,
    return_info: bool = False,
):
    """Compatibility sequential hand-eye rotation solver."""
    mask, vA, vB, info = extrinsic_rotation_transition_mask(
        qv,
        qr,
        angle_abs_threshold_deg=angle_abs_threshold_deg,
        angle_rel_threshold=angle_rel_threshold,
        relative_angle_floor_deg=relative_angle_floor_deg,
    )
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError(
            "Not enough consistent rotational excitation to estimate extrinsic rotation "
            f"({int(np.count_nonzero(mask))} accepted of {int(mask.size)} transitions)."
        )
    H = vB[mask].T @ vA[mask]
    U, _, Vt = np.linalg.svd(H)
    X = Vt.T @ U.T
    if np.linalg.det(X) < 0:
        Vt[2, :] *= -1
        X = Vt.T @ U.T
    return (X, mask, info) if return_info else X


def _rotation_angle_consistency(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
    *,
    angle_abs_threshold_deg: float,
    angle_rel_threshold: float,
    relative_angle_floor_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle_scale = np.maximum(norm_a, norm_b)
    absolute_difference = np.abs(norm_a - norm_b)
    relative_floor_rad = np.deg2rad(max(float(relative_angle_floor_deg), 0.0))
    relative_eligible = angle_scale >= relative_floor_rad
    relative_difference = np.full_like(absolute_difference, np.inf)
    np.divide(
        absolute_difference,
        angle_scale,
        out=relative_difference,
        where=relative_eligible & (angle_scale > 0.0),
    )
    consistent = (
        absolute_difference <= np.deg2rad(max(float(angle_abs_threshold_deg), 0.0))
    ) | (
        relative_eligible
        & (relative_difference <= max(float(angle_rel_threshold), 0.0))
    )
    return consistent, absolute_difference, relative_difference


def _calibration_segments(
    qv,
    qr,
    *,
    timestamps_s: np.ndarray | None,
    source_timestamps_s: np.ndarray | None,
    gap_factor: float,
    gap_floor_s: float,
    angle_abs_threshold_deg: float,
    angle_rel_threshold: float,
    relative_angle_floor_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n = int(np.asarray(qv).shape[0])
    qv_safe, qv_valid = _safe_quaternion_array(qv)
    qr_safe, qr_valid = _safe_quaternion_array(qr)
    if qv_safe.shape != qr_safe.shape:
        raise ValueError("Extrinsic rotation trajectories must have the same length.")
    sample_valid = qv_valid & qr_valid
    interpolated_gap_invalid = np.zeros(n, dtype=bool)
    segment_ids = np.zeros(n, dtype=int)
    if timestamps_s is None:
        timestamps = np.arange(n, dtype=float)
        gap_threshold_s = float("inf")
    else:
        timestamps = np.asarray(timestamps_s, dtype=float).reshape(-1)
        if timestamps.size != n:
            raise ValueError("Calibration timestamps must match the pose count.")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("Calibration timestamps must be finite and strictly increasing.")
        source_timestamps = (
            timestamps
            if source_timestamps_s is None
            else np.asarray(source_timestamps_s, dtype=float).reshape(-1)
        )
        if source_timestamps.size < 2:
            raise ValueError("Source calibration timestamps require at least 2 samples.")
        if not np.all(np.isfinite(source_timestamps)) or np.any(
            np.diff(source_timestamps) <= 0.0
        ):
            raise ValueError(
                "Source calibration timestamps must be finite and strictly increasing."
            )
        source_dt = np.diff(source_timestamps)
        positive_dt = source_dt[source_dt > 0.0]
        median_dt = float(np.median(positive_dt)) if positive_dt.size else float("nan")
        gap_threshold_s = (
            max(float(gap_floor_s), float(gap_factor) * median_dt)
            if np.isfinite(median_dt)
            else float(gap_floor_s)
        )
        for gap_idx in np.flatnonzero(source_dt > gap_threshold_s):
            left = float(source_timestamps[gap_idx])
            right = float(source_timestamps[gap_idx + 1])
            inside_gap = (timestamps > left) & (timestamps < right)
            interpolated_gap_invalid |= inside_gap
            sample_valid &= ~inside_gap
            segment_ids[timestamps >= right] += 1

    Rv = R.from_quat(qv_safe)
    Rr = R.from_quat(qr_safe)
    adjacent_a = (Rv[:-1].inv() * Rv[1:]).magnitude()
    adjacent_b = (Rr[:-1].inv() * Rr[1:]).magnitude()
    adjacent_consistent, _, _ = _rotation_angle_consistency(
        adjacent_a,
        adjacent_b,
        angle_abs_threshold_deg=angle_abs_threshold_deg,
        angle_rel_threshold=angle_rel_threshold,
        relative_angle_floor_deg=relative_angle_floor_deg,
    )
    reset_boundaries = (
        ~qv_valid[:-1]
        | ~qv_valid[1:]
        | ~qr_valid[:-1]
        | ~qr_valid[1:]
        | ~np.isfinite(adjacent_a)
        | ~np.isfinite(adjacent_b)
        | ~adjacent_consistent
    )
    if timestamps_s is not None:
        reset_boundaries |= np.diff(timestamps) > gap_threshold_s
    for boundary in np.flatnonzero(reset_boundaries):
        segment_ids[boundary + 1 :] += 1

    return sample_valid, segment_ids, {
        "gap_threshold_s": float(gap_threshold_s),
        "invalid_rotation_sample_count": int(np.count_nonzero(~(qv_valid & qr_valid))),
        "invalid_interpolated_sample_count": int(
            np.count_nonzero(interpolated_gap_invalid)
        ),
        "reset_boundary_count": int(np.count_nonzero(reset_boundaries)),
        "segment_count": int(np.unique(segment_ids[sample_valid]).size),
    }


def _multibaseline_pair_indices(
    timestamps_s: np.ndarray,
    sample_valid: np.ndarray,
    segment_ids: np.ndarray,
    *,
    max_pair_dt_s: float,
    max_pairs: int,
) -> np.ndarray:
    n = int(timestamps_s.size)
    pairs: list[np.ndarray] = []
    lag = 1
    while lag < n:
        pair_i = np.arange(0, n - lag, dtype=int)
        pair_j = pair_i + lag
        keep = (
            sample_valid[pair_i]
            & sample_valid[pair_j]
            & (segment_ids[pair_i] == segment_ids[pair_j])
        )
        if np.isfinite(float(max_pair_dt_s)) and float(max_pair_dt_s) > 0.0:
            pair_dt = timestamps_s[pair_j] - timestamps_s[pair_i]
            keep &= (pair_dt > 0.0) & (pair_dt <= float(max_pair_dt_s))
        if np.any(keep):
            pairs.append(np.column_stack([pair_i[keep], pair_j[keep]]))
        lag *= 2
    if not pairs:
        return np.empty((0, 2), dtype=int)
    pair_indices = np.vstack(pairs)
    if int(max_pairs) > 0 and pair_indices.shape[0] > int(max_pairs):
        selected = np.linspace(0, pair_indices.shape[0] - 1, int(max_pairs), dtype=int)
        pair_indices = pair_indices[selected]
    return pair_indices


def solve_extrinsic_rotation_multibaseline(
    qv,
    qr,
    *,
    timestamps_s: np.ndarray | None = None,
    source_timestamps_s: np.ndarray | None = None,
    reset_intervals=(),
    angle_abs_threshold_deg: float = 0.5,
    angle_rel_threshold: float = 0.2,
    relative_angle_floor_deg: float = 1.0,
    gap_factor: float = 20.0,
    gap_floor_s: float = 0.5,
    max_pair_dt_s: float = 2.0,
    noise_multiplier: float = 5.0,
    min_rotation_floor_deg: float = 0.1,
    influence_cap_deg: float = 15.0,
    residual_floor_deg: float = 1.0,
    robust_max_iterations: int = 5,
    max_pairs: int = 50000,
    min_information_ratio: float = 0.03,
    constrain_unobservable: bool = True,
):
    """Fit rotation and report observability; the world solver gates its use.

    Observability describes the unconstrained fit. If enabled, weak directions
    are fixed to zero in log coordinates and the supported components refitted.
    The original final pair mask/weights are frozen for this constrained solve
    and the mask remains available for translation calibration.
    """
    qv_arr = np.asarray(qv, dtype=float)
    qr_arr = np.asarray(qr, dtype=float)
    if qv_arr.shape != qr_arr.shape or qv_arr.ndim != 2 or qv_arr.shape[1] != 4:
        raise ValueError("Extrinsic rotation calibration requires paired Nx4 quaternions.")
    if qv_arr.shape[0] < 4:
        raise InsufficientRotationExcitationError(
            "Extrinsic rotation calibration requires at least 4 poses."
        )

    sample_valid, segment_ids, segment_info = _calibration_segments(
        qv_arr,
        qr_arr,
        timestamps_s=timestamps_s,
        source_timestamps_s=source_timestamps_s,
        gap_factor=gap_factor,
        gap_floor_s=gap_floor_s,
        angle_abs_threshold_deg=angle_abs_threshold_deg,
        angle_rel_threshold=angle_rel_threshold,
        relative_angle_floor_deg=relative_angle_floor_deg,
    )
    timestamps = (
        np.arange(qv_arr.shape[0], dtype=float)
        if timestamps_s is None
        else np.asarray(timestamps_s, dtype=float).reshape(-1)
    )
    from .adaptive_association import reset_crossing_mask
    reset_edges = reset_crossing_mask(timestamps, reset_intervals)
    segment_ids = np.r_[0, np.cumsum((np.diff(segment_ids) != 0) | reset_edges)]
    segment_info["independent_reset_boundary_count"] = int(np.count_nonzero(reset_edges))
    pair_indices = _multibaseline_pair_indices(
        timestamps,
        sample_valid,
        segment_ids,
        max_pair_dt_s=float("inf") if timestamps_s is None else max_pair_dt_s,
        max_pairs=max_pairs,
    )
    if pair_indices.shape[0] < 3:
        raise InsufficientRotationExcitationError(
            "Not enough within-segment pose pairs for extrinsic rotation calibration."
        )

    qv_safe, _ = _safe_quaternion_array(qv_arr)
    qr_safe, _ = _safe_quaternion_array(qr_arr)
    Rv = R.from_quat(qv_safe)
    Rr = R.from_quat(qr_safe)
    pair_i = pair_indices[:, 0]
    pair_j = pair_indices[:, 1]
    rel_a = Rv[pair_i].inv() * Rv[pair_j]
    rel_b = Rr[pair_i].inv() * Rr[pair_j]
    vA = rel_a.as_rotvec()
    vB = rel_b.as_rotvec()
    norm_a = np.linalg.norm(vA, axis=1)
    norm_b = np.linalg.norm(vB, axis=1)

    adjacent = pair_j == pair_i + 1
    adjacent_difference = np.abs(norm_a[adjacent] - norm_b[adjacent])
    noise_rad = robust_scale(
        adjacent_difference[np.isfinite(adjacent_difference)],
        floor=np.deg2rad(0.005),
    )
    min_rotation_rad = max(
        np.deg2rad(max(float(min_rotation_floor_deg), 0.0)),
        max(float(noise_multiplier), 0.0) * noise_rad,
    )
    angle_consistent, _, _ = _rotation_angle_consistency(
        norm_a,
        norm_b,
        angle_abs_threshold_deg=angle_abs_threshold_deg,
        angle_rel_threshold=angle_rel_threshold,
        relative_angle_floor_deg=relative_angle_floor_deg,
    )
    finite = np.all(np.isfinite(vA), axis=1) & np.all(np.isfinite(vB), axis=1)
    excited = np.minimum(norm_a, norm_b) >= min_rotation_rad
    candidate_mask = finite & excited & angle_consistent
    if int(np.count_nonzero(candidate_mask)) < 3:
        raise InsufficientRotationExcitationError(
            "Not enough consistent multi-baseline rotational excitation to estimate extrinsics."
        )

    angle_scale = np.maximum(norm_a, norm_b)
    influence_cap_rad = np.deg2rad(max(float(influence_cap_deg), 1e-6))
    influence_scale = np.minimum(1.0, influence_cap_rad / np.maximum(angle_scale, 1e-12))

    def fit_rotation(mask: np.ndarray, residual_weights: np.ndarray | None = None) -> np.ndarray:
        weights = influence_scale[mask] ** 2
        if residual_weights is not None:
            weights *= residual_weights[mask]
        root_weight = np.sqrt(weights)
        H = (vB[mask] * root_weight[:, None]).T @ (vA[mask] * root_weight[:, None])
        U, _, Vt = np.linalg.svd(H)
        X = Vt.T @ U.T
        if np.linalg.det(X) < 0.0:
            Vt[2, :] *= -1.0
            X = Vt.T @ U.T
        return X

    inlier_mask = candidate_mask.copy()
    X = fit_rotation(inlier_mask)
    residual_threshold_rad = np.deg2rad(max(float(residual_floor_deg), 0.0))
    iterations = 0
    for iteration in range(max(1, int(robust_max_iterations))):
        rotation_x = R.from_matrix(X)
        predicted_a = rotation_x * rel_b * rotation_x.inv()
        residuals = (rel_a.inv() * predicted_a).magnitude()
        candidate_residuals = residuals[candidate_mask]
        scale = robust_scale(candidate_residuals, floor=np.deg2rad(0.01))
        residual_threshold_rad = max(
            np.deg2rad(max(float(residual_floor_deg), 0.0)),
            float(np.median(candidate_residuals)) + 4.0 * scale,
        )
        updated_inliers = candidate_mask & (residuals <= residual_threshold_rad)
        if int(np.count_nonzero(updated_inliers)) < 3:
            break
        updated_X = fit_rotation(
            updated_inliers,
            huber_weights(residuals, residual_threshold_rad),
        )
        iterations = iteration + 1
        converged = np.array_equal(updated_inliers, inlier_mask) and (
            R.from_matrix(updated_X @ X.T).magnitude() <= 1e-10
        )
        X = updated_X
        inlier_mask = updated_inliers
        if converged:
            break

    rotation_x = R.from_matrix(X)
    final_residuals = (rel_a.inv() * rotation_x * rel_b * rotation_x.inv()).magnitude()
    final_weights = influence_scale[inlier_mask] ** 2 * huber_weights(
        final_residuals[inlier_mask], residual_threshold_rad
    )
    observability = _rotation_observability(
        vA[inlier_mask], vB[inlier_mask], X, final_weights,
        min_information_ratio=min_information_ratio,
    )
    unconstrained_angle_deg = float(np.degrees(R.from_matrix(X).magnitude()))
    constraint_info = {
        "constraint_applied": False, "constraint_success": False,
        "constraint_dimension": 0, "constraint_nfev": 0,
        "constraint_information_source": "", "constraint_weak_axes": [],
    }
    if constrain_unobservable and not observability["observable"]:
        X, constraint_info = _constrain_weak_rotation(
            vA[inlier_mask], vB[inlier_mask], X, final_weights, observability,
        )
    info = {
        **segment_info,
        **observability,
        **constraint_info,
        "unconstrained_angle_deg": unconstrained_angle_deg,
        "constrained_angle_deg": float(np.degrees(R.from_matrix(X).magnitude())),
        "pair_count": int(pair_indices.shape[0]),
        "transition_count": int(pair_indices.shape[0]),
        "candidate_pair_count": int(np.count_nonzero(candidate_mask)),
        "accepted_count": int(np.count_nonzero(inlier_mask)),
        "rejected_nonfinite_count": int(np.count_nonzero(~finite)),
        "rejected_low_excitation_count": int(np.count_nonzero(finite & ~excited)),
        "rejected_angle_count": int(np.count_nonzero(finite & excited & ~angle_consistent)),
        "rejected_residual_count": int(np.count_nonzero(candidate_mask & ~inlier_mask)),
        "orientation_noise_deg": float(np.degrees(noise_rad)),
        "min_rotation_deg": float(np.degrees(min_rotation_rad)),
        "influence_cap_deg": float(influence_cap_deg),
        "residual_threshold_deg": float(np.degrees(residual_threshold_rad)),
        "robust_iterations": int(iterations),
    }
    return X, pair_indices, inlier_mask, info


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
    transition_mask: np.ndarray | None = None,
    pair_indices: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
    return_info: bool = False,
):
    C, d, constraint_count = build_translation_system(
        pv,
        qv,
        pr,
        qr,
        Rext,
        transition_mask=transition_mask,
        pair_indices=pair_indices,
        pair_mask=pair_mask,
    )
    kernel = str(robust_kernel or "none").strip().lower()
    if kernel not in {"standard", "none", "huber", "cauchy"}:
        raise ValueError(f"Unsupported robust kernel: {robust_kernel}")

    estimate, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
    residual_vec = (C @ estimate - d).reshape(-1, 3)
    residuals = np.linalg.norm(residual_vec, axis=1)
    standard_rmse = float(np.sqrt(np.mean(residuals * residuals)))
    info = {
        "kernel": kernel,
        "constraint_count": int(constraint_count),
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


def _rotation_median_inliers(relative: R, *, min_threshold_deg: float = 1.0, return_info=False):
    """Reject angular outliers around a median-consensus alignment rotation.

    Use a bounded, deterministic medoid search to avoid starting from a mean
    already pulled towards orientation jumps. Quaternions are compared through
    absolute dot products, so their arbitrary signs do not affect selection.
    Position residuals and genuine motion shared by both trajectories do not
    enter this check.
    """
    n = len(relative)
    mask = np.ones(n, dtype=bool)
    info = {"inlier_count": n, "rejected_count": 0, "threshold_deg": None,
            "median_deg": None, "mad_deg": None, "iterations": 0}
    if n < 3:
        return (mask, info) if return_info else mask
    quats = relative.as_quat()
    candidate_ids = np.linspace(0, n - 1, min(n, 64), dtype=int)
    sample_ids = np.linspace(0, n - 1, min(n, 2048), dtype=int)
    dots = np.clip(np.abs(quats[candidate_ids] @ quats[sample_ids].T), 0.0, 1.0)
    scores = np.median(2.0 * np.arccos(dots), axis=1)
    center = relative[candidate_ids[int(np.argmin(scores))]]
    for _ in range(3):
        errors = (center.inv() * relative).magnitude()
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        threshold = max(np.radians(min_threshold_deg), median + 3.0 * 1.4826 * mad)
        updated = errors <= threshold
        # An ambiguous/minuscule consensus must not produce an underdetermined fit.
        if np.count_nonzero(updated) < max(2, (n + 1) // 2):
            break
        unchanged = np.array_equal(updated, mask)
        mask = updated
        info.update(inlier_count=int(mask.sum()), rejected_count=int(n - mask.sum()),
                    threshold_deg=float(np.degrees(threshold)),
                    median_deg=float(np.degrees(median)), mad_deg=float(np.degrees(mad)),
                    iterations=info["iterations"] + 1)
        center = relative[mask].mean()
        if unchanged:
            break
    return (mask, info) if return_info else mask


def solve_rotation_first_alignment(P, Q, qP, qQ, weights=None):
    """Solve a rigid transform with rotation fixed by orientation pairs.

    The returned transform maps source poses ``(P, qP)`` into reference poses
    ``(Q, qQ)``.  Unlike position-only SE(3), position residuals cannot rotate
    the trajectory; they determine translation only after the orientation mean.
    A median/MAD angular check removes orientation outliers from that mean.
    Translation still uses all supplied position pairs, and no evaluation poses
    are removed by this fitting-only filter.
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
    rotation_inliers = _rotation_median_inliers(relative)
    rotation_weights = None if mean_weights is None else mean_weights[rotation_inliers]
    Rw = relative[rotation_inliers].mean(weights=rotation_weights).as_matrix()
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
