import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .math_utils import normalize_quat_array


def interpolate_linear_extrapolate(t_src, values, t_query):
    """Match scipy interp1d's axis-0 linear/extrapolation arithmetic."""
    x = np.asarray(t_src, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float)
    query = np.asarray(t_query, dtype=float).reshape(-1)
    if x.size < 2 or y.shape[0] != x.size:
        raise ValueError("Linear interpolation requires at least two paired samples.")

    # Preserve scipy's legacy sorting/duplicate behavior for irregular inputs,
    # but keep its heavy interpolate module off ordinary monotonic trajectories.
    if np.any(np.diff(x) <= 0.0):
        from scipy.interpolate import interp1d

        return np.asarray(
            interp1d(x, y, axis=0, fill_value="extrapolate")(query),
            dtype=float,
        )

    right = np.searchsorted(x, query).clip(1, x.size - 1).astype(int)
    left = right - 1
    x_left = x[left]
    x_right = x[right]
    denominator = x_right - x_left
    weight_shape = (query.size,) + (1,) * (y.ndim - 1)
    weight_right = ((query - x_left) / denominator).reshape(weight_shape)
    weight_left = ((x_right - query) / denominator).reshape(weight_shape)
    return weight_right * y[right] + weight_left * y[left]


def get_angular_velocity_norm(t, quats):
    t = np.asarray(t, dtype=float).reshape(-1)
    q = np.asarray(quats, dtype=float)
    if t.size < 2 or q.shape[0] < 2:
        raise ValueError("Need at least 2 timestamped poses to compute angular velocity.")

    rot = R.from_quat(q)
    rel_rot = rot[:-1].inv() * rot[1:]
    angles = rel_rot.magnitude()
    dt = np.diff(t)

    # Filter non-increasing timestamps (duplicate or reversed samples).
    valid_dt = dt > 1e-9
    if not np.any(valid_dt):
        raise ValueError(
            "Timestamp sequence has no positive deltas; cannot compute angular velocity for time alignment."
        )

    omega = angles[valid_dt] / dt[valid_dt]
    t_mid = t[:-1][valid_dt] + dt[valid_dt] / 2.0

    finite = np.isfinite(omega) & np.isfinite(t_mid)
    if int(np.sum(finite)) < 2:
        raise ValueError(
            "Angular-velocity signal is invalid after filtering non-finite values; check timestamps/quaternions."
        )
    return t_mid[finite], omega[finite]


def compute_psr(corr, peak_idx, guard_bins):
    corr = np.asarray(corr, dtype=float)
    if corr.size <= 2 * guard_bins + 1:
        return np.nan
    mask = np.isfinite(corr)
    lo = max(0, peak_idx - guard_bins)
    hi = min(corr.size, peak_idx + guard_bins + 1)
    mask[lo:hi] = False
    sidelobe = corr[mask]
    if sidelobe.size < 5:
        return np.nan
    std_side = np.std(sidelobe)
    if std_side < 1e-12:
        return np.nan
    return (corr[peak_idx] - np.mean(sidelobe)) / std_side


def interpolate_quat_slerp(t_src, q_src, t_query):
    t_src = np.asarray(t_src, dtype=float)
    q_src = normalize_quat_array(np.asarray(q_src, dtype=float))
    t_query = np.asarray(t_query, dtype=float)

    t_unique, unique_idx = np.unique(t_src, return_index=True)
    q_unique = q_src[unique_idx]
    if t_unique.size < 2:
        return np.repeat(q_unique[:1], t_query.size, axis=0)

    slerp = Slerp(t_unique, R.from_quat(q_unique))
    q_out = np.zeros((t_query.size, 4), dtype=float)

    in_mask = (t_query >= t_unique[0]) & (t_query <= t_unique[-1])
    if np.any(in_mask):
        q_out[in_mask] = slerp(t_query[in_mask]).as_quat()
    if np.any(~in_mask):
        q_out[t_query < t_unique[0]] = q_unique[0]
        q_out[t_query > t_unique[-1]] = q_unique[-1]

    return normalize_quat_array(q_out)


def interpolate_quat_linear(t_src, q_src, t_query):
    """Normalized linear interpolation with consistent quaternion signs."""
    t_src = np.asarray(t_src, dtype=float).reshape(-1)
    q_src = normalize_quat_array(np.asarray(q_src, dtype=float))
    if np.any(np.diff(t_src) < 0.0):
        # Match the underlying interpolator's chronological ordering before
        # choosing the signs of neighboring quaternions.
        order = np.argsort(t_src, kind="stable")
        t_src, q_src = t_src[order], q_src[order]
    # q and -q represent the same rotation. Accumulate sign corrections so
    # every adjacent pair shares a hemisphere and cannot cancel at its midpoint.
    dots = np.sum(q_src[:-1] * q_src[1:], axis=1)
    q_src[1:] *= np.cumprod(np.where(dots < 0.0, -1.0, 1.0))[:, None]
    return normalize_quat_array(interpolate_linear_extrapolate(t_src, q_src, t_query))


def matching_time_indices(stamps_1, stamps_2, max_diff=0.01, offset_2=0.0):
    """
    One-to-one, monotonic timestamp association:
    for each stamp in stamps_1 (reference), match at most one nearest
    stamp in (stamps_2 + offset_2), and each stamps_2 sample can be used once.
    """
    s1 = np.asarray(stamps_1, dtype=float).reshape(-1)
    s2 = np.asarray(stamps_2, dtype=float).reshape(-1) + float(offset_2)
    max_diff = float(max_diff)

    idx_1 = []
    idx_2 = []
    if s1.size == 0 or s2.size == 0:
        return idx_1, idx_2

    if (
        np.all(np.isfinite(s1))
        and np.all(np.isfinite(s2))
        and np.all(np.diff(s1) >= 0.0)
        and np.all(np.diff(s2) >= 0.0)
    ):
        # Reproduce the loop's exact two candidates: the first estimate not
        # below t-max_diff and its successor.  Directly accept the batch only
        # when consuming a previous match cannot change any later candidate;
        # otherwise retain the sequential fallback below.
        starts = np.searchsorted(s2, s1 - max_diff, side="left")
        active_count = int(np.searchsorted(starts, s2.size, side="left"))
        if active_count > 0:
            ref_ids = np.arange(active_count, dtype=int)
            first = starts[:active_count]
            second = np.minimum(first + 1, s2.size - 1)
            first_diff = np.abs(s2[first] - s1[:active_count])
            second_diff = np.abs(s2[second] - s1[:active_count])
            use_second = second_diff < first_diff
            best = np.where(use_second, second, first)
            best_diff = np.where(use_second, second_diff, first_diff)
            valid = best_diff <= max_diff
            matched_ref = ref_ids[valid]
            matched_est = best[valid]
            matched_starts = first[valid]
            conflict_free = matched_est.size <= 1 or np.all(
                matched_starts[1:] > matched_est[:-1]
            )
            if conflict_free:
                return matched_ref.tolist(), matched_est.astype(int, copy=False).tolist()

    j = 0
    for i, t in enumerate(s1):
        while j < s2.size and s2[j] < (t - max_diff):
            j += 1
        if j >= s2.size:
            break

        best_j = -1
        best_diff = np.inf
        for cand in (j, j + 1):
            if cand >= s2.size:
                continue
            diff = abs(s2[cand] - t)
            if diff <= max_diff and diff < best_diff:
                best_j = int(cand)
                best_diff = float(diff)

        if best_j >= 0:
            idx_1.append(int(i))
            idx_2.append(best_j)
            j = best_j + 1

    return idx_1, idx_2
