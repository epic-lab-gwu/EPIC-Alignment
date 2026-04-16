import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

from .math_utils import normalize_quat_array


def get_angular_velocity_norm(t, quats):
    rot = R.from_quat(quats)
    rel_rot = rot[:-1].inv() * rot[1:]
    angles = rel_rot.magnitude()
    dt = np.diff(t)
    omega = angles / dt
    t_mid = t[:-1] + dt / 2.0
    return t_mid, omega


def compute_psr(corr, peak_idx, guard_bins):
    if corr.size <= 2 * guard_bins + 1:
        return np.nan
    mask = np.ones_like(corr, dtype=bool)
    lo = max(0, peak_idx - guard_bins)
    hi = min(corr.size, peak_idx + guard_bins + 1)
    mask[lo:hi] = False
    sidelobe = corr[mask]
    std_side = np.std(sidelobe)
    if sidelobe.size < 5 or std_side < 1e-12:
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
    interp_q = interp1d(t_src, q_src, axis=0, fill_value="extrapolate")
    return normalize_quat_array(interp_q(t_query))


def matching_time_indices(stamps_1, stamps_2, max_diff=0.01, offset_2=0.0):
    """
    Evo-compatible timestamp association:
    for each stamp in stamps_1, find nearest in (stamps_2 + offset_2),
    and accept if absolute difference <= max_diff.
    """
    s1 = np.asarray(stamps_1, dtype=float).reshape(-1)
    s2 = np.asarray(stamps_2, dtype=float).reshape(-1) + float(offset_2)
    idx_1 = []
    idx_2 = []
    for i, t in enumerate(s1):
        diffs = np.abs(s2 - t)
        j = int(np.argmin(diffs))
        if diffs[j] <= float(max_diff):
            idx_1.append(i)
            idx_2.append(j)
    return idx_1, idx_2
