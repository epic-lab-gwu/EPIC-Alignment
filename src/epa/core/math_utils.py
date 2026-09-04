import numpy as np
from scipy.spatial.transform import Rotation as R


def rmse(x):
    return np.sqrt(np.mean(np.square(x)))


def compute_error_statistics(errors):
    values = np.asarray(errors, dtype=float).reshape(-1)
    if values.size == 0:
        return {
            "rmse": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "sse": np.nan,
        }
    sq = np.square(values)
    return {
        "rmse": np.sqrt(np.mean(sq)),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "sse": np.sum(sq),
    }


def normalize_quat_array(quat):
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / np.clip(norm, 1e-12, None)


def quat_multiply_xyzw(lhs, rhs):
    """Vectorized Hamilton product using SciPy's scalar-last convention."""
    q_left = np.asarray(lhs, dtype=float)
    q_right = np.asarray(rhs, dtype=float)
    left_vec = q_left[..., :3]
    left_w = q_left[..., 3:4]
    right_vec = q_right[..., :3]
    right_w = q_right[..., 3:4]
    vector = (
        left_w * right_vec
        + right_w * left_vec
        + np.cross(left_vec, right_vec)
    )
    scalar = left_w * right_w - np.sum(
        left_vec * right_vec,
        axis=-1,
        keepdims=True,
    )
    return np.concatenate([vector, scalar], axis=-1)


def quat_angle_error_rad(q_ref, q_est):
    """Return shortest rotation-angle errors for paired xyzw quaternions."""
    ref = np.asarray(q_ref, dtype=float)
    est = np.asarray(q_est, dtype=float)
    ref = ref / np.clip(np.linalg.norm(ref, axis=-1, keepdims=True), 1e-12, None)
    est = est / np.clip(np.linalg.norm(est, axis=-1, keepdims=True), 1e-12, None)
    est_inv = est.copy()
    est_inv[..., :3] *= -1.0
    relative = quat_multiply_xyzw(est_inv, ref)
    relative = relative / np.clip(
        np.linalg.norm(relative, axis=-1, keepdims=True),
        1e-12,
        None,
    )
    return 2.0 * np.arctan2(
        np.linalg.norm(relative[..., :3], axis=-1),
        np.abs(relative[..., 3]),
    )


def normalize_time_to_seconds(t, *, zero_start=True, unit="auto"):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("Timestamp sequence must be a 1D array with at least two points.")

    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Timestamp sequence must be strictly increasing.")

    unit = str(unit).strip().lower()
    unit_scales = {
        "s": 1.0,
        "sec": 1.0,
        "seconds": 1.0,
        "ms": 1e3,
        "milliseconds": 1e3,
        "us": 1e6,
        "microseconds": 1e6,
        "ns": 1e9,
        "nanoseconds": 1e9,
    }
    if unit == "auto":
        median_dt = np.median(dt)
        if median_dt > 1e6:
            scale = 1e9
        elif median_dt > 1e3:
            scale = 1e6
        elif median_dt > 1.0:
            scale = 1e3
        else:
            scale = 1.0
    elif unit in unit_scales:
        scale = unit_scales[unit]
    else:
        raise ValueError(f"Unsupported timestamp unit: {unit}")

    t_sec = t / scale
    if bool(zero_start):
        t_sec = t_sec - t_sec[0]
    return t_sec


def poses_se3_from_traj(pos, quat):
    pos = np.asarray(pos, dtype=float)
    rot = R.from_quat(np.asarray(quat, dtype=float)).as_matrix()
    n = pos.shape[0]
    T = np.repeat(np.eye(4)[None, :, :], n, axis=0)
    T[:, :3, :3] = rot
    T[:, :3, 3] = pos
    return T


def relative_se3(a, b):
    Ra = a[:3, :3]
    ta = a[:3, 3]
    Rb = b[:3, :3]
    tb = b[:3, 3]
    R_rel = Ra.T @ Rb
    t_rel = Ra.T @ (tb - ta)
    out = np.eye(4)
    out[:3, :3] = R_rel
    out[:3, 3] = t_rel
    return out
