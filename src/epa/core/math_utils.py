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


def normalize_time_to_seconds(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("Timestamp sequence must be a 1D array with at least two points.")

    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Timestamp sequence must be strictly increasing.")

    median_dt = np.median(dt)
    if median_dt > 1e6:
        scale = 1e9
    elif median_dt > 1e3:
        scale = 1e6
    elif median_dt > 1.0:
        scale = 1e3
    else:
        scale = 1.0

    t_sec = (t - t[0]) / scale
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

