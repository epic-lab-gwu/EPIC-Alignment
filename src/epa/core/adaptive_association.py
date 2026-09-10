"""Local-rate pose association and independent, conservative VIO reset detection."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _local_median(values):
    # Exclude the candidate; bounded batches avoid a large N x 20 allocation.
    values = np.asarray(values, dtype=float)
    if len(values) <= 2:
        return np.full_like(values, np.median(values))
    radius = min(10, (len(values)-1)//2)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(values, radius, mode="reflect"), 2*radius+1)
    neighbors = np.r_[0:radius, radius+1:2*radius+1]
    result = np.empty_like(values)
    for start in range(0, len(values), 32768):
        result[start:start+32768] = np.median(windows[start:start+32768, neighbors], axis=1)
    return result


def local_period(t):
    t = np.asarray(t, dtype=float)
    if t.size < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Association requires at least two finite, strictly increasing timestamps")
    return _local_median(np.diff(t))


def detect_resets(t, pos, quat, *, candidate_intervals=None):
    """Require an unusually long gap AND unusually large elapsed-time-scaled motion.

    Normal timestamp spacing is never a reset. Thresholds are deliberately
    conservative: dt > max(0.5 s, 5*local dt), and displacement > max(5 m,
    5*local speed*dt) or rotation > max(45 deg, 5*local angular speed*dt).
    These are discontinuity heuristics, not proof of tracking health.
    """
    t = np.asarray(t, dtype=float)
    period = local_period(t)
    dt = np.diff(t)
    distance = np.linalg.norm(np.diff(np.asarray(pos), axis=0), axis=1)
    rotations = Rotation.from_quat(quat)
    angle = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude())
    distance_limit = np.maximum(5., 5. * _local_median(distance / dt) * dt)
    angle_limit = np.maximum(45., 5. * _local_median(angle / dt) * dt)
    gap_limit = np.maximum(.5, 5. * period)
    reset = (dt > gap_limit) & ((distance > distance_limit) | (angle > angle_limit))
    if candidate_intervals is not None:
        reset &= np.asarray(candidate_intervals, dtype=bool)
    return [dict(start_s=float(t[i]), end_s=float(t[i+1]), index=int(i),
                 duration_s=float(dt[i]), translation_m=float(distance[i]),
                 rotation_deg=float(angle[i]), gap_limit_s=float(gap_limit[i]),
                 translation_limit_m=float(distance_limit[i]),
                 rotation_limit_deg=float(angle_limit[i])) for i in np.flatnonzero(reset)]


def reset_crossing_mask(t, resets):
    t = np.asarray(t, dtype=float)
    blocked = np.zeros(max(0, t.size - 1), dtype=bool)
    for gap in resets:
        blocked |= (t[:-1] < gap["end_s"]) & (t[1:] > gap["start_s"])
    return blocked


def _inside_resets(t, resets):
    inside = np.zeros(len(t), dtype=bool)
    for gap in resets:
        inside |= (t > gap["start_s"]) & (t < gap["end_s"])
    return inside


def _period_at(t, period, query):
    ids = np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(period)-1)
    return period[ids]


def detect_interpolation_resets(t_est, pos, quat, t_gt):
    """Check only VIO brackets requested by the sparse-GT association branch.

    Exact matches need no interpolation. Use local rates before source-gap
    rejection, so an unsafe requested bracket can still be recorded as failed.
    """
    te, tg = np.asarray(t_est, dtype=float), np.asarray(t_gt, dtype=float)
    ep, gp = local_period(te), local_period(tg)
    hi = np.searchsorted(te, tg)
    exact = te[np.clip(hi, 0, len(te)-1)] == tg
    requested = (_period_at(tg, gp, tg) > _period_at(te, ep, tg) * 1.001)
    requested &= (hi > 0) & (hi < len(te)) & ~exact
    candidates = np.zeros(len(te)-1, dtype=bool)
    candidates[hi[requested]-1] = True
    if not np.any(candidates):
        return []
    return detect_resets(te, pos, quat, candidate_intervals=candidates)


def interpolate_supported(t, pos, quat, query, *, resets=()):
    """Interpolate within locally short source intervals; retain exact samples."""
    t, query = np.asarray(t, dtype=float), np.asarray(query, dtype=float)
    period = local_period(t)
    hi = np.searchsorted(t, query)
    safe_hi = np.clip(hi, 0, len(t)-1)
    lo = np.clip(hi-1, 0, len(t)-2)
    exact = (hi < len(t)) & (t[safe_hi] == query)
    bracket = (hi > 0) & (hi < len(t))
    short = t[lo+1] - t[lo] <= 3. * period[lo] + 1e-9
    supported = np.isfinite(query) & (exact | (bracket & short))
    supported &= ~_inside_resets(query, resets)
    ids = np.flatnonzero(supported)
    p = np.asarray(pos)[safe_hi[ids]].copy()
    q = np.asarray(quat)[safe_hi[ids]].copy()
    interp = ~exact[ids]
    if np.any(interp):
        left = lo[ids[interp]]
        fraction = ((query[ids[interp]] - t[left]) / (t[left+1] - t[left]))[:, None]
        p[interp] = (1-fraction) * np.asarray(pos)[left] + fraction * np.asarray(pos)[left+1]
        q[interp] = Slerp(t, Rotation.from_quat(quat))(query[ids[interp]]).as_quat()
    source_ids = np.where(exact[ids], safe_hi[ids], lo[ids])
    return ids, p, q, source_ids


def associate_adaptive(t_est, p_est, q_est, t_gt, p_gt, q_gt, *, offset=0.):
    """Sample the locally slower trajectory, interpolating the faster one.

    Equal rates prefer GT-to-estimate. No extrapolation or dense fallback is
    involved. Interpolated estimate indices identify their left source pose.
    """
    te = np.asarray(t_est, dtype=float) + float(offset)
    tg = np.asarray(t_gt, dtype=float)
    ep, gp = local_period(te), local_period(tg)
    resets = detect_interpolation_resets(te, p_est, q_est, tg)
    # Retain exact timestamp matches irrespective of the local rate decision.
    hi = np.clip(np.searchsorted(tg, te), 0, len(tg)-1)
    target_est = (_period_at(tg, gp, te) <= _period_at(te, ep, te) * 1.001) | (tg[hi] == te)
    target_gt = _period_at(tg, gp, tg) > _period_at(te, ep, tg) * 1.001
    est_ids = np.flatnonzero(target_est & ~_inside_resets(te, resets))
    keep, pg, qg, _ = interpolate_supported(tg, p_gt, q_gt, te[est_ids])
    est_ids = est_ids[keep]
    gt_ids = np.flatnonzero(target_gt)
    keep, pe, qe, source_ids = interpolate_supported(te, p_est, q_est, tg[gt_ids], resets=resets)
    gt_ids = gt_ids[keep]
    times = np.r_[te[est_ids], tg[gt_ids]]
    # np.unique sorts and selects the first occurrence, preferring exact estimates.
    times, unique = np.unique(times, return_index=True)
    if times.size < 3:
        raise ValueError(f"Unable to associate enough timestamps between estimate and ground truth. matches={times.size}, offset={offset}.")
    return (times, np.concatenate((np.asarray(p_est)[est_ids], pe))[unique],
            np.concatenate((np.asarray(q_est)[est_ids], qe))[unique], times,
            np.concatenate((pg, np.asarray(p_gt)[gt_ids]))[unique],
            np.concatenate((qg, np.asarray(q_gt)[gt_ids]))[unique],
            np.r_[est_ids, source_ids][unique])
