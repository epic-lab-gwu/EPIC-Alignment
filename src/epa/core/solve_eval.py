from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .time_alignment import interpolate_quat_linear, interpolate_quat_slerp


def _downsample_by_max_hz(tvals, pos, quat, max_hz):
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    max_hz = float(max_hz)
    if max_hz <= 0.0 or tvals.size <= 2:
        return (
            tvals,
            pos,
            quat,
            {
                "enabled": False,
                "max_hz": max_hz,
                "input_samples": int(tvals.size),
                "output_samples": int(tvals.size),
            },
        )

    min_dt = 1.0 / max_hz
    keep = [0]
    last_t = float(tvals[0])
    for idx in range(1, tvals.size - 1):
        if float(tvals[idx]) - last_t >= min_dt * (1.0 - 1e-9):
            keep.append(idx)
            last_t = float(tvals[idx])
    if keep[-1] != tvals.size - 1:
        keep.append(tvals.size - 1)

    ids = np.asarray(keep, dtype=int)
    return (
        tvals[ids],
        pos[ids],
        quat[ids],
        {
            "enabled": bool(ids.size < tvals.size),
            "max_hz": max_hz,
            "input_samples": int(tvals.size),
            "output_samples": int(ids.size),
        },
    )


def _select_gt_overlap_window(tvals, pos, quat, t_est_sync, min_samples=10):
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    t_est_sync = np.asarray(t_est_sync, dtype=float).reshape(-1)

    finite_est = t_est_sync[np.isfinite(t_est_sync)]
    input_samples = int(tvals.size)
    info = {
        "mode": "full_gt_extrapolate_fallback",
        "fallback_used": True,
        "fallback_reason": "",
        "input_samples": input_samples,
        "selected_samples": input_samples,
        "overlap_samples": 0,
        "overlap_ratio": 0.0,
        "extrapolated_sample_ratio": 1.0,
        "est_sync_start_s": float("nan"),
        "est_sync_end_s": float("nan"),
        "min_samples": int(min_samples),
    }
    if input_samples == 0 or finite_est.size == 0:
        info["fallback_reason"] = "empty timestamp input"
        return tvals, pos, quat, info

    est_start = float(np.min(finite_est))
    est_end = float(np.max(finite_est))
    info["est_sync_start_s"] = est_start
    info["est_sync_end_s"] = est_end
    eps = (
        max(1e-9, float(np.nanmedian(np.abs(np.diff(tvals)))) * 1e-6) if input_samples > 1 else 1e-9
    )
    mask = (tvals >= est_start - eps) & (tvals <= est_end + eps)
    overlap_samples = int(np.sum(mask))
    overlap_ratio = float(overlap_samples / max(1, input_samples))
    info["overlap_samples"] = overlap_samples
    info["overlap_ratio"] = overlap_ratio
    info["extrapolated_sample_ratio"] = float(1.0 - overlap_ratio)

    if overlap_samples >= int(min_samples):
        info.update(
            {
                "mode": "overlap",
                "fallback_used": False,
                "fallback_reason": "",
                "selected_samples": overlap_samples,
                "extrapolated_sample_ratio": 0.0,
            }
        )
        return tvals[mask], pos[mask], quat[mask], info

    info["fallback_reason"] = f"overlap_samples={overlap_samples} < min_samples={int(min_samples)}"
    return tvals, pos, quat, info


def _map_est_to_ref_nearest(
    *,
    t_ref,
    t_est,
    pos_est,
    quat_est,
    offset_est_s: float,
):
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    t_est_shift = np.asarray(t_est, dtype=float).reshape(-1) - float(offset_est_s)
    pos_est = np.asarray(pos_est, dtype=float)
    quat_est = np.asarray(quat_est, dtype=float)
    if t_est_shift.size == 0:
        raise ValueError("Empty estimation timestamps for nearest mapping.")

    idx = np.empty(t_ref.size, dtype=int)
    j = np.searchsorted(t_est_shift, t_ref, side="left")
    for i, k in enumerate(j):
        if k <= 0:
            idx[i] = 0
            continue
        if k >= t_est_shift.size:
            idx[i] = t_est_shift.size - 1
            continue
        d0 = abs(t_ref[i] - t_est_shift[k - 1])
        d1 = abs(t_ref[i] - t_est_shift[k])
        idx[i] = (k - 1) if d0 <= d1 else k

    return pos_est[idx], quat_est[idx]


def _prepare_solve_eval_trajectories(
    *,
    t_gt,
    pos_gt,
    quat_gt,
    t_est,
    pos_est,
    quat_est,
    calculated_offset: float,
    downsample_hz: float,
    quat_interp: str,
):
    t_est_sync = np.asarray(t_est, dtype=float) - float(calculated_offset)
    t_gt_full = np.asarray(t_gt, dtype=float)
    pos_gt_full = np.asarray(pos_gt, dtype=float)
    quat_gt_full = np.asarray(quat_gt, dtype=float)
    t_gt_eval, pos_gt_eval, quat_gt_eval, overlap_info = _select_gt_overlap_window(
        t_gt_full,
        pos_gt_full,
        quat_gt_full,
        t_est_sync,
        min_samples=10,
    )

    downsample_hz = float(downsample_hz)
    if downsample_hz < 0.0:
        raise ValueError("--downsample-hz must be >= 0.")
    t_gt_out, pos_gt_out, quat_gt_out, downsample_info = _downsample_by_max_hz(
        t_gt_eval,
        pos_gt_eval,
        quat_gt_eval,
        downsample_hz,
    )

    interp_p = interp1d(t_est_sync, pos_est, axis=0, fill_value="extrapolate")
    pr_sync = interp_p(t_gt_out)
    if str(quat_interp) == "slerp":
        qr_sync = interpolate_quat_slerp(t_est_sync, quat_est, t_gt_out)
    else:
        qr_sync = interpolate_quat_linear(t_est_sync, quat_est, t_gt_out)

    return {
        "t_est_sync": t_est_sync,
        "t_gt": t_gt_out,
        "pos_gt": pos_gt_out,
        "quat_gt": quat_gt_out,
        "pr_sync": pr_sync,
        "qr_sync": qr_sync,
        "pos_gt_solve": np.asarray(pos_gt_out, dtype=float),
        "quat_gt_solve": np.asarray(quat_gt_out, dtype=float),
        "pr_solve": np.asarray(pr_sync, dtype=float),
        "qr_solve": np.asarray(qr_sync, dtype=float),
        "overlap_info": overlap_info,
        "downsample_info": downsample_info,
    }
