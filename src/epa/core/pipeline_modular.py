from datetime import datetime
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R

from .calibration import (
    build_translation_system,
    solve_extrinsic_rotation,
    solve_extrinsic_translation,
    solve_world_alignment,
)
from .evaluation import (
    compute_ape_evo_style,
    compute_rpe_evo_style,
    normalize_pose_relation,
    print_metric_block,
    summarize_abs_errors,
)
from .io_utils import (
    load_estimation_trajectory,
    load_reference_trajectory,
    make_output_dir,
    save_metrics,
    write_result_bundle,
    write_run_reports,
)
from .math_utils import normalize_quat_array, rmse
from .time_alignment import (
    compute_psr,
    get_angular_velocity_norm,
    interpolate_quat_linear,
    interpolate_quat_slerp,
    matching_time_indices,
)
from ..viz.rerun_viz import log_alignment_to_rerun
from ..viz.metric_plots import generate_ape_stage_raw_plot, generate_metric_plots


def _apply_time_window(tvals, pos, quat, t_start=None, t_end=None):
    if t_start is None and t_end is None:
        return tvals, pos, quat
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    t_rel = tvals - tvals[0]
    mask = np.ones(tvals.shape[0], dtype=bool)
    if t_start is not None:
        mask &= t_rel >= float(t_start)
    if t_end is not None:
        mask &= t_rel <= float(t_end)
    if np.sum(mask) < 2:
        raise ValueError("Time window filtering kept fewer than 2 trajectory samples.")
    return tvals[mask], np.asarray(pos, dtype=float)[mask], np.asarray(quat, dtype=float)[mask]


def _cum_distance(points_xyz):
    points_xyz = np.asarray(points_xyz, dtype=float)
    if points_xyz.shape[0] == 0:
        return np.array([], dtype=float)
    d = np.zeros(points_xyz.shape[0], dtype=float)
    if points_xyz.shape[0] > 1:
        d[1:] = np.cumsum(np.linalg.norm(np.diff(points_xyz, axis=0), axis=1))
    return d


def _line_segments_xyz(points_xyz):
    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=float)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _set_axes_equal_3d(ax) -> None:
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmean = float(np.mean(xlim))
    ymean = float(np.mean(ylim))
    zmean = float(np.mean(zlim))
    plot_radius = max(
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    )
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def _plot_alignment_map(
    fig,
    ax,
    *,
    pos_ref,
    pos_est,
    errors_m,
    title: str,
):
    pref = np.asarray(pos_ref, dtype=float)
    pest = np.asarray(pos_est, dtype=float)
    err = np.asarray(errors_m, dtype=float).reshape(-1)
    if pref.shape != pest.shape or pref.shape[0] < 2 or err.size != pref.shape[0]:
        raise ValueError("alignment map requires equal-shape gt/est trajectories with per-pose errors.")

    ax.set_title(title)
    ax.plot(
        pref[:, 0],
        pref[:, 1],
        pref[:, 2],
        linestyle="--",
        color="#8a8a8a",
        alpha=0.75,
        linewidth=1.2,
        label="ground truth",
    )

    seg = _line_segments_xyz(pest)
    seg_err = 0.5 * (err[:-1] + err[1:])
    coll = Line3DCollection(seg, cmap="jet", linewidth=1.8)
    coll.set_array(seg_err)
    coll.set_clim(float(np.min(err)), float(np.max(err)))
    ax.add_collection3d(coll)
    ax.scatter(
        pest[:, 0],
        pest[:, 1],
        pest[:, 2],
        c=err,
        cmap="jet",
        s=3.0,
        alpha=0.95,
        linewidths=0.0,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper right")
    fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
    _set_axes_equal_3d(ax)


def _umeyama_transform(src_xyz, dst_xyz, with_scale=False):
    src = np.asarray(src_xyz, dtype=float)
    dst = np.asarray(dst_xyz, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Umeyama alignment requires at least 3 paired 3D points.")

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = (dst_centered.T @ src_centered) / float(src.shape[0])
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R_align = U @ S @ Vt

    scale = 1.0
    if with_scale:
        var_src = np.mean(np.sum(src_centered**2, axis=1))
        if var_src < 1e-12:
            raise ValueError("Degenerate source trajectory for scale alignment.")
        scale = float(np.trace(np.diag(D) @ S) / var_src)
    t_align = mu_dst - scale * (R_align @ mu_src)
    return scale, R_align, t_align


def _align_for_eval(pos_ref, quat_ref, pos_est, quat_est, mode="none", n_to_align=-1):
    mode = str(mode).lower()
    if mode == "none":
        return pos_est, quat_est

    n = pos_ref.shape[0]
    n_use = n if int(n_to_align) <= 0 else min(n, int(n_to_align))
    if n_use < 2:
        return pos_est, quat_est

    pref = np.asarray(pos_ref[:n_use], dtype=float)
    pest = np.asarray(pos_est[:n_use], dtype=float)
    q_est = np.asarray(quat_est, dtype=float)

    if mode == "se3":
        _, R_eval, t_eval = _umeyama_transform(pest, pref, with_scale=False)
        pos_new = (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        return pos_new, q_new

    if mode == "sim3":
        s_eval, R_eval, t_eval = _umeyama_transform(pest, pref, with_scale=True)
        pos_new = s_eval * (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        return pos_new, q_new

    if mode == "scale":
        src_centered = pest - np.mean(pest, axis=0)
        dst_centered = pref - np.mean(pref, axis=0)
        denom = np.sum(src_centered**2)
        scale = 1.0 if denom < 1e-12 else float(np.sqrt(np.sum(dst_centered**2) / denom))
        pos_new = scale * np.asarray(pos_est, dtype=float)
        return pos_new, q_est

    if mode == "origin":
        R0 = R.from_quat(quat_ref[0]).as_matrix() @ R.from_quat(quat_est[0]).as_matrix().T
        t0 = np.asarray(pos_ref[0], dtype=float) - (R0 @ np.asarray(pos_est[0], dtype=float))
        pos_new = (R0 @ np.asarray(pos_est, dtype=float).T).T + t0
        q_new = normalize_quat_array((R.from_matrix(R0) * R.from_quat(q_est)).as_quat())
        return pos_new, q_new

    raise ValueError(f"Unsupported eval alignment mode: {mode}")


def _project_to_plane(pos_xyz, quat_xyzw, plane="none"):
    plane = str(plane).lower()
    pos = np.asarray(pos_xyz, dtype=float).copy()
    quat = np.asarray(quat_xyzw, dtype=float).copy()
    if plane == "none":
        return pos, quat

    if plane == "xy":
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
        e2 = np.array([0.0, 1.0, 0.0], dtype=float)
        pos[:, 2] = 0.0
    elif plane == "xz":
        normal = np.array([0.0, 1.0, 0.0], dtype=float)
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
        e2 = np.array([0.0, 0.0, 1.0], dtype=float)
        pos[:, 1] = 0.0
    elif plane == "yz":
        normal = np.array([1.0, 0.0, 0.0], dtype=float)
        e1 = np.array([0.0, 1.0, 0.0], dtype=float)
        e2 = np.array([0.0, 0.0, 1.0], dtype=float)
        pos[:, 0] = 0.0
    else:
        raise ValueError(f"Unsupported projection plane: {plane}")

    mats = R.from_quat(quat).as_matrix()
    out_quat = np.zeros_like(quat)
    for i in range(mats.shape[0]):
        v = mats[i] @ e1
        v = v - float(np.dot(v, normal)) * normal
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = e1.copy()
            nv = 1.0
        v = v / nv
        ang = np.arctan2(np.dot(v, e2), np.dot(v, e1))
        out_quat[i] = R.from_rotvec(ang * normal).as_quat()

    return pos, normalize_quat_array(out_quat)


def _segment_ranges_by_time(
    tvals,
    *,
    segment_duration_s: float,
    overlap_ratio: float = 0.5,
    min_samples: int = 40,
):
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    n = int(tvals.size)
    if n <= 0:
        return []
    if n < int(min_samples):
        return [(0, n)]

    if segment_duration_s <= 0.0:
        return [(0, n)]
    overlap_ratio = float(np.clip(overlap_ratio, 0.0, 0.95))

    dts = np.diff(tvals)
    finite_dts = dts[np.isfinite(dts) & (dts > 1e-9)]
    if finite_dts.size == 0:
        return [(0, n)]
    dt_med = float(np.median(finite_dts))
    seg_len = max(int(min_samples), int(round(float(segment_duration_s) / dt_med)))
    seg_len = min(seg_len, n)
    step = max(1, int(round(seg_len * (1.0 - overlap_ratio))))

    ranges = []
    start = 0
    while start < n:
        end = min(n, start + seg_len)
        if end - start >= int(min_samples):
            ranges.append((start, end))
        if end >= n:
            break
        start += step

    if not ranges:
        ranges = [(0, n)]
    elif ranges[-1][1] < n:
        tail_start = max(0, n - seg_len)
        if n - tail_start >= int(min_samples):
            ranges.append((tail_start, n))
    return ranges


def _segment_heading_angles_deg(pos_ref, pos_est):
    pref = np.asarray(pos_ref, dtype=float)
    pest = np.asarray(pos_est, dtype=float)
    if pref.shape[0] < 3 or pest.shape[0] < 3:
        return np.array([], dtype=float)
    v_ref = np.diff(pref, axis=0)
    v_est = np.diff(pest, axis=0)
    n_ref = np.linalg.norm(v_ref, axis=1)
    n_est = np.linalg.norm(v_est, axis=1)
    valid = (n_ref > 1e-9) & (n_est > 1e-9)
    if not np.any(valid):
        return np.array([], dtype=float)
    cosang = np.sum(v_ref[valid] * v_est[valid], axis=1) / (n_ref[valid] * n_est[valid] + 1e-12)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _compute_alignment_quality(
    *,
    t_ref,
    pos_ref,
    pos_step3,
    raw_rmse_m: float,
    step3_rmse_m: float,
    segment_duration_s: float,
    overlap_ratio: float,
    min_samples: int,
    good_rmse_m: float,
    partial_rmse_m: float,
    good_seg_cv: float,
    good_heading_p90_deg: float,
    partial_min_improve_pct: float,
):
    err = np.linalg.norm(np.asarray(pos_step3, dtype=float) - np.asarray(pos_ref, dtype=float), axis=1)
    ranges = _segment_ranges_by_time(
        t_ref,
        segment_duration_s=segment_duration_s,
        overlap_ratio=overlap_ratio,
        min_samples=min_samples,
    )

    seg_rmse = []
    for s, e in ranges:
        if e - s < max(3, int(min_samples)):
            continue
        seg_rmse.append(float(np.sqrt(np.mean(err[s:e] ** 2))))
    seg_rmse = np.asarray(seg_rmse, dtype=float)
    seg_rmse_mean = float(np.nanmean(seg_rmse)) if seg_rmse.size else float(np.nan)
    seg_rmse_std = float(np.nanstd(seg_rmse)) if seg_rmse.size else float(np.nan)
    seg_rmse_cv = float(seg_rmse_std / (seg_rmse_mean + 1e-12)) if seg_rmse.size else float(np.nan)

    heading_deg = _segment_heading_angles_deg(pos_ref, pos_step3)
    heading_p90 = float(np.percentile(heading_deg, 90)) if heading_deg.size else float(np.nan)
    heading_median = float(np.median(heading_deg)) if heading_deg.size else float(np.nan)

    improve_pct = float((raw_rmse_m - step3_rmse_m) / (raw_rmse_m + 1e-12) * 100.0)
    is_good = (
        np.isfinite(step3_rmse_m)
        and np.isfinite(seg_rmse_cv)
        and np.isfinite(heading_p90)
        and step3_rmse_m <= good_rmse_m
        and seg_rmse_cv <= good_seg_cv
        and heading_p90 <= good_heading_p90_deg
    )
    is_partial = (
        np.isfinite(step3_rmse_m)
        and np.isfinite(improve_pct)
        and (step3_rmse_m <= partial_rmse_m or improve_pct >= partial_min_improve_pct)
    )
    if is_good:
        label = "good_align"
        label_code = 2.0
    elif is_partial:
        label = "partial_align"
        label_code = 1.0
    else:
        label = "poor_align"
        label_code = 0.0

    return {
        "quality_label_code": float(label_code),
        "step3_rmse_m": float(step3_rmse_m),
        "raw_to_step3_improve_pct": float(improve_pct),
        "segment_count": float(seg_rmse.size),
        "segment_rmse_mean_m": float(seg_rmse_mean),
        "segment_rmse_std_m": float(seg_rmse_std),
        "segment_rmse_cv": float(seg_rmse_cv),
        "heading_median_deg": float(heading_median),
        "heading_p90_deg": float(heading_p90),
        "_quality_label": label,
    }


def _compute_rigid_alignability(
    *,
    t_ref,
    pos_ref,
    pos_step2,
    pos_step3,
    segment_duration_s: float,
    overlap_ratio: float,
    min_samples: int,
    max_path_ratio: float,
    max_bbox_ratio: float,
    max_global_local_ratio: float,
    max_sim3_gain_ratio: float,
):
    pref = np.asarray(pos_ref, dtype=float)
    p2 = np.asarray(pos_step2, dtype=float)
    p3 = np.asarray(pos_step3, dtype=float)
    if pref.shape != p2.shape or pref.shape != p3.shape:
        raise ValueError("Rigid alignability check requires pos_ref/pos_step2/pos_step3 with equal shape.")
    if pref.shape[0] < 3:
        raise ValueError("Rigid alignability check requires at least 3 samples.")

    # Scale-free shape diagnostics (symmetric ratios so <1 and >1 are treated equally).
    path_ref = _cum_distance(pref)[-1] if pref.shape[0] > 1 else 0.0
    path_est = _cum_distance(p2)[-1] if p2.shape[0] > 1 else 0.0
    path_ratio_raw = float(path_est / (path_ref + 1e-12))
    path_ratio_sym = float(max(path_ratio_raw, 1.0 / (path_ratio_raw + 1e-12)))

    bbox_ref = float(np.linalg.norm(np.max(pref, axis=0) - np.min(pref, axis=0)))
    bbox_est = float(np.linalg.norm(np.max(p2, axis=0) - np.min(p2, axis=0)))
    bbox_ratio_raw = float(bbox_est / (bbox_ref + 1e-12))
    bbox_ratio_sym = float(max(bbox_ratio_raw, 1.0 / (bbox_ratio_raw + 1e-12)))

    err_global = np.linalg.norm(p3 - pref, axis=1)
    ranges = _segment_ranges_by_time(
        t_ref,
        segment_duration_s=segment_duration_s,
        overlap_ratio=overlap_ratio,
        min_samples=min_samples,
    )
    local_rmse = []
    global_seg_rmse = []
    for s, e in ranges:
        if e - s < max(3, int(min_samples)):
            continue
        try:
            R_seg, t_seg = solve_world_alignment(p2[s:e], pref[s:e])
        except Exception:
            continue
        pred_local = (R_seg @ p2[s:e].T).T + t_seg
        local_rmse.append(float(np.sqrt(np.mean(np.sum((pred_local - pref[s:e]) ** 2, axis=1)))))
        global_seg_rmse.append(float(np.sqrt(np.mean(err_global[s:e] ** 2))))
    local_rmse = np.asarray(local_rmse, dtype=float)
    global_seg_rmse = np.asarray(global_seg_rmse, dtype=float)

    if local_rmse.size and global_seg_rmse.size:
        global_local_ratio = float(np.median(global_seg_rmse / (local_rmse + 1e-12)))
        local_median = float(np.median(local_rmse))
        global_median = float(np.median(global_seg_rmse))
    else:
        global_local_ratio = float(np.nan)
        local_median = float(np.nan)
        global_median = float(np.nan)

    sim3_rmse = float(np.nan)
    sim3_scale = float(np.nan)
    sim3_gain_ratio = float(np.nan)
    sim3_scale_log10_abs = float(np.nan)
    try:
        s_sim3, R_sim3, t_sim3 = _umeyama_transform(p2, pref, with_scale=True)
        pred_sim3 = s_sim3 * (R_sim3 @ p2.T).T + t_sim3
        sim3_err = np.linalg.norm(pred_sim3 - pref, axis=1)
        sim3_rmse = float(np.sqrt(np.mean(sim3_err**2)))
        se3_rmse = float(np.sqrt(np.mean(err_global**2)))
        sim3_scale = float(s_sim3)
        sim3_gain_ratio = float((se3_rmse - sim3_rmse) / (se3_rmse + 1e-12))
        if sim3_scale > 0.0:
            sim3_scale_log10_abs = float(abs(np.log10(sim3_scale)))
    except Exception:
        pass

    severe_scale_mismatch = bool(
        (np.isfinite(sim3_scale_log10_abs) and sim3_scale_log10_abs >= 1.0)
        or (np.isfinite(path_ratio_sym) and path_ratio_sym >= 20.0)
        or (np.isfinite(bbox_ratio_sym) and bbox_ratio_sym >= 20.0)
    )

    failures = []
    if np.isfinite(path_ratio_sym) and path_ratio_sym > float(max_path_ratio):
        failures.append("path_ratio")
    if np.isfinite(bbox_ratio_sym) and bbox_ratio_sym > float(max_bbox_ratio):
        failures.append("bbox_ratio")
    if np.isfinite(global_local_ratio) and global_local_ratio > float(max_global_local_ratio):
        failures.append("global_local_ratio")
    if np.isfinite(sim3_gain_ratio) and sim3_gain_ratio > float(max_sim3_gain_ratio):
        failures.append("sim3_gain")
    if severe_scale_mismatch:
        failures.append("scale_mismatch_severe")

    alignable = len(failures) == 0
    label = "rigidly_alignable" if alignable else "not_rigidly_alignable"

    return {
        "rigid_alignability_code": 1.0 if alignable else 0.0,
        "path_length_ratio_sym": float(path_ratio_sym),
        "bbox_diag_ratio_sym": float(bbox_ratio_sym),
        "segment_local_se3_rmse_median_m": float(local_median),
        "segment_global_se3_rmse_median_m": float(global_median),
        "segment_global_local_rmse_ratio": float(global_local_ratio),
        "sim3_rmse_m": float(sim3_rmse),
        "sim3_scale": float(sim3_scale),
        "sim3_scale_log10_abs": float(sim3_scale_log10_abs),
        "sim3_gain_ratio": float(sim3_gain_ratio),
        "scale_mismatch_severe_code": 1.0 if severe_scale_mismatch else 0.0,
        "rigid_check_fail_count": float(len(failures)),
        "_rigid_alignability_label": label,
        "_rigid_alignability_reasons": ",".join(failures),
    }


def _compute_piecewise_diagnostics(
    *,
    t_ref,
    pos_ref,
    pos_step2,
    pos_step3,
    segment_duration_s: float,
    overlap_ratio: float,
    min_samples: int,
):
    """
    Diagnostic-only piecewise analysis:
    - global_rmse: error of the single global Step3 transform on each segment.
    - local_rmse: per-segment best SE3 fit from Step2 trajectory to GT.
    This does not modify the main aligned trajectory.
    """
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    pref = np.asarray(pos_ref, dtype=float)
    p2 = np.asarray(pos_step2, dtype=float)
    p3 = np.asarray(pos_step3, dtype=float)

    ranges = _segment_ranges_by_time(
        t_ref,
        segment_duration_s=segment_duration_s,
        overlap_ratio=overlap_ratio,
        min_samples=min_samples,
    )

    seg_mid_s = []
    seg_global_rmse = []
    seg_local_rmse = []
    for s, e in ranges:
        if e - s < max(3, int(min_samples)):
            continue

        mid = 0.5 * (float(t_ref[s]) + float(t_ref[e - 1])) - float(t_ref[0])
        seg_mid_s.append(mid)

        err_g = np.linalg.norm(p3[s:e] - pref[s:e], axis=1)
        seg_global_rmse.append(float(np.sqrt(np.mean(err_g**2))))

        try:
            R_seg, t_seg = solve_world_alignment(p2[s:e], pref[s:e])
            pred_local = (R_seg @ p2[s:e].T).T + t_seg
            err_l = np.linalg.norm(pred_local - pref[s:e], axis=1)
            seg_local_rmse.append(float(np.sqrt(np.mean(err_l**2))))
        except Exception:
            seg_local_rmse.append(float(np.nan))

    seg_mid_s = np.asarray(seg_mid_s, dtype=float)
    seg_global_rmse = np.asarray(seg_global_rmse, dtype=float)
    seg_local_rmse = np.asarray(seg_local_rmse, dtype=float)

    valid = np.isfinite(seg_global_rmse) & np.isfinite(seg_local_rmse)
    if np.any(valid):
        g = seg_global_rmse[valid]
        l = seg_local_rmse[valid]
        gap = g - l
        ratio = g / (l + 1e-12)

        peak_idx = int(np.argmax(g))
        if g.size >= 2 and peak_idx < g.size - 1:
            post_best = float(np.min(g[peak_idx + 1 :]))
            recovery_pct = float((g[peak_idx] - post_best) / (g[peak_idx] + 1e-12) * 100.0)
        else:
            recovery_pct = float(np.nan)

        q = max(1, g.size // 4)
        early = float(np.mean(g[:q]))
        late = float(np.mean(g[-q:]))
        early_late_delta = late - early

        metrics = {
            "piecewise_segment_count": float(g.size),
            "piecewise_global_rmse_median_m": float(np.median(g)),
            "piecewise_local_rmse_median_m": float(np.median(l)),
            "piecewise_global_local_ratio_median": float(np.median(ratio)),
            "piecewise_gap_median_m": float(np.median(gap)),
            "piecewise_gap_p90_m": float(np.percentile(gap, 90)),
            "piecewise_gap_max_m": float(np.max(gap)),
            "piecewise_peak_recovery_pct": float(recovery_pct),
            "piecewise_early_late_delta_m": float(early_late_delta),
        }
    else:
        metrics = {
            "piecewise_segment_count": 0.0,
            "piecewise_global_rmse_median_m": float(np.nan),
            "piecewise_local_rmse_median_m": float(np.nan),
            "piecewise_global_local_ratio_median": float(np.nan),
            "piecewise_gap_median_m": float(np.nan),
            "piecewise_gap_p90_m": float(np.nan),
            "piecewise_gap_max_m": float(np.nan),
            "piecewise_peak_recovery_pct": float(np.nan),
            "piecewise_early_late_delta_m": float(np.nan),
        }

    details = {
        "segment_mid_s": seg_mid_s,
        "segment_global_rmse_m": seg_global_rmse,
        "segment_local_rmse_m": seg_local_rmse,
    }
    return metrics, details


def _build_user_alert(
    *,
    time_metrics,
    traj_metrics,
    quality_label: str,
    rigid_label: str,
    rigid_reasons: str,
):
    issues = []
    peak = float(time_metrics.get("xcorr_peak_normalized", np.nan))
    psr = float(time_metrics.get("xcorr_psr", np.nan))
    match_ratio = float(time_metrics.get("evo_matches_ratio_equivalent", np.nan))
    omega_gain = float(time_metrics.get("omega_rmse_improve_pct", np.nan))
    offset_est = float(time_metrics.get("offset_est_s", np.nan))
    step3_rmse = float(traj_metrics.get("ate_rmse_step3_m", np.nan))
    improve_pct = float(traj_metrics.get("ate_rmse_improve_raw_to_step3_pct", np.nan))
    strong_final_alignment = (
        np.isfinite(step3_rmse)
        and np.isfinite(improve_pct)
        and step3_rmse <= 0.3
        and improve_pct >= 80.0
    )

    if (np.isfinite(peak) and peak < 0.75) or (np.isfinite(psr) and psr < 6.0):
        issues.append("time_alignment_low_confidence")
    if np.isfinite(match_ratio) and match_ratio < 0.5:
        issues.append("time_overlap_low")
    if np.isfinite(omega_gain) and omega_gain < 5.0 and np.isfinite(offset_est) and abs(offset_est) > 0.02:
        issues.append("time_alignment_weak_gain")
    if quality_label == "partial_align":
        if not strong_final_alignment:
            issues.append("alignment_partial")
    elif quality_label == "poor_align":
        issues.append("alignment_poor")
    if rigid_label != "rigidly_alignable":
        if not strong_final_alignment:
            issues.append("rigid_diagnostic_flag")
    rigid_reason_tokens = {tok.strip() for tok in str(rigid_reasons).split(",") if tok.strip()}
    if "scale_mismatch_severe" in rigid_reason_tokens:
        issues.append("scale_mismatch_severe")
    if np.isfinite(step3_rmse) and step3_rmse > 10.0:
        issues.append("step3_rmse_high")
    if np.isfinite(improve_pct) and improve_pct < 10.0:
        issues.append("step3_improvement_small")

    if strong_final_alignment:
        issues = [x for x in issues if x in {"scale_mismatch_severe", "step3_rmse_high"}]

    critical = any(
        item in issues
        for item in (
            "alignment_poor",
            "scale_mismatch_severe",
            "step3_rmse_high",
        )
    )
    warning = (not critical) and len(issues) > 0

    if critical:
        level = "critical"
        level_code = 2.0
    elif warning:
        level = "warning"
        level_code = 1.0
    else:
        level = "ok"
        level_code = 0.0

    reason_map = {
        "time_alignment_low_confidence": "Weak time-alignment confidence",
        "time_overlap_low": "Low effective time overlap",
        "time_alignment_weak_gain": "Limited improvement from time alignment",
        "alignment_partial": "Residual local misalignment remains",
        "alignment_poor": "Overall alignment quality is poor",
        "rigid_diagnostic_flag": "Rigid-diagnostic checks flagged potential model mismatch",
        "scale_mismatch_severe": "Severe scale mismatch between trajectory and ground truth",
        "step3_rmse_high": "High absolute Step-3 error",
        "step3_improvement_small": "Limited Step-3 improvement over raw",
    }
    reasons_en = [reason_map[item] for item in issues if item in reason_map]

    if level == "ok":
        message_en = "The result is stable overall; Step-3 output is reliable."
    elif level == "warning":
        message_en = "Suspicious indicators were detected; please review plots and raw data."
    else:
        if "scale_mismatch_severe" in issues:
            message_en = "Critical issue: VIO and ground truth differ by orders of magnitude in scale; Step-3 may be unreliable."
        else:
            message_en = "Critical issue detected; Step-3 may be unreliable. Check data pairing and time alignment first."

    return {
        "alert_level_code": float(level_code),
        "alert_count": float(len(issues)),
        "_alert_level": level,
        "_alert_message": message_en,
        "_alert_reasons": "; ".join(reasons_en),
    }


def _segment_blend_weights(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=float)
    if length == 1:
        return np.ones(1, dtype=float)
    x = np.linspace(-1.0, 1.0, int(length))
    return np.maximum(1e-3, 1.0 - np.abs(x))


def _compute_piecewise_alignment(
    *,
    t_ref,
    pos_ref,
    pos_base,
    quat_base,
    fallback_pos,
    fallback_quat,
    segment_duration_s: float,
    overlap_ratio: float,
    min_samples: int,
    with_scale: bool = True,
):
    """
    Blend local segment-wise similarity transforms to handle cases where a
    single global rigid transform is not sufficient.
    """
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    pref = np.asarray(pos_ref, dtype=float)
    pbase = np.asarray(pos_base, dtype=float)
    qbase = normalize_quat_array(np.asarray(quat_base, dtype=float))
    pfallback = np.asarray(fallback_pos, dtype=float)
    qfallback = normalize_quat_array(np.asarray(fallback_quat, dtype=float))

    if pref.shape != pbase.shape or pref.shape[0] < 3:
        return None

    ranges = _segment_ranges_by_time(
        t_ref,
        segment_duration_s=segment_duration_s,
        overlap_ratio=overlap_ratio,
        min_samples=min_samples,
    )
    if not ranges:
        return None

    n = pref.shape[0]
    pos_acc = np.zeros((n, 3), dtype=float)
    quat_acc = np.zeros((n, 4), dtype=float)
    weight_acc = np.zeros(n, dtype=float)
    seg_rmse = []
    valid_segments = 0

    for s, e in ranges:
        if e - s < max(3, int(min_samples)):
            continue
        try:
            scale_seg, R_seg, t_seg = _umeyama_transform(pbase[s:e], pref[s:e], with_scale=with_scale)
        except Exception:
            continue

        pred_seg = scale_seg * (R_seg @ pbase[s:e].T).T + t_seg
        q_seg = normalize_quat_array((R.from_matrix(R_seg) * R.from_quat(qbase[s:e])).as_quat())
        w_seg = _segment_blend_weights(e - s)

        pos_acc[s:e] += w_seg[:, None] * pred_seg
        for local_idx, global_idx in enumerate(range(s, e)):
            q_curr = np.asarray(q_seg[local_idx], dtype=float)
            if weight_acc[global_idx] > 0.0 and np.dot(quat_acc[global_idx], q_curr) < 0.0:
                q_curr = -q_curr
            quat_acc[global_idx] += w_seg[local_idx] * q_curr
        weight_acc[s:e] += w_seg
        seg_rmse.append(float(np.sqrt(np.mean(np.sum((pred_seg - pref[s:e]) ** 2, axis=1)))))
        valid_segments += 1

    if valid_segments == 0:
        return None

    pos_out = np.asarray(pfallback, dtype=float).copy()
    quat_out = np.asarray(qfallback, dtype=float).copy()
    valid = weight_acc > 1e-9
    if not np.any(valid):
        return None

    pos_out[valid] = pos_acc[valid] / weight_acc[valid, None]
    quat_norm = np.linalg.norm(quat_acc, axis=1)
    quat_valid = valid & (quat_norm > 1e-9)
    if np.any(quat_valid):
        quat_out[quat_valid] = quat_acc[quat_valid] / quat_norm[quat_valid, None]
    quat_out = normalize_quat_array(quat_out)

    err = np.linalg.norm(pos_out - pref, axis=1)
    seg_rmse = np.asarray(seg_rmse, dtype=float)
    return {
        "pos": pos_out,
        "quat": quat_out,
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "p95_m": float(np.percentile(err, 95)),
        "coverage_ratio": float(np.mean(valid)),
        "segment_count": float(valid_segments),
        "segment_rmse_median_m": float(np.median(seg_rmse)) if seg_rmse.size else float("nan"),
        "mode": "piecewise_local_sim3" if with_scale else "piecewise_local_se3",
    }

def _offset_grid(min_offset_s: float, max_offset_s: float, step_s: float):
    if step_s <= 0.0:
        raise ValueError("offset step must be > 0.")
    values = np.arange(min_offset_s, max_offset_s + 0.5 * step_s, step_s)
    return [float(v) for v in values]


def _match_nearest_timestamps(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
):
    """
    Evo-style nearest-neighbor matching from the first timestamp array to the second.
    The second array is shifted by `offset_est_s` before matching.
    """
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1) + float(offset_est_s)
    idx_ref = []
    idx_est = []
    for i, t_ref in enumerate(s_ref):
        diffs = np.abs(s_est - t_ref)
        j = int(np.argmin(diffs))
        if float(diffs[j]) <= float(max_diff_s):
            idx_ref.append(int(i))
            idx_est.append(int(j))
    return np.asarray(idx_ref, dtype=int), np.asarray(idx_est, dtype=int)


def _associate_gt_est(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
):
    """
    Time association with the same shorter-vs-longer semantics used by the
    metric comparison path:
    - Match from the shorter trajectory to the longer one.
    - Apply `offset_est_s` to estimation timestamps in the metric association convention.
    - Return indices mapped back to (ref_idx, est_idx).
    """
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1)

    if s_ref.size == 0 or s_est.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    est_longer = s_est.size > s_ref.size
    if est_longer:
        # short=ref, long=est, offset applies directly to long(est)
        ids_short, ids_long = _match_nearest_timestamps(
            s_ref,
            s_est,
            max_diff_s=float(max_diff_s),
            offset_est_s=float(offset_est_s),
        )
        ids_ref = ids_short
        ids_est = ids_long
    else:
        # short=est, long=ref, use inverse sign when est is first trajectory
        ids_short, ids_long = _match_nearest_timestamps(
            s_est,
            s_ref,
            max_diff_s=float(max_diff_s),
            offset_est_s=-float(offset_est_s),
        )
        ids_ref = ids_long
        ids_est = ids_short

    return np.asarray(ids_ref, dtype=int), np.asarray(ids_est, dtype=int)


def _offset_match_diagnostics(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
    overlap_gate_min_pairs: int | None = None,
):
    s_ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    s_est = np.asarray(stamps_est, dtype=float).reshape(-1)
    pair_cap_global = max(1, min(s_ref.size, s_est.size))
    if overlap_gate_min_pairs is None:
        overlap_gate_min_pairs = max(80, int(0.1 * float(pair_cap_global)))
    overlap_gate_min_pairs = max(1, int(overlap_gate_min_pairs))

    ids_ref, ids_est = matching_time_indices(
        s_ref,
        s_est,
        max_diff=float(max_diff_s),
        offset_2=float(offset_est_s),
    )
    match_count = int(len(ids_ref))
    ratio_global = min(1.0, float(match_count) / float(pair_cap_global))

    s_est_shift = s_est + float(offset_est_s)
    overlap_start = max(float(s_ref[0]), float(s_est_shift[0]))
    overlap_end = min(float(s_ref[-1]), float(s_est_shift[-1]))
    overlap_cap = 0
    ratio_overlap = ratio_global
    if overlap_end >= overlap_start:
        n_ref_overlap = int(np.sum((s_ref >= overlap_start) & (s_ref <= overlap_end)))
        n_est_overlap = int(np.sum((s_est_shift >= overlap_start) & (s_est_shift <= overlap_end)))
        overlap_cap = int(min(n_ref_overlap, n_est_overlap))
        if overlap_cap > 0:
            ratio_overlap = min(1.0, float(match_count) / float(overlap_cap))

    ratio_gate = ratio_global
    if overlap_cap >= overlap_gate_min_pairs:
        ratio_gate = max(ratio_global, ratio_overlap)

    return {
        "match_count": match_count,
        "ratio_global": float(ratio_global),
        "ratio_overlap": float(ratio_overlap),
        "ratio_gate": float(ratio_gate),
        "pair_cap_global": int(pair_cap_global),
        "pair_cap_overlap": int(overlap_cap),
        "overlap_gate_min_pairs": int(overlap_gate_min_pairs),
        "ids_ref": ids_ref,
        "ids_est": ids_est,
    }


def _map_est_to_ref_nearest(
    *,
    t_ref,
    t_est,
    pos_est,
    quat_est,
    offset_est_s: float,
):
    """
    Map estimation poses to reference timestamps by nearest-neighbor matching
    after applying a constant time offset to estimation timestamps.
    Unlike interpolation, this preserves original estimation samples.
    """
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


def _prefer_offset_candidate(curr, cand):
    if curr is None:
        return True
    curr_valid, curr_rmse, curr_matches, curr_abs = curr
    cand_valid, cand_rmse, cand_matches, cand_abs = cand
    if cand_valid and not curr_valid:
        return True
    if curr_valid and not cand_valid:
        return False
    if cand_rmse < curr_rmse - 1e-12:
        return True
    if np.isclose(cand_rmse, curr_rmse, atol=1e-12, rtol=0.0):
        if cand_matches > curr_matches:
            return True
        if cand_matches == curr_matches and cand_abs < curr_abs - 1e-12:
            return True
    return False


def _search_direct_offset_from_matched_pairs(
    *,
    t_ref,
    pos_ref,
    t_est,
    pos_est,
    initial_offset_s: float,
    max_diff_s: float,
    min_match_ratio: float,
    coarse_min_s: float = -20.0,
    coarse_max_s: float = 20.0,
    coarse_step_s: float = 0.2,
    refine_window_s: float = 0.5,
    refine_step_s: float = 0.01,
):
    t_ref = np.asarray(t_ref, dtype=float).reshape(-1)
    t_est = np.asarray(t_est, dtype=float).reshape(-1)
    pref = np.asarray(pos_ref, dtype=float)
    pest = np.asarray(pos_est, dtype=float)
    if t_ref.size != pref.shape[0] or t_est.size != pest.shape[0]:
        raise ValueError("Timestamp and position array lengths are inconsistent.")

    pair_cap = max(1, min(t_ref.size, t_est.size))
    min_required = max(3, int(float(min_match_ratio) * float(pair_cap)))

    best_key = None
    best = None

    def _eval_offset(offset_s: float):
        ids_ref, ids_est = _associate_gt_est(
            t_ref,
            t_est,
            max_diff_s=float(max_diff_s),
            offset_est_s=-float(offset_s),
        )
        n = len(ids_ref)
        if n < 3:
            return None
        pref_sel = pref[np.asarray(ids_ref, dtype=int)]
        pest_sel = pest[np.asarray(ids_est, dtype=int)]
        Rw, tw = solve_world_alignment(pest_sel, pref_sel)
        pred = (Rw @ pest_sel.T).T + tw
        rmse = float(np.sqrt(np.mean(np.sum((pred - pref_sel) ** 2, axis=1))))
        valid = bool(n >= min_required)
        key = (valid, rmse, int(n), abs(float(offset_s)))
        return {
            "offset_s": float(offset_s),
            "matches": int(n),
            "valid": valid,
            "rmse_pairs_m": rmse,
            "ids_ref": np.asarray(ids_ref, dtype=int),
            "ids_est": np.asarray(ids_est, dtype=int),
            "_key": key,
        }

    for off in _offset_grid(coarse_min_s, coarse_max_s, coarse_step_s):
        candidate = _eval_offset(off)
        if candidate is None:
            continue
        if _prefer_offset_candidate(best_key, candidate["_key"]):
            best_key = candidate["_key"]
            best = candidate

    init_candidate = _eval_offset(float(initial_offset_s))
    if init_candidate is not None and _prefer_offset_candidate(best_key, init_candidate["_key"]):
        best_key = init_candidate["_key"]
        best = init_candidate

    if best is None:
        return None

    refine_start = max(float(coarse_min_s), float(best["offset_s"]) - float(refine_window_s))
    refine_end = min(float(coarse_max_s), float(best["offset_s"]) + float(refine_window_s))
    for off in _offset_grid(refine_start, refine_end, refine_step_s):
        candidate = _eval_offset(off)
        if candidate is None:
            continue
        if _prefer_offset_candidate(best_key, candidate["_key"]):
            best_key = candidate["_key"]
            best = candidate

    best.pop("_key", None)
    best["required_matches"] = int(min_required)
    return best


def run_pipeline_modular(args, script_dir: Path):
    run_dir = make_output_dir(script_dir)
    print(f"Saving outputs to: {run_dir}")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(args.gt_csv)
    if not gt_path.is_absolute():
        gt_path = script_dir / gt_path
    if not gt_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_path}")

    print("Loading GT trajectory...")
    gt_format = getattr(args, "gt_format", "csv")
    gt_topic = getattr(args, "gt_topic", "")
    t_gt, pos_gt, quat_gt = load_reference_trajectory(
        gt_path,
        gt_format=gt_format,
        gt_topic=gt_topic,
    )

    mode = "synthetic" if args.synthetic else "real"
    sanity_metrics = {
        "extrinsic_rotation_error_deg": np.nan,
        "extrinsic_translation_error_m": np.nan,
        "world_rotation_error_deg": np.nan,
        "world_translation_error_m": np.nan,
    }

    if args.synthetic:
        ARTIFICIAL_OFFSET = 0.123
        R_ext_true = R.from_euler("zyx", [45, -20, 30], degrees=True).as_matrix()
        t_ext_true = np.array([0.5, -0.2, 0.1])
        Rw_true = R.from_euler("z", 25, degrees=True).as_matrix()
        tw_true = np.array([1.5, -0.5, 0.3])

        t_est = t_gt + ARTIFICIAL_OFFSET
        pos_est = np.zeros_like(pos_gt)
        quat_est = np.zeros_like(quat_gt)
        R_gt = R.from_quat(quat_gt).as_matrix()

        for i in range(len(t_gt)):
            R_abs = R_gt[i] @ R_ext_true
            quat_est[i] = R.from_matrix(Rw_true.T @ R_abs).as_quat()
            p_offset = pos_gt[i] + R_gt[i] @ t_ext_true
            pos_est[i] = Rw_true.T @ (p_offset - tw_true)
    else:
        if args.est_path is None or args.est_path == "":
            raise ValueError("Real mode requires --est-path")

        est_path = Path(args.est_path)
        if not est_path.is_absolute():
            est_path = script_dir / est_path
        if not est_path.exists():
            raise FileNotFoundError(f"Estimation trajectory file not found: {est_path}")

        print("Loading estimation trajectory...")
        est_format = args.est_format
        est_topic = getattr(args, "est_topic", "")
        t_est, pos_est, quat_est = load_estimation_trajectory(
            est_path,
            est_format,
            est_topic=est_topic,
        )

    t_start = getattr(args, "t_start", None)
    t_end = getattr(args, "t_end", None)
    t_offset = float(getattr(args, "t_offset", 0.0))
    quality_segment_duration_s = float(getattr(args, "quality_segment_duration_s", 10.0))
    quality_segment_overlap_ratio = float(getattr(args, "quality_segment_overlap_ratio", 0.5))
    quality_good_rmse_m = float(getattr(args, "quality_good_rmse_m", 0.5))
    quality_partial_rmse_m = float(getattr(args, "quality_partial_rmse_m", 8.0))
    quality_good_segment_cv = float(getattr(args, "quality_good_segment_cv", 0.4))
    quality_good_heading_p90_deg = float(getattr(args, "quality_good_heading_p90_deg", 60.0))
    quality_partial_min_improve_pct = float(getattr(args, "quality_partial_min_improve_pct", 20.0))
    quality_min_segment_samples = int(getattr(args, "quality_min_segment_samples", 80))
    t_gt, pos_gt, quat_gt = _apply_time_window(
        t_gt, pos_gt, quat_gt, t_start=t_start, t_end=t_end
    )
    t_est = np.asarray(t_est, dtype=float) + t_offset
    t_est, pos_est, quat_est = _apply_time_window(
        t_est, pos_est, quat_est, t_start=t_start, t_end=t_end
    )

    print("--- STEP 1: TIME ALIGNMENT ---")
    t_gt_mid, om_gt = get_angular_velocity_norm(t_gt, quat_gt)
    t_est_mid, om_est = get_angular_velocity_norm(t_est, quat_est)

    dt_resample = float(getattr(args, "dt_resample", 0.001))
    if dt_resample <= 0.0:
        raise ValueError("--dt-resample must be > 0.")
    offset_search_window_s = float(getattr(args, "offset_search_window_s", 0.0))
    offset_min_match_ratio = float(getattr(args, "offset_min_match_ratio", 0.3))
    if not (0.0 <= offset_min_match_ratio <= 1.0):
        raise ValueError("--offset-min-match-ratio must be in [0, 1].")

    t_min = max(t_gt_mid[0], t_est_mid[0])
    t_max = min(t_gt_mid[-1], t_est_mid[-1])
    if t_max - t_min < 100 * dt_resample:
        raise ValueError("Not enough overlap between GT and estimation for robust time alignment.")

    t_uniform = np.arange(t_min, t_max, dt_resample)
    sig_gt = interp1d(t_gt_mid, om_gt, kind="linear")(t_uniform)
    sig_est = interp1d(t_est_mid, om_est, kind="linear")(t_uniform)
    sig_gt -= np.mean(sig_gt)
    sig_est -= np.mean(sig_est)
    if (not np.all(np.isfinite(sig_gt))) or (not np.all(np.isfinite(sig_est))):
        raise ValueError(
            "Step-1 signals contain non-finite values. Check timestamps for duplicates/non-monotonic samples."
        )

    corr = correlate(sig_est, sig_gt, mode="full")
    if not np.all(np.isfinite(corr)):
        raise ValueError(
            "Cross-correlation contains non-finite values (possible invalid timestamp deltas or signal values)."
        )
    lags = np.arange(-len(sig_gt) + 1, len(sig_est))
    offsets_s = lags.astype(float) * dt_resample

    search_mask = np.ones_like(offsets_s, dtype=bool)
    if offset_search_window_s > 0.0:
        search_mask = np.abs(offsets_s) <= float(offset_search_window_s)
        if not np.any(search_mask):
            raise ValueError("Offset search window has no valid lag candidates. Increase --offset-search-window-s.")

    search_indices = np.flatnonzero(search_mask)
    peak_idx = int(search_indices[int(np.argmax(corr[search_indices]))])
    calculated_offset = float(offsets_s[peak_idx])

    evo_match_max_diff_s = float(getattr(args, "t_max_diff", 0.02))

    def _count_matches_for_offset(offset_s: float) -> dict[str, object]:
        return _offset_match_diagnostics(
            t_gt,
            t_est,
            max_diff_s=evo_match_max_diff_s,
            offset_est_s=-float(offset_s),
        )

    match_diag = _count_matches_for_offset(calculated_offset)
    match_count = int(match_diag["match_count"])
    match_ratio_global = float(match_diag["ratio_global"])
    match_ratio_overlap = float(match_diag["ratio_overlap"])
    match_ratio_gate = float(match_diag["ratio_gate"])
    overlap_pair_cap = int(match_diag["pair_cap_overlap"])
    overlap_gate_min_pairs = int(match_diag["overlap_gate_min_pairs"])
    fallback_used = 0.0
    if match_ratio_gate < offset_min_match_ratio:
        zero_window_s = 1.0
        zero_mask = np.abs(offsets_s) <= zero_window_s
        if offset_search_window_s > 0.0:
            zero_mask &= search_mask

        if np.any(zero_mask):
            zero_indices = np.flatnonzero(zero_mask)
            zero_peak_idx = int(zero_indices[int(np.argmax(corr[zero_indices]))])
            fallback_offset = float(offsets_s[zero_peak_idx])
        else:
            zero_peak_idx = int(np.argmin(np.abs(offsets_s)))
            fallback_offset = float(offsets_s[zero_peak_idx])

        fallback_diag = _count_matches_for_offset(fallback_offset)
        fallback_count = int(fallback_diag["match_count"])
        fallback_ratio_global = float(fallback_diag["ratio_global"])
        fallback_ratio_overlap = float(fallback_diag["ratio_overlap"])
        fallback_ratio_gate = float(fallback_diag["ratio_gate"])
        if fallback_ratio_gate >= offset_min_match_ratio:
            calculated_offset = fallback_offset
            peak_idx = zero_peak_idx
            match_count = fallback_count
            match_ratio_global = fallback_ratio_global
            match_ratio_overlap = fallback_ratio_overlap
            match_ratio_gate = fallback_ratio_gate
            overlap_pair_cap = int(fallback_diag["pair_cap_overlap"])
            overlap_gate_min_pairs = int(fallback_diag["overlap_gate_min_pairs"])
            fallback_used = 1.0
        else:
            raise ValueError(
                "Unreliable step-1 offset estimate: "
                f"best_match_ratio_global={match_ratio_global:.3f}, "
                f"best_match_ratio_overlap={match_ratio_overlap:.3f}, "
                f"best_match_ratio_gate={match_ratio_gate:.3f}, "
                f"fallback_match_ratio_global={fallback_ratio_global:.3f}, "
                f"fallback_match_ratio_overlap={fallback_ratio_overlap:.3f}, "
                f"fallback_match_ratio_gate={fallback_ratio_gate:.3f}, "
                f"required>={offset_min_match_ratio:.3f}. "
                "Check timestamp quality (duplicates/non-monotonic) or adjust offset search settings."
            )

    print(f"Calculated Time Offset: {calculated_offset:.4f} s")

    sig_est_shifted = interp1d(
        t_uniform - calculated_offset,
        sig_est,
        kind="linear",
        fill_value="extrapolate",
    )(t_uniform)

    corr_norm_peak = corr[peak_idx] / (np.linalg.norm(sig_gt) * np.linalg.norm(sig_est) + 1e-12)
    psr = compute_psr(corr, peak_idx, guard_bins=max(1, int(0.02 / dt_resample)))
    time_metrics = {
        "offset_est_s": calculated_offset,
        "offset_err_ms": np.nan,
        "xcorr_peak_normalized": corr_norm_peak,
        "xcorr_psr": psr,
        "omega_rmse_before": rmse(sig_est - sig_gt),
        "omega_rmse_after": rmse(sig_est_shifted - sig_gt),
        "offset_search_window_s": offset_search_window_s,
        "offset_min_match_ratio": offset_min_match_ratio,
        "offset_fallback_used": fallback_used,
        "offset_match_ratio_global": match_ratio_global,
        "offset_match_ratio_overlap": match_ratio_overlap,
        "offset_match_ratio_gate": match_ratio_gate,
        "offset_overlap_pair_cap": float(overlap_pair_cap),
        "offset_overlap_gate_min_pairs": float(overlap_gate_min_pairs),
    }
    # Evo-compatible interpretation:
    # - evo applies t_offset to est timestamps before matching.
    # - our internal shift is t_est_sync = t_est - offset_est_s.
    #   Therefore, evo-equivalent t_offset is -offset_est_s.
    evo_t_offset_used_s = -calculated_offset
    match_ids_ref, match_ids_est = matching_time_indices(
        t_gt, t_est, max_diff=evo_match_max_diff_s, offset_2=evo_t_offset_used_s
    )
    time_metrics["evo_t_offset_used_s"] = evo_t_offset_used_s
    time_metrics["evo_match_max_diff_s"] = evo_match_max_diff_s
    time_metrics["evo_matches_equivalent"] = float(len(match_ids_ref))
    time_metrics["evo_matches_ratio_equivalent"] = float(match_ratio_global)
    time_metrics["evo_matches_ratio_overlap_aware"] = float(match_ratio_overlap)
    time_metrics["evo_matches_ratio_for_gate"] = float(match_ratio_gate)
    if args.synthetic:
        time_metrics["offset_err_ms"] = abs(calculated_offset - ARTIFICIAL_OFFSET) * 1e3

    time_metrics["omega_rmse_improve_pct"] = (
        (time_metrics["omega_rmse_before"] - time_metrics["omega_rmse_after"])
        / (time_metrics["omega_rmse_before"] + 1e-12)
        * 100.0
    )

    print_metric_block(
        "TIME ALIGNMENT METRICS",
        time_metrics,
        unit_map={
            "offset_est_s": "s",
            "offset_err_ms": "ms",
            "omega_rmse_before": "rad/s",
            "omega_rmse_after": "rad/s",
            "omega_rmse_improve_pct": "%",
            "evo_t_offset_used_s": "s",
            "evo_match_max_diff_s": "s",
        },
    )

    fig_corr, ax_corr = plt.subplots(figsize=(12, 4))
    lag_times = lags * dt_resample
    ax_corr.set_title("Step 1: Cross-Correlation vs Lag")
    ax_corr.plot(lag_times, corr, color="purple", linewidth=1.2)
    ax_corr.axvline(
        calculated_offset,
        color="red",
        linestyle="--",
        label=f"Estimated offset: {calculated_offset:.4f}s",
    )
    ax_corr.set_xlabel("Lag (s)")
    ax_corr.set_ylabel("Correlation")
    ax_corr.grid(True, linestyle=":", alpha=0.5)
    ax_corr.legend(loc="upper right")
    fig_corr.tight_layout()
    fig_corr_path = plots_dir / "step1_cross_correlation.png"
    fig_corr.savefig(fig_corr_path, dpi=200, bbox_inches="tight")

    fig1, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(12, 8))

    title_offset = f"Known offset: {ARTIFICIAL_OFFSET}s" if args.synthetic else "Unknown offset"
    ax_b.set_title(f"Step 1: Full Sequence BEFORE Alignment ({title_offset})")
    ax_b.plot(t_uniform, sig_gt, label="GT Omega", color="green", alpha=0.6)
    ax_b.plot(t_uniform, sig_est, "r--", label="Estimation Omega", alpha=0.6)
    ax_b.set_ylabel("Omega Norm")
    ax_b.legend(loc="upper right")
    ax_b.grid(True, linestyle=":", alpha=0.5)

    ax_a.set_title(f"Step 1: Full Sequence AFTER Alignment (Calculated: {calculated_offset:.4f}s)")
    ax_a.plot(t_uniform, sig_gt, label="GT Omega", color="green", alpha=0.6)
    ax_a.plot(t_uniform - calculated_offset, sig_est, "b--", label="Estimation Omega (Corrected)", alpha=0.8)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Omega Norm")
    ax_a.legend(loc="upper right")
    ax_a.grid(True, linestyle=":", alpha=0.5)

    view_window_s = 40.0
    x0 = t_uniform[0]
    x1 = min(t_uniform[-1], x0 + view_window_s)
    ax_b.set_xlim(x0, x1)
    ax_a.set_xlim(x0, x1)

    plt.tight_layout()
    fig1_path = plots_dir / "step1_time_alignment.png"
    fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")

    t_est_sync = t_est - calculated_offset
    interp_p = interp1d(t_est_sync, pos_est, axis=0, fill_value="extrapolate")
    pr_sync = interp_p(t_gt)
    if args.quat_interp == "slerp":
        qr_sync = interpolate_quat_slerp(t_est_sync, quat_est, t_gt)
    else:
        qr_sync = interpolate_quat_linear(t_est_sync, quat_est, t_gt)

    pos_gt_solve = np.asarray(pos_gt, dtype=float)
    quat_gt_solve = np.asarray(quat_gt, dtype=float)
    pr_solve = np.asarray(pr_sync, dtype=float)
    qr_solve = np.asarray(qr_sync, dtype=float)

    print("\n--- STEP 2: SOLVING EXTRINSICS ---")
    R_calc = solve_extrinsic_rotation(quat_gt_solve, qr_solve)
    t_calc = solve_extrinsic_translation(pos_gt_solve, quat_gt_solve, pr_solve, qr_solve, R_calc)
    print(f"Calculated Extrinsic Rotation Matrix:\n{np.round(R_calc, 4)}")
    print(f"Calculated Translation: {np.round(t_calc, 4)} m")

    pr_corrected = np.zeros_like(pr_sync)
    Rr_mats = R.from_quat(qr_sync).as_matrix()
    for i in range(len(pr_sync)):
        pr_corrected[i] = pr_sync[i] - (Rr_mats[i] @ R_calc.T) @ t_calc

    print("\n--- STEP 3: WORLD ALIGNMENT ---")
    pr_corrected_solve = np.zeros_like(pr_solve)
    Rr_mats_solve = R.from_quat(qr_solve).as_matrix()
    for i in range(len(pr_solve)):
        pr_corrected_solve[i] = pr_solve[i] - (Rr_mats_solve[i] @ R_calc.T) @ t_calc

    Rw_calc, tw_calc = solve_world_alignment(pr_corrected_solve, pos_gt_solve)
    pred_step3_pairs = (Rw_calc @ pr_corrected_solve.T).T + tw_calc
    selected_rmse_m = float(np.sqrt(np.mean(np.sum((pred_step3_pairs - pos_gt_solve) ** 2, axis=1))))
    pr_final = (Rw_calc @ pr_corrected.T).T + tw_calc

    R_step2_mats = np.einsum("nij,jk->nik", Rr_mats, R_calc.T)
    q_step2 = normalize_quat_array(R.from_matrix(R_step2_mats).as_quat())
    q_step3 = normalize_quat_array((R.from_matrix(Rw_calc) * R.from_quat(q_step2)).as_quat())

    pr_final_global = np.asarray(pr_final, dtype=float).copy()

    step3_choice = {
        "step3_rmse_selected_m": float(selected_rmse_m),
    }

    print(f"Calculated World Rotation Matrix:\n{np.round(Rw_calc, 4)}")
    print(f"Calculated World Translation: {np.round(tw_calc, 4)} m")
    print(
        f"Step3 selected_rmse={selected_rmse_m:.6f} m"
    )

    if args.synthetic:
        rot_ext_err_deg = float(np.degrees(R.from_matrix(R_calc.T @ R_ext_true).magnitude()))
        t_ext_err = float(np.linalg.norm(t_calc - t_ext_true))
        if np.all(np.isfinite(Rw_calc)) and np.all(np.isfinite(tw_calc)):
            rot_world_err_deg = float(np.degrees(R.from_matrix(Rw_calc.T @ Rw_true).magnitude()))
            t_world_err = float(np.linalg.norm(tw_calc - tw_true))
        else:
            rot_world_err_deg = float("nan")
            t_world_err = float("nan")
        print("\n--- SANITY CHECK (vs Injected Truth) ---")
        print(f"Extrinsic rotation error: {rot_ext_err_deg:.4f} deg")
        print(f"Extrinsic translation error: {t_ext_err:.4f} m")
        print(f"World rotation error: {rot_world_err_deg:.4f} deg")
        print(f"World translation error: {t_world_err:.4f} m")
        sanity_metrics = {
            "extrinsic_rotation_error_deg": rot_ext_err_deg,
            "extrinsic_translation_error_m": t_ext_err,
            "world_rotation_error_deg": rot_world_err_deg,
            "world_translation_error_m": t_world_err,
        }

    Rv_rot = R.from_quat(quat_gt_solve)
    Rr_rot = R.from_quat(qr_solve)
    dA = Rv_rot[:-1].inv() * Rv_rot[1:]
    dB = Rr_rot[:-1].inv() * Rr_rot[1:]
    Rext_rot = R.from_matrix(R_calc)
    dA_pred = Rext_rot * dB * Rext_rot.inv()
    rot_res_deg = np.degrees((dA.inv() * dA_pred).magnitude())

    C_mat, d_vec, num_constraints = build_translation_system(
        pos_gt_solve, quat_gt_solve, pr_solve, qr_solve, R_calc
    )
    trans_res = C_mat @ t_calc - d_vec
    trans_res_norm = np.linalg.norm(trans_res.reshape(-1, 3), axis=1)

    step2_metrics = {
        "rot_res_mean_deg": np.mean(rot_res_deg),
        "rot_res_median_deg": np.median(rot_res_deg),
        "rot_res_p95_deg": np.percentile(rot_res_deg, 95),
        "trans_eq_rmse_m": rmse(trans_res_norm),
        "trans_eq_p95_m": np.percentile(trans_res_norm, 95),
        "translation_system_cond": np.linalg.cond(C_mat),
        "translation_constraints": float(num_constraints),
    }

    print_metric_block(
        "STEP 2 RESIDUAL METRICS",
        step2_metrics,
        unit_map={
            "rot_res_mean_deg": "deg",
            "rot_res_median_deg": "deg",
            "rot_res_p95_deg": "deg",
            "trans_eq_rmse_m": "m",
            "trans_eq_p95_m": "m",
        },
    )

    raw_err = np.linalg.norm(pr_sync - pos_gt, axis=1)
    step2_err = np.linalg.norm(pr_corrected - pos_gt, axis=1)
    step3_err = np.linalg.norm(pr_final - pos_gt, axis=1)
    raw_stats = summarize_abs_errors(raw_err)
    step2_stats = summarize_abs_errors(step2_err)
    step3_stats = summarize_abs_errors(step3_err)

    traj_metrics = {
        "ate_rmse_raw_m": raw_stats["rmse"],
        "ate_rmse_step2_m": step2_stats["rmse"],
        "ate_rmse_step3_m": step3_stats["rmse"],
        "ate_p95_raw_m": raw_stats["p95"],
        "ate_p95_step2_m": step2_stats["p95"],
        "ate_p95_step3_m": step3_stats["p95"],
        "ate_rmse_improve_raw_to_step3_pct": (
            (raw_stats["rmse"] - step3_stats["rmse"]) / (raw_stats["rmse"] + 1e-12) * 100.0
        ),
        "ate_rmse_improve_step2_to_step3_pct": (
            (step2_stats["rmse"] - step3_stats["rmse"]) / (step2_stats["rmse"] + 1e-12) * 100.0
        ),
    }

    print_metric_block(
        "TRAJECTORY METRICS",
        traj_metrics,
        unit_map={
            "ate_rmse_raw_m": "m",
            "ate_rmse_step2_m": "m",
            "ate_rmse_step3_m": "m",
            "ate_p95_raw_m": "m",
            "ate_p95_step2_m": "m",
            "ate_p95_step3_m": "m",
            "ate_rmse_improve_raw_to_step3_pct": "%",
            "ate_rmse_improve_step2_to_step3_pct": "%",
        },
    )

    alignment_quality = _compute_alignment_quality(
        t_ref=t_gt,
        pos_ref=pos_gt,
        pos_step3=pr_final,
        raw_rmse_m=raw_stats["rmse"],
        step3_rmse_m=step3_stats["rmse"],
        segment_duration_s=quality_segment_duration_s,
        overlap_ratio=quality_segment_overlap_ratio,
        min_samples=quality_min_segment_samples,
        good_rmse_m=quality_good_rmse_m,
        partial_rmse_m=quality_partial_rmse_m,
        good_seg_cv=quality_good_segment_cv,
        good_heading_p90_deg=quality_good_heading_p90_deg,
        partial_min_improve_pct=quality_partial_min_improve_pct,
    )
    quality_label = str(alignment_quality.pop("_quality_label", "unknown"))
    print("\n--- ALIGNMENT QUALITY ---")
    print(f"quality_label: {quality_label}")
    print_metric_block(
        "ALIGNMENT QUALITY METRICS",
        alignment_quality,
        unit_map={
            "step3_rmse_m": "m",
            "raw_to_step3_improve_pct": "%",
            "segment_rmse_mean_m": "m",
            "segment_rmse_std_m": "m",
            "heading_median_deg": "deg",
            "heading_p90_deg": "deg",
        },
    )

    rigid_check_max_path_ratio = float(getattr(args, "rigid_check_max_path_ratio", 3.0))
    rigid_check_max_bbox_ratio = float(getattr(args, "rigid_check_max_bbox_ratio", 3.0))
    rigid_check_max_global_local_ratio = float(getattr(args, "rigid_check_max_global_local_ratio", 6.0))
    rigid_check_max_sim3_gain_ratio = float(getattr(args, "rigid_check_max_sim3_gain_ratio", 0.3))

    rigid_alignability = _compute_rigid_alignability(
        t_ref=t_gt,
        pos_ref=pos_gt,
        pos_step2=pr_corrected,
        pos_step3=pr_final_global,
        segment_duration_s=quality_segment_duration_s,
        overlap_ratio=quality_segment_overlap_ratio,
        min_samples=quality_min_segment_samples,
        max_path_ratio=rigid_check_max_path_ratio,
        max_bbox_ratio=rigid_check_max_bbox_ratio,
        max_global_local_ratio=rigid_check_max_global_local_ratio,
        max_sim3_gain_ratio=rigid_check_max_sim3_gain_ratio,
    )
    rigid_alignability_label = str(rigid_alignability.pop("_rigid_alignability_label", "unknown"))
    rigid_alignability_reasons = str(rigid_alignability.pop("_rigid_alignability_reasons", ""))

    user_alert = _build_user_alert(
        time_metrics=time_metrics,
        traj_metrics=traj_metrics,
        quality_label=quality_label,
        rigid_label=rigid_alignability_label,
        rigid_reasons=rigid_alignability_reasons,
    )
    alert_level = str(user_alert.pop("_alert_level", "ok"))
    alert_message = str(user_alert.pop("_alert_message", ""))
    alert_reasons = str(user_alert.pop("_alert_reasons", ""))

    print("\n--- USER ALERT ---")
    print(f"alert_level: {alert_level}")
    if alert_message:
        print(f"alert_message: {alert_message}")
    if alert_reasons:
        print(f"alert_reasons: {alert_reasons}")

    piecewise_diag, piecewise_detail = _compute_piecewise_diagnostics(
        t_ref=t_gt,
        pos_ref=pos_gt,
        pos_step2=pr_corrected,
        pos_step3=pr_final_global,
        segment_duration_s=quality_segment_duration_s,
        overlap_ratio=quality_segment_overlap_ratio,
        min_samples=quality_min_segment_samples,
    )

    seg_t = np.asarray(piecewise_detail["segment_mid_s"], dtype=float)
    seg_g = np.asarray(piecewise_detail["segment_global_rmse_m"], dtype=float)
    seg_l = np.asarray(piecewise_detail["segment_local_rmse_m"], dtype=float)
    valid_pw = np.isfinite(seg_t) & np.isfinite(seg_g) & np.isfinite(seg_l)
    if np.any(valid_pw):
        fig_pw, ax_pw = plt.subplots(figsize=(12, 4.2))
        ax_pw.set_title("Piecewise Diagnostics: Global vs Local Segment RMSE")
        ax_pw.plot(seg_t[valid_pw], seg_g[valid_pw], "-o", markersize=3.0, linewidth=1.2, label="Global single-SE3 RMSE")
        ax_pw.plot(seg_t[valid_pw], seg_l[valid_pw], "-o", markersize=3.0, linewidth=1.2, label="Local per-segment SE3 RMSE")
        ax_pw.set_xlabel("Time from start (s)")
        ax_pw.set_ylabel("Segment RMSE (m)")
        ax_pw.grid(True, linestyle=":", alpha=0.5)
        ax_pw.legend(loc="upper right")
        fig_pw.tight_layout()
        fig_pw_path = plots_dir / "piecewise_segment_rmse.png"
        fig_pw.savefig(fig_pw_path, dpi=220, bbox_inches="tight")

    stage_trajs = {
        "raw": {"pos": pr_sync, "quat": qr_sync},
        "step2": {"pos": pr_corrected, "quat": q_step2},
        "step3": {"pos": pr_final, "quat": q_step3},
    }
    eval_align_mode = str(getattr(args, "eval_align", "none"))
    eval_n_to_align = int(getattr(args, "eval_n_to_align", -1))
    eval_project_to_plane = str(getattr(args, "eval_project_to_plane", "none"))

    ape_metrics_by_stage = {}
    rpe_metrics_by_stage = {}
    seconds_from_start = np.asarray(t_gt, dtype=float) - float(t_gt[0])
    distances_from_start = _cum_distance(pos_gt)
    for stage_name, stage_data in stage_trajs.items():
        eval_pos, eval_quat = _align_for_eval(
            pos_ref=pos_gt,
            quat_ref=quat_gt,
            pos_est=stage_data["pos"],
            quat_est=stage_data["quat"],
            mode=eval_align_mode,
            n_to_align=eval_n_to_align,
        )
        ref_eval_pos, ref_eval_quat = _project_to_plane(
            pos_gt, quat_gt, plane=eval_project_to_plane
        )
        est_eval_pos, est_eval_quat = _project_to_plane(
            eval_pos, eval_quat, plane=eval_project_to_plane
        )
        ape_metrics_by_stage[stage_name] = compute_ape_evo_style(
            pos_ref=ref_eval_pos,
            quat_ref=ref_eval_quat,
            pos_est=est_eval_pos,
            quat_est=est_eval_quat,
            include_raw=True,
        )
        rpe_metrics_by_stage[stage_name] = compute_rpe_evo_style(
            pos_ref=ref_eval_pos,
            quat_ref=ref_eval_quat,
            pos_est=est_eval_pos,
            quat_est=est_eval_quat,
            delta=args.rpe_delta,
            delta_unit=args.rpe_delta_unit,
            rel_delta_tol=args.rpe_delta_tol,
            all_pairs=args.rpe_all_pairs,
            pairs_from_reference=args.rpe_pairs_from_reference,
            include_raw=True,
        )
        ape_metrics_by_stage[stage_name]["_x_axis"]["seconds_from_start"] = seconds_from_start
        ape_metrics_by_stage[stage_name]["_x_axis"]["distances_from_start"] = distances_from_start
        delta_ids = rpe_metrics_by_stage[stage_name]["_x_axis"]["delta_ids"].astype(int)
        valid = (delta_ids >= 0) & (delta_ids < seconds_from_start.size)
        rpe_metrics_by_stage[stage_name]["_x_axis"]["seconds_from_start"] = seconds_from_start[delta_ids[valid]]
        rpe_metrics_by_stage[stage_name]["_x_axis"]["distances_from_start"] = distances_from_start[delta_ids[valid]]

    pose_metrics = {
        "ape": ape_metrics_by_stage,
        "rpe": rpe_metrics_by_stage,
        "rpe_config": {
            "delta": args.rpe_delta,
            "delta_unit": args.rpe_delta_unit,
            "delta_tol": args.rpe_delta_tol,
            "all_pairs": bool(args.rpe_all_pairs),
            "pairs_from_reference": bool(args.rpe_pairs_from_reference),
        },
        "eval_config": {
            "t_max_diff": float(getattr(args, "t_max_diff", 0.02)),
            "t_offset": float(getattr(args, "t_offset", 0.0)),
            "t_start": None if getattr(args, "t_start", None) is None else float(args.t_start),
            "t_end": None if getattr(args, "t_end", None) is None else float(args.t_end),
            "align": eval_align_mode,
            "n_to_align": eval_n_to_align,
            "project_to_plane": eval_project_to_plane,
        },
    }

    stage_order = ["raw", "step2", "step3"]

    print("\n--- METRICS (APE/RPE) ---")
    ape_pose_relation = normalize_pose_relation("ape", getattr(args, "ape_pose_relation", "trans_part"))
    rpe_pose_relation = normalize_pose_relation("rpe", getattr(args, "rpe_pose_relation", "trans_part"))
    for stage_name in stage_order:
        ape_t = pose_metrics["ape"][stage_name][ape_pose_relation]["rmse"]
        ape_r = pose_metrics["ape"][stage_name]["rotation_angle_deg"]["rmse"]
        rpe_t = pose_metrics["rpe"][stage_name][rpe_pose_relation]["rmse"]
        rpe_r = pose_metrics["rpe"][stage_name]["rotation_angle_deg"]["rmse"]
        pairs = pose_metrics["rpe"][stage_name]["pair_count"]
        print(
            f"{stage_name}: "
            f"APE_{ape_pose_relation}_rmse={ape_t:.6f}, "
            f"APE_rot_rmse={ape_r:.6f} deg, "
            f"RPE_{rpe_pose_relation}_rmse={rpe_t:.6f}, "
            f"RPE_rot_rmse={rpe_r:.6f} deg, "
            f"pairs={pairs}"
        )

    fig2 = plt.figure(figsize=(18, 5.8))
    stage_titles = {
        "raw": "Raw (After Step-1 Sync)",
        "step2": "Sensor Fixed (Step 2)",
        "step3": "Aligned (Step 3)",
    }
    stage_pos = {
        "raw": np.asarray(pr_sync, dtype=float),
        "step2": np.asarray(pr_corrected, dtype=float),
        "step3": np.asarray(pr_final, dtype=float),
    }
    for i, stage_name in enumerate(stage_order):
        ax = fig2.add_subplot(131 + i, projection="3d")
        stage_ref = np.asarray(pos_gt, dtype=float)
        stage_est = np.asarray(stage_pos[stage_name], dtype=float)
        stage_err = np.linalg.norm(stage_est - stage_ref, axis=1)
        _plot_alignment_map(
            fig2,
            ax,
            pos_ref=stage_ref,
            pos_est=stage_est,
            errors_m=stage_err,
            title=(
                f"{stage_titles[stage_name]}\n"
                f"rmse={float(np.sqrt(np.mean(stage_err**2))):.6f} m"
            ),
        )
    fig2.tight_layout()
    fig2_path = plots_dir / "step23_trajectory_alignment_3d.png"
    fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")

    fig_step3_map_path = None
    step3_err_subset = np.linalg.norm(np.asarray(pr_final, dtype=float) - np.asarray(pos_gt, dtype=float), axis=1)
    fig_step3_map = plt.figure(figsize=(8.4, 6.8))
    ax_step3_map = fig_step3_map.add_subplot(111, projection="3d")
    _plot_alignment_map(
        fig_step3_map,
        ax_step3_map,
        pos_ref=np.asarray(pos_gt, dtype=float),
        pos_est=np.asarray(pr_final, dtype=float),
        errors_m=step3_err_subset,
        title=f"Step3 Alignment Map\nrmse={float(np.sqrt(np.mean(step3_err_subset**2))):.6f} m",
    )
    fig_step3_map.tight_layout()
    fig_step3_map_path = plots_dir / "step3_alignment_map.png"
    fig_step3_map.savefig(fig_step3_map_path, dpi=220, bbox_inches="tight")

    rerun_info = {
        "enabled": "false",
        "status": "disabled",
        "message": "Use --rerun to enable.",
    }
    if bool(getattr(args, "rerun", False)):
        rerun_info = log_alignment_to_rerun(
            run_dir=run_dir,
            gt_xyz=pos_gt,
            raw_xyz=pr_sync,
            step2_xyz=pr_corrected,
            step3_xyz=pr_final,
            timestamps_s=t_gt,
            gt_quat=quat_gt,
            raw_quat=qr_sync,
            step2_quat=q_step2,
            step3_quat=q_step3,
            raw_error_m=raw_err,
            step2_error_m=step2_err,
            step3_error_m=step3_err,
            app_id="epa_alignment",
            spawn=not bool(getattr(args, "rerun_no_spawn", False)),
            stride=int(getattr(args, "rerun_stride", 20)),
            motion_stride=int(getattr(args, "rerun_motion_stride", 5)),
        )
        print(f"Rerun status: {rerun_info['status']} - {rerun_info['message']}")

    metrics_payload = {
        "time_alignment": time_metrics,
        "step2_residuals": step2_metrics,
        "trajectory": traj_metrics,
        "step3_selection": {
            "step3_rmse_selected_m": float(step3_choice["step3_rmse_selected_m"]),
        },
        "user_alert": user_alert,
        "alignment_quality": alignment_quality,
        "rigid_alignability": rigid_alignability,
        "piecewise_diagnostics": piecewise_diag,
        "pose_metrics": pose_metrics,
        "sanity_check": sanity_metrics,
        "estimated_params": {
            "R_ext": R_calc,
            "t_ext": t_calc,
            "R_world": Rw_calc,
            "t_world": tw_calc,
            "offset_s": calculated_offset,
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "gt_path": str(gt_path),
            "est_path": str(args.est_path) if args.est_path else "",
            "gt_format": str(gt_format),
            "est_format": str(getattr(args, "est_format", "auto")),
            "gt_topic": str(gt_topic),
            "est_topic": str(getattr(args, "est_topic", "")),
            "quat_interp": args.quat_interp,
            "t_offset": float(getattr(args, "t_offset", 0.0)),
            "t_max_diff": float(getattr(args, "t_max_diff", 0.02)),
            "t_start": None if getattr(args, "t_start", None) is None else float(args.t_start),
            "t_end": None if getattr(args, "t_end", None) is None else float(args.t_end),
            "ape_pose_relation": str(getattr(args, "ape_pose_relation", "trans_part")),
            "rpe_pose_relation": str(getattr(args, "rpe_pose_relation", "trans_part")),
            "eval_align": eval_align_mode,
            "eval_n_to_align": eval_n_to_align,
            "eval_project_to_plane": eval_project_to_plane,
            "alignment_quality_label": quality_label,
            "rigid_alignability_label": rigid_alignability_label,
            "rigid_alignability_reasons": rigid_alignability_reasons,
            "user_alert_level": alert_level,
            "user_alert_message": alert_message,
            "user_alert_reasons": alert_reasons,
            "rigid_check_max_path_ratio": float(rigid_check_max_path_ratio),
            "rigid_check_max_bbox_ratio": float(rigid_check_max_bbox_ratio),
            "rigid_check_max_global_local_ratio": float(rigid_check_max_global_local_ratio),
            "rigid_check_max_sim3_gain_ratio": float(rigid_check_max_sim3_gain_ratio),
            "rerun": rerun_info,
            "output_dir": str(run_dir),
        },
    }
    save_results_arg = str(getattr(args, "save_results", "") or "").strip()
    bundle_path = None
    if save_results_arg:
        bundle = Path(save_results_arg).expanduser()
        if not bundle.is_absolute():
            bundle = (Path.cwd() / bundle).resolve()
        else:
            bundle = bundle.resolve()
        bundle_path = bundle
        metrics_payload["metadata"]["result_bundle"] = str(bundle_path)

    plot_files = []
    plot_enabled = bool(getattr(args, "plot", True))
    if plot_enabled:
        ape_plot_rel = str(getattr(args, "plot_ape_relation", "translation_part"))
        rpe_plot_rel = str(getattr(args, "plot_rpe_relation", "translation_part"))
        plot_rel_ape = normalize_pose_relation("ape", ape_plot_rel)
        plot_rel_rpe = normalize_pose_relation("rpe", rpe_plot_rel)
        plot_files = generate_metric_plots(
            metrics_payload=metrics_payload,
            out_dir=plots_dir,
            ape_relation=plot_rel_ape,
            rpe_relation=plot_rel_rpe,
            x_dimension=str(getattr(args, "plot_x_dimension", "seconds")),
        )
        ape_slug = str(plot_rel_ape).replace("/", "_").replace(" ", "_")
        ape_stage_raw = generate_ape_stage_raw_plot(
            metrics_payload=metrics_payload,
            out_dir=plots_dir,
            ape_relation=plot_rel_ape,
            stage="step3",
            x_dimension=str(getattr(args, "plot_x_dimension", "seconds")),
            file_name=f"ape_{ape_slug}_se3_raw.png",
        )
        if ape_stage_raw is not None:
            plot_files.append(ape_stage_raw)
        plot_meta = {
            "enabled": True,
            "files": [str(p) for p in plot_files],
            "ape_relation": plot_rel_ape,
            "rpe_relation": plot_rel_rpe,
            "x_dimension": str(getattr(args, "plot_x_dimension", "seconds")),
        }
        metrics_payload["metadata"]["plot"] = plot_meta
    else:
        plot_meta = {"enabled": False, "files": []}
        metrics_payload["metadata"]["plot"] = plot_meta

    core_plot_names = {
        "piecewise_segment_rmse.png",
        "step1_cross_correlation.png",
        "step1_time_alignment.png",
        "step23_trajectory_alignment_3d.png",
    }
    retained_plot_names = set(core_plot_names)
    retained_plot_names.update(Path(p).name for p in plot_files)
    retained_plot_paths = []
    for plot_path in sorted(plots_dir.glob("*")):
        if not plot_path.is_file():
            continue
        if plot_path.name in retained_plot_names:
            retained_plot_paths.append(plot_path)
            continue
        plot_path.unlink(missing_ok=True)
    plot_files = [p for p in plot_files if p.name in retained_plot_names and p.exists()]
    metrics_payload["metadata"]["plot"]["files"] = [str(p) for p in retained_plot_paths]

    save_metrics(run_dir, metrics_payload)
    report_zh_path, report_en_path = write_run_reports(run_dir, metrics_payload)
    if bundle_path is not None:
        write_result_bundle(run_dir, metrics_payload, bundle_path)

    print("\n--- OUTPUT FILES ---")
    print(f"Saved figure: {fig_corr_path}")
    print(f"Saved figure: {fig1_path}")
    print(f"Saved figure: {fig2_path}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print(f"Saved metrics: {run_dir / 'metrics_summary.csv'}")
    print(f"Saved report: {report_zh_path}")
    print(f"Saved report: {report_en_path}")
    if bundle_path is not None:
        print(f"Saved results: {bundle_path}")

    plt.close(fig_corr)
    plt.close(fig1)
    plt.close(fig2)
