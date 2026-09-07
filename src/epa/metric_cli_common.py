from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.alignment.modes import resolve_metric_eval_align_mode
from epa.core.calibration import (
    InsufficientRotationExcitationError,
    solve_extrinsic_rotation_multibaseline,
    solve_rotation_first_alignment,
)
from epa.core.sim3 import solve_anchor_sim3, solve_epa_sim3, solve_epa_sim3_v1, solve_epa_sim3_v2, solve_epica_sim3_variant
from epa.io.trajectory import load_estimation_trajectory, load_reference_trajectory
from epa.core.math_utils import normalize_quat_array
from epa.core.time_alignment import matching_time_indices


@dataclass
class MetricInputs:
    subcommand: str
    ref_name: str
    est_name: str
    t_ref: np.ndarray
    pos_ref: np.ndarray
    quat_ref: np.ndarray
    t_est: np.ndarray
    pos_est: np.ndarray
    quat_est: np.ndarray
    full_ref_t: np.ndarray | None = None
    full_ref_pos: np.ndarray | None = None
    full_ref_quat: np.ndarray | None = None


@dataclass
class RosMapSpec:
    image_path: Path
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float


def default_out_dir(kind: str, cwd: Path) -> Path:
    root = cwd / "outputs" / kind
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = root / stamp
    idx = 1
    while out.exists():
        out = root / f"{stamp}_{idx:02d}"
        idx += 1
    out.mkdir(parents=True, exist_ok=False)
    return out


def cum_distance(points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    if points.shape[0] == 0:
        return np.array([], dtype=float)
    d = np.zeros(points.shape[0], dtype=float)
    if points.shape[0] > 1:
        d[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    return d


def apply_time_window(
    tvals: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    t_start: float | None,
    t_end: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t_start is None and t_end is None:
        return tvals, pos, quat
    tvals = np.asarray(tvals, dtype=float).reshape(-1)
    mask = np.ones(tvals.shape[0], dtype=bool)
    if t_start is not None:
        mask &= tvals >= float(t_start)
    if t_end is not None:
        mask &= tvals <= float(t_end)
    if int(np.sum(mask)) < 2:
        raise ValueError("Time window kept fewer than 2 trajectory samples.")
    t_new = tvals[mask]
    t_new = t_new - float(t_new[0])
    return t_new, np.asarray(pos, dtype=float)[mask], np.asarray(quat, dtype=float)[mask]


def downsample_traj(
    t: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    num_poses: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(num_poses) < 1:
        raise ValueError("downsample must be >= 1")
    n = int(t.shape[0])
    if n <= int(num_poses):
        return t, pos, quat
    ids = np.linspace(0, n - 1, int(num_poses), dtype=int)
    return t[ids], pos[ids], quat[ids]


def motion_filter_traj(
    t: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    distance_threshold: float,
    angle_threshold_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(t.shape[0]) <= 1:
        return t, pos, quat
    dist_thr = float(distance_threshold)
    ang_thr = float(np.deg2rad(float(angle_threshold_deg)))
    ids = [0]
    accum_dist = 0.0
    accum_ang = 0.0
    rots = R.from_quat(np.asarray(quat, dtype=float))
    for i in range(1, int(t.shape[0])):
        accum_dist += float(np.linalg.norm(pos[i] - pos[i - 1]))
        accum_ang += float((rots[i - 1].inv() * rots[i]).magnitude())
        if accum_dist >= dist_thr or accum_ang >= ang_thr:
            ids.append(i)
            accum_dist = 0.0
            accum_ang = 0.0
    if ids[-1] != int(t.shape[0]) - 1:
        ids.append(int(t.shape[0]) - 1)
    ids_arr = np.asarray(ids, dtype=int)
    return t[ids_arr], pos[ids_arr], quat[ids_arr]


def sync_trajectories(
    t_ref: np.ndarray,
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    t_est: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    max_diff: float,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_ref, idx_est = matching_time_indices(
        t_ref, t_est, max_diff=float(max_diff), offset_2=float(offset)
    )
    if len(idx_ref) < 2:
        raise ValueError(
            f"Trajectory association produced only {len(idx_ref)} matches "
            f"(t_max_diff={max_diff}, t_offset={offset})."
        )
    i_ref = np.asarray(idx_ref, dtype=int)
    i_est = np.asarray(idx_est, dtype=int)
    t_ref_m = np.asarray(t_ref, dtype=float)[i_ref]
    t_ref_m = t_ref_m - float(t_ref_m[0])
    t_est_m = np.asarray(t_est, dtype=float)[i_est]
    t_est_m = t_est_m - float(t_est_m[0])
    return (
        t_ref_m,
        np.asarray(pos_ref, dtype=float)[i_ref],
        np.asarray(quat_ref, dtype=float)[i_ref],
        t_est_m,
        np.asarray(pos_est, dtype=float)[i_est],
        np.asarray(quat_est, dtype=float)[i_est],
    )


def resolve_eval_align_mode(args) -> str:
    explicit = str(getattr(args, "eval_align", "") or "").strip().lower()
    if explicit:
        return resolve_metric_eval_align_mode(explicit)
    public_mode = str(getattr(args, "mode", "") or "").strip().lower()
    if public_mode:
        return resolve_metric_eval_align_mode(public_mode)
    if bool(getattr(args, "align_origin", False)):
        return "origin"
    align = bool(getattr(args, "align", False))
    correct_scale = bool(getattr(args, "correct_scale", False))
    if align and correct_scale:
        return "sim3"
    if align:
        return "se3"
    if correct_scale:
        return "scale"
    return "none"


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


def _rot_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _yaw_only_transform(src_xyz, dst_xyz):
    src = np.asarray(src_xyz, dtype=float)
    dst = np.asarray(dst_xyz, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 2:
        raise ValueError("PosYaw alignment requires at least 2 paired 3D points.")

    src_xy = src[:, :2]
    dst_xy = dst[:, :2]
    mu_src_xy = np.mean(src_xy, axis=0)
    mu_dst_xy = np.mean(dst_xy, axis=0)
    src0 = src_xy - mu_src_xy
    dst0 = dst_xy - mu_dst_xy
    cov = dst0.T @ src0
    yaw = float(np.arctan2(cov[1, 0] - cov[0, 1], cov[0, 0] + cov[1, 1]))
    r_fit = _rot_z(yaw)
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    t_fit = mu_dst - r_fit @ mu_src
    return r_fit, t_fit, yaw


def sim3_scale_guard(
    scale: float,
    *,
    warning_min: float = 0.5,
    warning_max: float = 2.0,
    severe_min: float = 0.1,
    severe_max: float = 10.0,
) -> dict[str, object]:
    scale_f = float(scale)
    finite_positive = bool(np.isfinite(scale_f) and scale_f > 0.0)
    log10_abs = float(abs(np.log10(scale_f))) if finite_positive else float("nan")
    warning = bool(
        not finite_positive
        or scale_f < float(warning_min)
        or scale_f > float(warning_max)
    )
    severe = bool(
        not finite_positive
        or scale_f < float(severe_min)
        or scale_f > float(severe_max)
    )
    if severe:
        level = "severe"
        message = (
            "Sim3 estimated scale is outside the reliable range; "
            "Sim3-aligned RMSE/SR may be misleading."
        )
    elif warning:
        level = "warning"
        message = (
            "Sim3 estimated scale is outside the expected range; "
            "check whether scale correction is appropriate."
        )
    else:
        level = "ok"
        message = ""
    return {
        "sim3_scale": scale_f,
        "sim3_scale_log10_abs": log10_abs,
        "sim3_scale_warning": bool(warning),
        "sim3_scale_severe": bool(severe),
        "sim3_reliable": bool(not severe),
        "sim3_warning_level": level,
        "sim3_warning": message,
        "sim3_scale_warning_min": float(warning_min),
        "sim3_scale_warning_max": float(warning_max),
        "sim3_scale_severe_min": float(severe_min),
        "sim3_scale_severe_max": float(severe_max),
    }


def _sim3_reliable_from_info(sim3_info: dict[str, object], scale_info: dict[str, object]) -> bool:
    return bool(sim3_info.get("sim3_reliable", True) and scale_info.get("sim3_reliable", True))


def _finalize_sim3_alignment_info(
    info: dict[str, object],
    *,
    scale: float,
    r_fit: np.ndarray,
    t_fit: np.ndarray,
    sim3_info: dict[str, object],
    use_scale_reliability: bool = False,
) -> dict[str, object]:
    info["align_mode"] = "sim3"
    info["align_scale"] = float(scale)
    info["align_rotation_matrix"] = np.asarray(r_fit, dtype=float).tolist()
    info["align_translation"] = np.asarray(t_fit, dtype=float).reshape(3).tolist()
    info.update(sim3_info)
    scale_info = sim3_scale_guard(float(scale))
    scale_info["sim3_scale_reliable"] = bool(scale_info.get("sim3_reliable", True))
    scale_info["sim3_reliable"] = (
        _sim3_reliable_from_info(sim3_info, scale_info)
        if use_scale_reliability
        else bool(sim3_info.get("sim3_reliable", True))
    )
    if not scale_info["sim3_reliable"] and not str(scale_info.get("sim3_warning", "")):
        scale_info["sim3_warning"] = str(sim3_info.get("sim3_warning", ""))
    info.update(scale_info)
    return info


def _alignment_subset_errors(
    *,
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est_aligned: np.ndarray,
    quat_est_aligned: np.ndarray,
    idx: np.ndarray,
) -> tuple[float, float]:
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if idx.size == 0:
        return float("inf"), float("inf")
    p_ref = np.asarray(pos_ref, dtype=float)[idx]
    p_est = np.asarray(pos_est_aligned, dtype=float)[idx]
    q_ref = np.asarray(quat_ref, dtype=float)[idx]
    q_est = np.asarray(quat_est_aligned, dtype=float)[idx]
    pos_err = np.linalg.norm(p_ref - p_est, axis=1)
    rot_err = (R.from_quat(q_ref) * R.from_quat(q_est).inv()).magnitude()
    pos_rmse = float(np.sqrt(np.mean(np.square(pos_err))))
    rot_rmse = float(np.degrees(np.sqrt(np.mean(np.square(rot_err)))))
    return pos_rmse, rot_rmse


def _should_use_extrinsic_sim3_candidate(
    *,
    raw_pos_rmse: float,
    raw_rot_rmse_deg: float,
    corrected_pos_rmse: float,
    corrected_rot_rmse_deg: float,
) -> bool:
    values = (raw_pos_rmse, raw_rot_rmse_deg, corrected_pos_rmse, corrected_rot_rmse_deg)
    if not all(np.isfinite(v) for v in values):
        return False
    if raw_rot_rmse_deg < 5.0:
        return False
    orientation_much_better = corrected_rot_rmse_deg <= max(5.0, 0.5 * raw_rot_rmse_deg)
    position_not_worse = corrected_pos_rmse <= max(raw_pos_rmse * 1.10, raw_pos_rmse + 0.05)
    position_much_better = corrected_pos_rmse <= max(0.75 * raw_pos_rmse, raw_pos_rmse - 0.10)
    return bool(orientation_much_better and (position_not_worse or position_much_better))


def align_for_eval_with_info(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    mode: str,
    n_to_align: int = -1,
    *,
    t_ref: np.ndarray | None = None,
    align_indices: np.ndarray | None = None,
    disable_extrinsic_calibration: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    requested_mode = str(mode).lower()
    mode = resolve_metric_eval_align_mode(requested_mode, default="none")
    info: dict[str, object] = {
        "align_mode": mode,
        "n_to_align": int(n_to_align),
        "extrinsic_calibration_disabled": bool(disable_extrinsic_calibration),
    }
    if requested_mode != mode:
        info["requested_align_mode"] = requested_mode
    if mode == "none":
        return np.asarray(pos_est, dtype=float), np.asarray(quat_est, dtype=float), info

    n = pos_ref.shape[0]
    if align_indices is not None:
        idx = np.asarray(align_indices, dtype=int).reshape(-1)
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.size:
            idx = np.unique(idx)
    else:
        n_use = n if int(n_to_align) <= 0 else min(n, int(n_to_align))
        idx = np.arange(n_use, dtype=int)
    info["align_pair_count"] = int(idx.size)
    if align_indices is not None:
        info["align_index_count"] = int(idx.size)
        if idx.size:
            info["align_index_first"] = int(idx[0])
            info["align_index_last"] = int(idx[-1])
    if idx.size < 2:
        return np.asarray(pos_est, dtype=float), np.asarray(quat_est, dtype=float), info

    pref = np.asarray(pos_ref[idx], dtype=float)
    pest = np.asarray(pos_est[idx], dtype=float)
    q_est = np.asarray(quat_est, dtype=float)

    if mode == "se3-original":
        _, R_eval, t_eval = _umeyama_transform(pest, pref, with_scale=False)
        pos_new = (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        info["align_scale"] = 1.0
        info["align_rotation_matrix"] = np.asarray(R_eval, dtype=float).tolist()
        info["align_translation"] = np.asarray(t_eval, dtype=float).reshape(3).tolist()
        info["se3_original_solver"] = "position_only_umeyama"
        return pos_new, q_new, info

    if mode in {"se3", "se3r"}:
        R_eval, t_eval = solve_rotation_first_alignment(
            pest,
            pref,
            np.asarray(quat_est[idx], dtype=float),
            np.asarray(quat_ref[idx], dtype=float),
        )
        pos_new = (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        info.update(
            {
                "align_scale": 1.0,
                "align_rotation_matrix": np.asarray(R_eval, dtype=float).tolist(),
                "align_translation": np.asarray(t_eval, dtype=float).reshape(3).tolist(),
                "se3_solver": "orientation_chordal_mean_then_translation_mean",
                # Retained for consumers of the former public se3r mode.
                "se3r_solver": "orientation_chordal_mean_then_translation_mean",
            }
        )
        return pos_new, q_new, info

    if mode in {"posyaw", "epa_posyaw"}:
        R_eval, t_eval, yaw = _yaw_only_transform(pest, pref)
        pos_new = (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        info.update(
            {
                "align_mode": "posyaw",
                "align_scale": 1.0,
                "align_rotation_matrix": np.asarray(R_eval, dtype=float).tolist(),
                "align_translation": np.asarray(t_eval, dtype=float).reshape(3).tolist(),
                "posyaw_solver": "epa_yaw_only_umeyama",
                "posyaw_yaw_deg": float(np.degrees(yaw)),
                "posyaw_translation": [float(x) for x in np.asarray(t_eval, dtype=float)],
                "posyaw_pair_count": int(idx.size),
            }
        )
        return pos_new, q_new, info

    if mode == "ov_sim3":
        s_eval, R_eval, t_eval = _umeyama_transform(pest, pref, with_scale=True)
        pos_new = s_eval * (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        _finalize_sim3_alignment_info(
            info,
            scale=float(s_eval),
            r_fit=R_eval,
            t_fit=t_eval,
            sim3_info={"sim3_solver": "ov_position_only_umeyama"},
            use_scale_reliability=True,
        )
        info["align_mode"] = "ov_sim3"
        return pos_new, q_new, info

    if mode in {"epica_sim3", "epica_sim3_stable", "epica_sim3_joint", "epica_sim3_trimmed"}:
        s_eval, R_eval, t_eval, sim3_info = solve_epica_sim3_variant(
            pos_ref=pref,
            quat_ref=np.asarray(quat_ref[idx], dtype=float),
            pos_est=pest,
            quat_est=np.asarray(quat_est[idx], dtype=float),
            method=mode,
        )
        pos_new = s_eval * (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        info["align_mode"] = mode
        _finalize_sim3_alignment_info(
            info,
            scale=float(s_eval),
            r_fit=R_eval,
            t_fit=t_eval,
            sim3_info=sim3_info,
            use_scale_reliability=mode == "epica_sim3_stable",
        )
        info["align_mode"] = mode
        return pos_new, q_new, info

    if mode in {"sim3", "epa_sim3", "epa_sim3_v1", "epa_sim3_v2"}:
        solver_fn = {
            "sim3": solve_epa_sim3,
            "epa_sim3": solve_epa_sim3,
            "epa_sim3_v1": solve_epa_sim3_v1,
            "epa_sim3_v2": solve_epa_sim3_v2,
        }[mode]
        s_eval, R_eval, t_eval, sim3_info = solver_fn(
            pos_ref=pref,
            quat_ref=np.asarray(quat_ref[idx], dtype=float),
            pos_est=pest,
            quat_est=np.asarray(quat_est[idx], dtype=float),
            **({"timestamps_s": np.asarray(t_ref, dtype=float)[idx]} if t_ref is not None and mode in {"sim3", "epa_sim3", "epa_sim3_v2"} else {}),
        )
        pos_new = s_eval * (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        _finalize_sim3_alignment_info(
            info,
            scale=float(s_eval),
            r_fit=R_eval,
            t_fit=t_eval,
            sim3_info=sim3_info,
        )
        raw_pos_rmse, raw_rot_rmse = _alignment_subset_errors(
            pos_ref=pos_ref,
            quat_ref=quat_ref,
            pos_est_aligned=pos_new,
            quat_est_aligned=q_new,
            idx=idx,
        )
        info["sim3_extrinsic_rotation_correction_used"] = False
        info["sim3_raw_candidate_position_rmse_m"] = raw_pos_rmse
        info["sim3_raw_candidate_orientation_rmse_deg"] = raw_rot_rmse
        if mode in {"sim3", "epa_sim3"} and not bool(
            disable_extrinsic_calibration
        ):
            try:
                r_body, _, _, rotation_info = solve_extrinsic_rotation_multibaseline(
                    np.asarray(quat_ref[idx], dtype=float),
                    np.asarray(quat_est[idx], dtype=float),
                    timestamps_s=(
                        np.asarray(t_ref, dtype=float)[idx] if t_ref is not None else None
                    ),
                )
                info["sim3_extrinsic_rotation_solver"] = "multibaseline"
                info["sim3_extrinsic_rotation_info"] = rotation_info
                # Match the shared SE3 solver's observability gate: only use
                # an observable fit or a successfully constrained weak-axis fit.
                if not (
                    bool(rotation_info["observable"])
                    or bool(rotation_info.get("constraint_success", False))
                ):
                    raise InsufficientRotationExcitationError(
                        "Extrinsic rotation is unobservable and could not be constrained."
                    )
                q_est_body_corrected = normalize_quat_array(
                    R.from_matrix(
                        np.einsum(
                            "nij,jk->nik",
                            R.from_quat(np.asarray(quat_est, dtype=float)).as_matrix(),
                            np.asarray(r_body, dtype=float).T,
                        )
                    ).as_quat()
                )
                solver_kwargs = (
                    {"timestamps_s": np.asarray(t_ref, dtype=float)[idx]}
                    if t_ref is not None and mode in {"sim3", "epa_sim3"}
                    else {}
                )
                s_ext, r_ext_fit, t_ext_fit, ext_sim3_info = solver_fn(
                    pos_ref=pref,
                    quat_ref=np.asarray(quat_ref[idx], dtype=float),
                    pos_est=pest,
                    quat_est=np.asarray(q_est_body_corrected[idx], dtype=float),
                    **solver_kwargs,
                )
                pos_ext = s_ext * (r_ext_fit @ np.asarray(pos_est, dtype=float).T).T + t_ext_fit
                q_ext = normalize_quat_array(
                    (R.from_matrix(r_ext_fit) * R.from_quat(q_est_body_corrected)).as_quat()
                )
                ext_pos_rmse, ext_rot_rmse = _alignment_subset_errors(
                    pos_ref=pos_ref,
                    quat_ref=quat_ref,
                    pos_est_aligned=pos_ext,
                    quat_est_aligned=q_ext,
                    idx=idx,
                )
                info["sim3_extrinsic_candidate_position_rmse_m"] = ext_pos_rmse
                info["sim3_extrinsic_candidate_orientation_rmse_deg"] = ext_rot_rmse
                if _should_use_extrinsic_sim3_candidate(
                    raw_pos_rmse=raw_pos_rmse,
                    raw_rot_rmse_deg=raw_rot_rmse,
                    corrected_pos_rmse=ext_pos_rmse,
                    corrected_rot_rmse_deg=ext_rot_rmse,
                ):
                    ext_info: dict[str, object] = {
                        "align_mode": mode,
                        "n_to_align": int(n_to_align),
                        "align_pair_count": int(idx.size),
                        "sim3_extrinsic_rotation_correction_used": True,
                        "sim3_extrinsic_rotation_solver": "multibaseline",
                        "sim3_extrinsic_rotation_info": rotation_info,
                        "sim3_raw_candidate_position_rmse_m": raw_pos_rmse,
                        "sim3_raw_candidate_orientation_rmse_deg": raw_rot_rmse,
                        "sim3_extrinsic_candidate_position_rmse_m": ext_pos_rmse,
                        "sim3_extrinsic_candidate_orientation_rmse_deg": ext_rot_rmse,
                        "sim3_extrinsic_rotation_matrix": np.asarray(r_body, dtype=float).tolist(),
                    }
                    if requested_mode != mode:
                        ext_info["requested_align_mode"] = requested_mode
                    if align_indices is not None:
                        ext_info["align_index_count"] = int(idx.size)
                        if idx.size:
                            ext_info["align_index_first"] = int(idx[0])
                            ext_info["align_index_last"] = int(idx[-1])
                    _finalize_sim3_alignment_info(
                        ext_info,
                        scale=float(s_ext),
                        r_fit=r_ext_fit,
                        t_fit=t_ext_fit,
                        sim3_info=ext_sim3_info,
                    )
                    return pos_ext, q_ext, ext_info
            except Exception as exc:
                info["sim3_extrinsic_rotation_correction_error"] = str(exc)
        return pos_new, q_new, info

    if mode == "epica_anchor_sim3":
        s_eval, R_eval, t_eval, sim3_info = solve_anchor_sim3(
            pos_ref=pref,
            quat_ref=np.asarray(quat_ref[idx], dtype=float),
            pos_est=pest,
            quat_est=np.asarray(quat_est[idx], dtype=float),
        )
        pos_new = s_eval * (R_eval @ np.asarray(pos_est, dtype=float).T).T + t_eval
        q_new = normalize_quat_array((R.from_matrix(R_eval) * R.from_quat(q_est)).as_quat())
        info["align_scale"] = float(s_eval)
        info.update(sim3_info)
        info.update(sim3_scale_guard(float(s_eval)))
        return pos_new, q_new, info

    if mode == "scale":
        src_centered = pest - np.mean(pest, axis=0)
        dst_centered = pref - np.mean(pref, axis=0)
        denom = np.sum(src_centered**2)
        scale = 1.0 if denom < 1e-12 else float(np.sqrt(np.sum(dst_centered**2) / denom))
        pos_new = scale * np.asarray(pos_est, dtype=float)
        info["align_scale"] = float(scale)
        return pos_new, q_est, info

    if mode == "origin":
        R0 = R.from_quat(quat_ref[0]).as_matrix() @ R.from_quat(quat_est[0]).as_matrix().T
        t0 = np.asarray(pos_ref[0], dtype=float) - (R0 @ np.asarray(pos_est[0], dtype=float))
        pos_new = (R0 @ np.asarray(pos_est, dtype=float).T).T + t0
        q_new = normalize_quat_array((R.from_matrix(R0) * R.from_quat(q_est)).as_quat())
        info["align_scale"] = 1.0
        return pos_new, q_new, info

    raise ValueError(f"Unsupported eval alignment mode: {mode}")


def align_for_eval(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    mode: str,
    n_to_align: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    pos_new, quat_new, _ = align_for_eval_with_info(
        pos_ref=pos_ref,
        quat_ref=quat_ref,
        pos_est=pos_est,
        quat_est=quat_est,
        mode=mode,
        n_to_align=n_to_align,
    )
    return pos_new, quat_new


def project_to_plane(
    pos_xyz: np.ndarray,
    quat_xyzw: np.ndarray,
    plane: str,
) -> tuple[np.ndarray, np.ndarray]:
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
    projected = mats @ e1
    projected -= np.outer(projected @ normal, normal)
    norms = np.linalg.norm(projected, axis=1)
    degenerate = norms < 1e-12
    if np.any(degenerate):
        projected[degenerate] = e1
        norms[degenerate] = 1.0
    projected /= norms[:, None]
    angles = np.arctan2(projected @ e2, projected @ e1)
    out_quat = R.from_rotvec(angles[:, None] * normal).as_quat()

    return pos, normalize_quat_array(out_quat)


def _parse_scalar(text: str):
    stripped = str(text).strip()
    if stripped.startswith(("'", '"')) and stripped.endswith(("'", '"')):
        return stripped[1:-1]
    try:
        return int(stripped)
    except Exception:
        pass
    try:
        return float(stripped)
    except Exception:
        return stripped


def _load_ros_map_yaml_fallback(path: Path) -> dict:
    out: dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "origin":
            out[key] = ast.literal_eval(value)
        else:
            out[key] = _parse_scalar(value)
    return out


def load_ros_map_spec(path: str | Path) -> RosMapSpec:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"ROS map yaml not found: {yaml_path}")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception:
        data = _load_ros_map_yaml_fallback(yaml_path)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid ROS map yaml: {yaml_path}")
    image = data.get("image")
    resolution = data.get("resolution")
    origin = data.get("origin")
    if image is None or resolution is None or origin is None:
        raise ValueError(
            f"ROS map yaml must contain image/resolution/origin: {yaml_path}"
        )
    image_path = Path(str(image))
    if not image_path.is_absolute():
        image_path = (yaml_path.parent / image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"ROS map image not found: {image_path}")
    origin_vals = list(origin)
    if len(origin_vals) < 3:
        raise ValueError(f"ROS map origin must have 3 values: {yaml_path}")
    return RosMapSpec(
        image_path=image_path,
        resolution=float(resolution),
        origin_x=float(origin_vals[0]),
        origin_y=float(origin_vals[1]),
        origin_yaw=float(origin_vals[2]),
    )


def resolve_map_tile_contextily(cx, map_tile: str) -> dict[str, object]:
    tile = str(map_tile).strip()
    if tile == "":
        return {}
    if tile.lower().startswith("epsg:"):
        return {"crs": tile}

    source_obj: object = tile
    current: object = getattr(cx, "providers", {})
    for token in tile.split("."):
        next_obj = None
        if isinstance(current, dict):
            next_obj = current.get(token)
        else:
            next_obj = getattr(current, token, None)
        if next_obj is None:
            source_obj = tile
            break
        source_obj = next_obj
        current = next_obj
    return {"source": source_obj}


def load_inputs_from_args(args) -> MetricInputs:
    subcommand = str(args.subcommand)

    if subcommand == "tum":
        ref = Path(args.ref_file).expanduser().resolve()
        est = Path(args.est_file).expanduser().resolve()
        t_ref, pos_ref, quat_ref = load_reference_trajectory(ref, gt_format="tum")
        t_est, pos_est, quat_est = load_estimation_trajectory(est, est_format="tum")
        return MetricInputs(
            subcommand=subcommand,
            ref_name=str(ref),
            est_name=str(est),
            t_ref=t_ref,
            pos_ref=pos_ref,
            quat_ref=quat_ref,
            t_est=t_est,
            pos_est=pos_est,
            quat_est=quat_est,
        )

    if subcommand == "kitti":
        ref = Path(args.ref_file).expanduser().resolve()
        est = Path(args.est_file).expanduser().resolve()
        t_ref, pos_ref, quat_ref = load_reference_trajectory(ref, gt_format="kitti")
        t_est, pos_est, quat_est = load_estimation_trajectory(est, est_format="kitti")
        return MetricInputs(
            subcommand=subcommand,
            ref_name=str(ref),
            est_name=str(est),
            t_ref=t_ref,
            pos_ref=pos_ref,
            quat_ref=quat_ref,
            t_est=t_est,
            pos_est=pos_est,
            quat_est=quat_est,
        )

    if subcommand == "euroc":
        ref = Path(args.state_gt_csv).expanduser().resolve()
        est = Path(args.est_file).expanduser().resolve()
        t_ref, pos_ref, quat_ref = load_reference_trajectory(ref, gt_format="euroc")
        t_est, pos_est, quat_est = load_estimation_trajectory(est, est_format="tum")
        return MetricInputs(
            subcommand=subcommand,
            ref_name=str(ref),
            est_name=str(est),
            t_ref=t_ref,
            pos_ref=pos_ref,
            quat_ref=quat_ref,
            t_est=t_est,
            pos_est=pos_est,
            quat_est=quat_est,
        )

    if subcommand in {"bag", "bag2", "mcap"}:
        bag = Path(args.bag).expanduser().resolve()
        bag_fmt = "bag2" if subcommand == "mcap" else subcommand
        t_ref, pos_ref, quat_ref = load_reference_trajectory(
            bag, gt_format=bag_fmt, gt_topic=str(args.ref_topic)
        )
        t_est, pos_est, quat_est = load_estimation_trajectory(
            bag, est_format=bag_fmt, est_topic=str(args.est_topic)
        )
        return MetricInputs(
            subcommand=subcommand,
            ref_name=str(args.ref_topic),
            est_name=str(args.est_topic),
            t_ref=t_ref,
            pos_ref=pos_ref,
            quat_ref=quat_ref,
            t_est=t_est,
            pos_est=pos_est,
            quat_est=quat_est,
        )

    raise ValueError(f"Unknown subcommand: {subcommand}")
