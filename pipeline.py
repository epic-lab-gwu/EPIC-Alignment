import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R, Slerp


def get_angular_velocity_norm(t, quats):
    rot = R.from_quat(quats)
    rel_rot = rot[:-1].inv() * rot[1:]
    angles = rel_rot.magnitude()
    dt = np.diff(t)
    omega = angles / dt
    t_mid = t[:-1] + dt / 2.0
    return t_mid, omega


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


def build_translation_system(pv, qv, pr, qr, Rext):
    Rv = R.from_quat(qv).as_matrix()
    Rr = R.from_quat(qr).as_matrix()
    C, d = [], []
    for i in range(len(pv) - 1):
        ta = Rv[i].T @ (pv[i + 1] - pv[i])
        tb = Rr[i].T @ (pr[i + 1] - pr[i])
        if np.linalg.norm(ta) < 1e-3:
            continue
        C.append(Rv[i].T @ Rv[i + 1] - np.eye(3))
        d.append(Rext @ tb - ta)
    if not C:
        raise ValueError("No valid translation constraints for extrinsic translation solve.")
    return np.vstack(C), np.concatenate(d), len(C)


def summarize_abs_errors(errors):
    stats = compute_error_statistics(errors)
    values = np.asarray(errors, dtype=float).reshape(-1)
    stats["p95"] = np.percentile(values, 95) if values.size > 0 else np.nan
    return stats


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


def rpe_pairs_by_index(poses, delta, all_pairs=False):
    n = len(poses)
    if delta < 1:
        raise ValueError("RPE delta must be >= 1 for frame unit.")
    if all_pairs:
        ids = np.arange(n, dtype=int)
        return [(int(i), int(i + delta)) for i in ids if i + delta < n]
    ids = np.arange(0, n, delta, dtype=int)
    return [(int(i), int(j)) for i, j in zip(ids, ids[1:])]


def rpe_pairs_by_path(poses, delta, tol=0.0, all_pairs=False):
    id_pairs = []
    if all_pairs:
        positions = np.array([pose[:3, 3] for pose in poses])
        distances = np.zeros(positions.shape[0], dtype=float)
        if positions.shape[0] > 1:
            distances[1:] = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        for i in range(distances.size - 1):
            offset = i + 1
            distances_from_here = distances[offset:] - distances[i]
            candidate_index = int(np.argmin(np.abs(distances_from_here - delta)))
            if np.abs(distances_from_here[candidate_index] - delta) > tol:
                continue
            id_pairs.append((i, candidate_index + offset))
    else:
        ids = []
        previous_pose = poses[0]
        current_path = 0.0
        for i, current_pose in enumerate(poses):
            current_path += float(np.linalg.norm(current_pose[:3, 3] - previous_pose[:3, 3]))
            previous_pose = current_pose
            if current_path >= delta:
                ids.append(i)
                current_path = 0.0
        id_pairs = [(i, j) for i, j in zip(ids, ids[1:])]
    return id_pairs


def rpe_pairs_by_angle(poses, delta, tol=0.0, degrees=False, all_pairs=False):
    bounds = [0.0, 180.0] if degrees else [0.0, np.pi]
    if delta < bounds[0] or delta > bounds[1]:
        raise ValueError(f"RPE delta angle must be within {bounds}.")

    delta_rad = np.deg2rad(delta) if degrees else delta
    tol_rad = np.deg2rad(tol) if degrees else tol
    rot_mats = np.array([pose[:3, :3] for pose in poses])
    if all_pairs:
        upper_bound = delta_rad + tol_rad
        lower_bound = delta_rad - tol_rad
        id_pairs = []
        for i in range(len(poses) - 1):
            offset = i + 1
            end_rots = R.from_matrix(rot_mats[offset:])
            start_rots = R.from_matrix(np.repeat(rot_mats[i][None, :, :], len(end_rots), axis=0))
            delta_angles = np.linalg.norm((start_rots.inv() * end_rots).as_rotvec(), axis=1)
            matches = np.argwhere((lower_bound <= delta_angles) & (delta_angles <= upper_bound)) + offset
            id_pairs.extend([(i, int(j)) for j in matches.flatten().tolist()])
        return id_pairs

    delta_angles = np.linalg.norm((R.from_matrix(rot_mats[:-1]).inv() * R.from_matrix(rot_mats[1:])).as_rotvec(), axis=1)
    accumulated_delta = 0.0
    current_start_index = 0
    id_pairs = []
    for i, current_delta in enumerate(delta_angles):
        end_index = i + 1
        accumulated_delta += current_delta
        if accumulated_delta >= delta_rad:
            id_pairs.append((current_start_index, end_index))
            accumulated_delta = 0.0
            current_start_index = end_index
    return id_pairs


def build_rpe_pairs(poses, delta, delta_unit="f", rel_delta_tol=0.1, all_pairs=False):
    if len(poses) < 2:
        return []

    if delta_unit == "f":
        if float(delta).is_integer():
            delta_int = int(delta)
        else:
            raise ValueError("RPE delta must be integer when delta_unit is frames ('f').")
        return rpe_pairs_by_index(poses, delta_int, all_pairs=all_pairs)
    if delta_unit == "m":
        tol = float(delta) * rel_delta_tol if all_pairs else 0.0
        return rpe_pairs_by_path(poses, float(delta), tol=tol, all_pairs=all_pairs)
    if delta_unit == "d":
        tol = float(delta) * rel_delta_tol if all_pairs else 0.0
        return rpe_pairs_by_angle(poses, float(delta), tol=tol, degrees=True, all_pairs=all_pairs)
    if delta_unit == "r":
        tol = float(delta) * rel_delta_tol if all_pairs else 0.0
        return rpe_pairs_by_angle(poses, float(delta), tol=tol, degrees=False, all_pairs=all_pairs)
    raise ValueError(f"Unsupported RPE delta unit: {delta_unit}")


def compute_ape_evo_style(pos_ref, quat_ref, pos_est, quat_est):
    if len(pos_ref) != len(pos_est):
        raise ValueError("APE requires trajectories with the same number of poses.")

    pos_ref = np.asarray(pos_ref, dtype=float)
    pos_est = np.asarray(pos_est, dtype=float)
    T_ref = poses_se3_from_traj(pos_ref, quat_ref)
    T_est = poses_se3_from_traj(pos_est, quat_est)
    E = np.array([relative_se3(T_est[i], T_ref[i]) for i in range(len(T_ref))])

    I3 = np.eye(3)
    I4 = np.eye(4)
    trans_err = np.linalg.norm(pos_est - pos_ref, axis=1)
    rot_part_err = np.linalg.norm(E[:, :3, :3] - I3, axis=(1, 2))
    full_err = np.linalg.norm(E - I4, axis=(1, 2))
    rot_angle_rad = np.abs(R.from_matrix(E[:, :3, :3]).magnitude())
    rot_angle_deg = np.degrees(rot_angle_rad)

    return {
        "translation_part": compute_error_statistics(trans_err),
        "point_distance": compute_error_statistics(trans_err),
        "rotation_part": compute_error_statistics(rot_part_err),
        "full_transformation": compute_error_statistics(full_err),
        "rotation_angle_rad": compute_error_statistics(rot_angle_rad),
        "rotation_angle_deg": compute_error_statistics(rot_angle_deg),
    }


def compute_rpe_evo_style(
    pos_ref,
    quat_ref,
    pos_est,
    quat_est,
    delta=1.0,
    delta_unit="f",
    rel_delta_tol=0.1,
    all_pairs=False,
    pairs_from_reference=False,
):
    if len(pos_ref) != len(pos_est):
        raise ValueError("RPE requires trajectories with the same number of poses.")

    pos_ref = np.asarray(pos_ref, dtype=float)
    pos_est = np.asarray(pos_est, dtype=float)
    T_ref = poses_se3_from_traj(pos_ref, quat_ref)
    T_est = poses_se3_from_traj(pos_est, quat_est)
    pair_source = T_ref if pairs_from_reference else T_est
    id_pairs = build_rpe_pairs(
        pair_source, delta=delta, delta_unit=delta_unit, rel_delta_tol=rel_delta_tol, all_pairs=all_pairs
    )

    if len(id_pairs) == 0:
        empty = compute_error_statistics(np.array([]))
        return {
            "pair_count": 0,
            "translation_part": empty,
            "point_distance": empty,
            "point_distance_error_ratio": empty,
            "rotation_part": empty,
            "full_transformation": empty,
            "rotation_angle_rad": empty,
            "rotation_angle_deg": empty,
        }

    ref_distances = np.array([np.linalg.norm(pos_ref[i] - pos_ref[j]) for i, j in id_pairs])
    est_distances = np.array([np.linalg.norm(pos_est[i] - pos_est[j]) for i, j in id_pairs])
    point_distance_err = np.abs(ref_distances - est_distances)
    ratio_mask = ref_distances != 0.0
    point_ratio = np.divide(point_distance_err[ratio_mask], ref_distances[ratio_mask]) * 100.0

    E = []
    for i, j in id_pairs:
        Q_rel = relative_se3(T_ref[i], T_ref[j])
        P_rel = relative_se3(T_est[i], T_est[j])
        E.append(relative_se3(Q_rel, P_rel))
    E = np.array(E)

    I3 = np.eye(3)
    I4 = np.eye(4)
    translation_part_err = np.linalg.norm(E[:, :3, 3], axis=1)
    rotation_part_err = np.linalg.norm(E[:, :3, :3] - I3, axis=(1, 2))
    full_err = np.linalg.norm(E - I4, axis=(1, 2))
    rot_angle_rad = np.abs(R.from_matrix(E[:, :3, :3]).magnitude())
    rot_angle_deg = np.degrees(rot_angle_rad)

    return {
        "pair_count": int(len(id_pairs)),
        "translation_part": compute_error_statistics(translation_part_err),
        "point_distance": compute_error_statistics(point_distance_err),
        "point_distance_error_ratio": compute_error_statistics(point_ratio),
        "rotation_part": compute_error_statistics(rotation_part_err),
        "full_transformation": compute_error_statistics(full_err),
        "rotation_angle_rad": compute_error_statistics(rot_angle_rad),
        "rotation_angle_deg": compute_error_statistics(rot_angle_deg),
    }


def print_metric_block(title, metrics, unit_map=None):
    unit_map = unit_map or {}
    print(f"\n--- {title} ---")
    for key, value in metrics.items():
        unit = unit_map.get(key, "")
        if np.isnan(value):
            print(f"{key}: nan {unit}".rstrip())
        else:
            print(f"{key}: {value:.6f} {unit}".rstrip())


def to_builtin(value):
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_metrics(output_dir, metrics_payload):
    output_dir = Path(output_dir)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics_summary.csv"

    json_path.write_text(
        json.dumps(to_builtin(metrics_payload), indent=2),
        encoding="utf-8",
    )

    rows = []
    for section, block in metrics_payload.items():
        if not isinstance(block, dict):
            continue
        for metric, value in block.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                rows.append(
                    {
                        "section": section,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    if rows:
        if pd is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        else:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
                writer.writeheader()
                writer.writerows(rows)


METRIC_ZH_EXPLAIN = {
    "offset_est_s": "估计得到的时间偏移（秒），表示 estimation 相对 GT 的时间差。",
    "offset_s": "最终采用的时间偏移参数（秒），用于把 estimation 时间轴对齐到 GT 时间轴。",
    "offset_err_ms": "时间偏移误差（毫秒），仅在 synthetic 有真值时有效。",
    "xcorr_peak_normalized": "互相关峰值归一化强度，越接近 1 表示时间匹配越强。",
    "xcorr_psr": "互相关峰值与旁瓣比（PSR），越大表示峰值越清晰唯一。",
    "omega_rmse_before": "时间对齐前，两路角速度模长差异的 RMSE（rad/s）。",
    "omega_rmse_after": "时间对齐后，两路角速度模长差异的 RMSE（rad/s）。",
    "omega_rmse_improve_pct": "时间对齐后 RMSE 改善百分比，越大越好。",
    "rot_res_mean_deg": "Step2 相对旋转约束残差均值（度），越小越好。",
    "rot_res_median_deg": "Step2 相对旋转约束残差中位数（度），越小越好。",
    "rot_res_p95_deg": "Step2 相对旋转约束残差 95 分位（度），越小越好。",
    "trans_eq_rmse_m": "Step2 平移方程残差 RMSE（米），越小越好。",
    "trans_eq_p95_m": "Step2 平移方程残差 95 分位（米），越小越好。",
    "translation_system_cond": "Step2 平移线性系统条件数，越接近 1 越稳定。",
    "translation_constraints": "Step2 有效平移约束数量，越多通常越稳。",
    "ate_rmse_raw_m": "未完成空间对齐前（Step1 后）的绝对轨迹误差 RMSE（米）。",
    "ate_rmse_step2_m": "完成 Step2 外参修正后的绝对轨迹误差 RMSE（米）。",
    "ate_rmse_step3_m": "完成 Step3 世界系对齐后的绝对轨迹误差 RMSE（米）。",
    "ate_p95_raw_m": "raw 绝对轨迹误差 95 分位（米）。",
    "ate_p95_step2_m": "step2 绝对轨迹误差 95 分位（米）。",
    "ate_p95_step3_m": "step3 绝对轨迹误差 95 分位（米）。",
    "ate_rmse_improve_raw_to_step3_pct": "从 raw 到 step3 的 RMSE 改善百分比。",
    "ate_rmse_improve_step2_to_step3_pct": "从 step2 到 step3 的 RMSE 改善百分比。",
    "extrinsic_rotation_error_deg": "外参旋转误差（度），仅 synthetic 有真值时有效。",
    "extrinsic_translation_error_m": "外参平移误差（米），仅 synthetic 有真值时有效。",
    "world_rotation_error_deg": "世界对齐旋转误差（度），仅 synthetic 有真值时有效。",
    "world_translation_error_m": "世界对齐平移误差（米），仅 synthetic 有真值时有效。",
}


def _as_float(block, key):
    if key not in block:
        return np.nan
    val = block[key]
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    return np.nan


def _analyze_metric_value(section, metric, value, block):
    if np.isnan(value):
        return "当前无真值或该指标不适用（nan）。"

    if metric == "offset_est_s":
        peak = _as_float(block, "xcorr_peak_normalized")
        psr = _as_float(block, "xcorr_psr")
        if np.isnan(peak) or np.isnan(psr):
            return f"估计时间偏移为 {value:.4f}s。"
        confidence = "高" if (peak >= 0.9 and psr >= 10) else ("中" if (peak >= 0.75 and psr >= 6) else "低")
        return (
            f"估计时间偏移为 {value:.4f}s；互相关峰值 {peak:.4f}、PSR {psr:.2f}，"
            f"峰值可信度为{confidence}。"
        )

    if metric == "xcorr_peak_normalized":
        if value >= 0.9:
            return f"归一化峰值为 {value:.4f}（>0.9），时间相关性很强。"
        if value >= 0.75:
            return f"归一化峰值为 {value:.4f}（0.75~0.9），时间相关性中等。"
        return f"归一化峰值为 {value:.4f}（<0.75），时间相关性偏弱。"

    if metric == "xcorr_psr":
        if value >= 10:
            return f"PSR 为 {value:.2f}（>=10），峰值清晰且唯一性较好。"
        if value >= 6:
            return f"PSR 为 {value:.2f}（6~10），峰值可用但需要留意次峰。"
        return f"PSR 为 {value:.2f}（<6），峰值分辨度偏低。"

    if metric in {"omega_rmse_before", "omega_rmse_after"}:
        before = _as_float(block, "omega_rmse_before")
        after = _as_float(block, "omega_rmse_after")
        if not np.isnan(before) and not np.isnan(after):
            delta = before - after
            ratio = delta / (before + 1e-12) * 100.0
            return (
                f"角速度 RMSE 从 {before:.4f} 降到 {after:.4f} rad/s，"
                f"绝对下降 {delta:.4f} rad/s（{ratio:.2f}%）。"
            )
        return f"该值为 {value:.6f}，越小越好。"

    if metric == "omega_rmse_improve_pct":
        if value >= 50:
            return f"时间对齐改善 {value:.2f}%（>50%），收益明显。"
        if value >= 20:
            return f"时间对齐改善 {value:.2f}%（20%~50%），有中等收益。"
        return f"时间对齐仅改善 {value:.2f}%（<20%），建议复核 offset。"

    if metric.startswith("rot_res_"):
        if value < 0.1:
            return f"旋转残差 {value:.4f} deg（<0.1），一致性很好。"
        if value < 1.0:
            return f"旋转残差 {value:.4f} deg（0.1~1），整体可接受。"
        return f"旋转残差 {value:.4f} deg（>=1），偏大。"

    if metric in {"trans_eq_rmse_m", "trans_eq_p95_m"}:
        cond = _as_float(block, "translation_system_cond")
        if np.isnan(cond):
            return f"平移残差为 {value:.6f} m，越小越好。"
        return (
            f"平移残差为 {value:.6f} m；对应系统条件数 {cond:.3f}，"
            f"{'解算稳定' if cond <= 5 else '解算稳定性一般'}。"
        )

    if metric == "translation_system_cond":
        if value <= 5:
            return f"条件数为 {value:.3f}（接近 1），平移解算稳定。"
        if value <= 20:
            return f"条件数为 {value:.3f}，可解但有一定病态风险。"
        return f"条件数为 {value:.3f}，病态风险较高。"

    if metric == "translation_constraints":
        if value >= 10000:
            return f"有效约束 {value:.0f} 条，约束非常充足。"
        if value >= 1000:
            return f"有效约束 {value:.0f} 条，数量充足。"
        return f"有效约束仅 {value:.0f} 条，可能受噪声影响。"

    if metric in {"ate_rmse_raw_m", "ate_rmse_step2_m", "ate_rmse_step3_m", "ate_p95_raw_m", "ate_p95_step2_m", "ate_p95_step3_m"}:
        raw = _as_float(block, "ate_rmse_raw_m")
        step2 = _as_float(block, "ate_rmse_step2_m")
        step3 = _as_float(block, "ate_rmse_step3_m")
        if metric == "ate_rmse_step3_m":
            if not np.isnan(raw):
                drop = raw - value
                ratio = drop / (raw + 1e-12) * 100.0
                quality = (
                    "达到 10cm 内" if value < 0.1 else
                    "达到分米级" if value < 0.3 else
                    "仍偏大"
                )
                return (
                    f"Step3 RMSE={value:.4f} m（{quality}）；"
                    f"相对 raw 降低 {drop:.4f} m（{ratio:.2f}%）。"
                )
        if metric == "ate_rmse_step2_m" and not np.isnan(step3):
            delta = value - step3
            return f"Step2 RMSE={value:.4f} m；比 Step3 高 {delta:.4f} m。"
        if metric == "ate_rmse_raw_m" and not np.isnan(step3):
            delta = value - step3
            return f"Raw RMSE={value:.4f} m；比 Step3 高 {delta:.4f} m。"
        if metric == "ate_p95_step3_m":
            return f"Step3 P95={value:.4f} m，表示 95% 时刻误差不超过该值。"
        if metric == "ate_p95_step2_m" and not np.isnan(step2):
            return f"Step2 P95={value:.4f} m。"
        if metric == "ate_p95_raw_m" and not np.isnan(raw):
            return f"Raw P95={value:.4f} m。"
        return f"该值为 {value:.6f}，越小越好。"

    if metric == "ate_rmse_improve_raw_to_step3_pct":
        raw = _as_float(block, "ate_rmse_raw_m")
        step3 = _as_float(block, "ate_rmse_step3_m")
        if not np.isnan(raw) and not np.isnan(step3):
            return (
                f"raw->step3 改善 {value:.2f}%（{raw:.4f} m -> {step3:.4f} m）。"
            )
        return f"改善比例为 {value:.2f}%。"

    if metric == "ate_rmse_improve_step2_to_step3_pct":
        step2 = _as_float(block, "ate_rmse_step2_m")
        step3 = _as_float(block, "ate_rmse_step3_m")
        if not np.isnan(step2) and not np.isnan(step3):
            return (
                f"step2->step3 改善 {value:.2f}%（{step2:.4f} m -> {step3:.4f} m）。"
            )
        return f"改善比例为 {value:.2f}%。"

    if metric in {
        "extrinsic_rotation_error_deg",
        "extrinsic_translation_error_m",
        "world_rotation_error_deg",
        "world_translation_error_m",
    }:
        return f"该值为 {value:.6f}，越接近 0 越好（仅 synthetic 有意义）。"

    if section == "estimated_params" and metric == "offset_s":
        return f"估计的全局时间偏移为 {value:.4f}s，用于将 estimation 时间轴对齐到 GT。"

    return f"当前值为 {value:.6f}，建议结合同组指标综合判断。"


def write_metrics_zh_report(output_dir, metrics_payload):
    output_dir = Path(output_dir)
    report_path = output_dir / "metrics_zh.md"

    lines = []
    lines.append("# 指标解读（中文）")
    lines.append("")
    lines.append("本文件自动生成：对每个英文指标给出中文释义，并结合当前数值做简要分析。")
    lines.append("")

    for section, block in metrics_payload.items():
        if not isinstance(block, dict):
            continue
        has_scalar = any(isinstance(v, (int, float, np.integer, np.floating)) for v in block.values())
        if not has_scalar:
            continue
        lines.append(f"## {section}")
        lines.append("")
        for metric, value in block.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            value_float = float(value)
            value_text = "nan" if np.isnan(value_float) else f"{value_float:.6f}"
            zh_explain = METRIC_ZH_EXPLAIN.get(metric, "该指标暂无预置中文解释。")
            analysis = _analyze_metric_value(section, metric, value_float, block)
            lines.append(f"- `{metric}`")
            lines.append(f"  - 中文解释：{zh_explain}")
            lines.append(f"  - 当前数值：`{value_text}`")
            lines.append(f"  - 数值分析：{analysis}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


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


def solve_extrinsic_translation(pv, qv, pr, qr, Rext):
    C, d, _ = build_translation_system(pv, qv, pr, qr, Rext)
    res, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
    return res


def solve_world_alignment(P, Q):
    cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    Rw = Vt.T @ U.T
    if np.linalg.det(Rw) < 0:
        Vt[2, :] *= -1
        Rw = Vt.T @ U.T
    return Rw, cQ - Rw @ cP


def normalize_quat_array(quat):
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / np.clip(norm, 1e-12, None)


def interpolate_quat_slerp(t_src, q_src, t_query):
    t_src = np.asarray(t_src, dtype=float)
    q_src = normalize_quat_array(np.asarray(q_src, dtype=float))
    t_query = np.asarray(t_query, dtype=float)

    # Slerp requires strictly increasing key times.
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


def find_col(columns, candidates):
    for cand in candidates:
        if cand in columns:
            return cand
    stripped = {c.strip(): c for c in columns}
    for cand in candidates:
        if cand in stripped:
            return stripped[cand]
    return None


def load_csv_numeric_columns(path):
    if pd is not None:
        df = pd.read_csv(path)
        columns = list(df.columns)
        data = {c: np.asarray(df[c].values, dtype=float) for c in columns}
        return columns, data

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns = list(reader.fieldnames)
        data = {c: [] for c in columns}
        for row_id, row in enumerate(reader, start=2):
            for c in columns:
                raw = row.get(c, "")
                if raw is None or str(raw).strip() == "":
                    raise ValueError(f"Missing numeric value at row {row_id}, column '{c}' in {path}")
                try:
                    data[c].append(float(raw))
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value at row {row_id}, column '{c}' in {path}: {raw}"
                    ) from exc
    data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    return columns, data


def load_vicon_csv(path):
    columns, data = load_csv_numeric_columns(path)
    t_col = find_col(columns, ["#timestamp", "timestamp"])
    px_col = find_col(columns, ["p_RS_R_x [m]", "p_x", "px", "tx"])
    py_col = find_col(columns, ["p_RS_R_y [m]", "p_y", "py", "ty"])
    pz_col = find_col(columns, ["p_RS_R_z [m]", "p_z", "pz", "tz"])
    qx_col = find_col(columns, ["q_RS_x []", "q_x", "qx"])
    qy_col = find_col(columns, ["q_RS_y []", "q_y", "qy"])
    qz_col = find_col(columns, ["q_RS_z []", "q_z", "qz"])
    qw_col = find_col(columns, ["q_RS_w []", "q_w", "qw"])

    needed = [t_col, px_col, py_col, pz_col, qx_col, qy_col, qz_col, qw_col]
    if any(c is None for c in needed):
        raise ValueError(f"GT CSV missing required columns: {path}")

    t = normalize_time_to_seconds(data[t_col])
    pos = np.column_stack([data[px_col], data[py_col], data[pz_col]])
    quat = normalize_quat_array(np.column_stack([data[qx_col], data[qy_col], data[qz_col], data[qw_col]]))
    return t, pos, quat


def load_estimation_csv(path):
    columns, data = load_csv_numeric_columns(path)
    t_col = find_col(columns, ["#timestamp", "timestamp", "time", "t"])
    px_col = find_col(columns, ["p_RS_R_x [m]", "p_x", "px", "tx", "x"])
    py_col = find_col(columns, ["p_RS_R_y [m]", "p_y", "py", "ty", "y"])
    pz_col = find_col(columns, ["p_RS_R_z [m]", "p_z", "pz", "tz", "z"])
    qx_col = find_col(columns, ["q_RS_x []", "q_x", "qx"])
    qy_col = find_col(columns, ["q_RS_y []", "q_y", "qy"])
    qz_col = find_col(columns, ["q_RS_z []", "q_z", "qz"])
    qw_col = find_col(columns, ["q_RS_w []", "q_w", "qw"])

    needed = [t_col, px_col, py_col, pz_col, qx_col, qy_col, qz_col, qw_col]
    if any(c is None for c in needed):
        raise ValueError(f"Estimation CSV missing required columns: {path}")

    t = normalize_time_to_seconds(data[t_col])
    pos = np.column_stack([data[px_col], data[py_col], data[pz_col]])
    quat = normalize_quat_array(np.column_stack([data[qx_col], data[qy_col], data[qz_col], data[qw_col]]))
    return t, pos, quat


def load_estimation_tum(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 8:
        raise ValueError("Trajectory file must have at least 8 columns: t tx ty tz qx qy qz qw")

    t = normalize_time_to_seconds(arr[:, 0])
    pos = arr[:, 1:4]
    quat = normalize_quat_array(arr[:, 4:8])
    return t, pos, quat


def load_estimation_trajectory(path, est_format):
    if est_format == "csv":
        return load_estimation_csv(path)
    if est_format == "tum":
        return load_estimation_tum(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_estimation_csv(path)
    return load_estimation_tum(path)


def make_output_dir(script_dir):
    output_root = script_dir / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_stamp
    idx = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_stamp}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="3-step trajectory alignment and extrinsic calibration pipeline"
    )
    parser.add_argument("--gt-csv", default="gt.csv", help="Path to GT CSV")
    parser.add_argument("--est-path", default=None, help="Path to estimation trajectory")
    parser.add_argument(
        "--est-format",
        choices=["auto", "csv", "tum"],
        default="auto",
        help="Estimation trajectory format",
    )
    parser.add_argument(
        "--dt-resample",
        type=float,
        default=0.001,
        help="Resampling step for step-1 correlation (seconds)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic offset/injected transforms instead of real estimation",
    )
    parser.add_argument(
        "--quat-interp",
        choices=["linear", "slerp"],
        default="linear",
        help="Quaternion interpolation method after time alignment",
    )
    parser.add_argument(
        "--rpe-delta",
        type=float,
        default=1.0,
        help="RPE delta (same meaning as evo: frame/path/angle increment).",
    )
    parser.add_argument(
        "--rpe-delta-unit",
        choices=["f", "m", "d", "r"],
        default="f",
        help="RPE delta unit: f=frames, m=meters, d=degrees, r=radians.",
    )
    parser.add_argument(
        "--rpe-delta-tol",
        type=float,
        default=0.1,
        help="Relative RPE delta tolerance (used in all-pairs mode for m/d/r).",
    )
    parser.add_argument(
        "--rpe-all-pairs",
        action="store_true",
        help="Use all candidate pairs for RPE (evo-style).",
    )
    parser.add_argument(
        "--rpe-pairs-from-reference",
        action="store_true",
        help="Build RPE pairs from reference trajectory instead of estimate.",
    )
    return parser.parse_args()


def run_pipeline(args):
    script_dir = Path(__file__).resolve().parent
    run_dir = make_output_dir(script_dir)
    print(f"Saving outputs to: {run_dir}")

    gt_path = Path(args.gt_csv)
    if not gt_path.is_absolute():
        gt_path = script_dir / gt_path
    if not gt_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_path}")

    print("Loading GT trajectory...")
    t_vicon, pos_vicon, quat_vicon = load_vicon_csv(gt_path)

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

        t_robot = t_vicon + ARTIFICIAL_OFFSET
        pos_robot = np.zeros_like(pos_vicon)
        quat_robot = np.zeros_like(quat_vicon)
        Rv_mats = R.from_quat(quat_vicon).as_matrix()

        for i in range(len(t_vicon)):
            R_abs = Rv_mats[i] @ R_ext_true
            quat_robot[i] = R.from_matrix(Rw_true.T @ R_abs).as_quat()
            p_offset = pos_vicon[i] + Rv_mats[i] @ t_ext_true
            pos_robot[i] = Rw_true.T @ (p_offset - tw_true)
    else:
        if args.est_path is None:
            raise ValueError("Real mode requires --est-path")

        est_path = Path(args.est_path)
        if not est_path.is_absolute():
            est_path = script_dir / est_path
        if not est_path.exists():
            raise FileNotFoundError(f"Estimation trajectory file not found: {est_path}")

        print("Loading estimation trajectory...")
        est_format = args.est_format
        t_robot, pos_robot, quat_robot = load_estimation_trajectory(est_path, est_format)

    print("--- STEP 1: TIME ALIGNMENT ---")
    t_v_mid, om_v = get_angular_velocity_norm(t_vicon, quat_vicon)
    t_r_mid, om_r = get_angular_velocity_norm(t_robot, quat_robot)

    dt_resample = args.dt_resample
    t_min = max(t_v_mid[0], t_r_mid[0])
    t_max = min(t_v_mid[-1], t_r_mid[-1])
    if t_max - t_min < 100 * dt_resample:
        raise ValueError("Not enough overlap between GT and estimation for robust time alignment.")

    t_uniform = np.arange(t_min, t_max, dt_resample)

    sig_v = interp1d(t_v_mid, om_v, kind="linear")(t_uniform)
    sig_r = interp1d(t_r_mid, om_r, kind="linear")(t_uniform)
    sig_v -= np.mean(sig_v)
    sig_r -= np.mean(sig_r)

    corr = correlate(sig_r, sig_v, mode="full")
    lags = np.arange(-len(sig_v) + 1, len(sig_r))
    peak_idx = np.argmax(corr)
    calculated_offset = lags[peak_idx] * dt_resample
    print(f"Calculated Time Offset: {calculated_offset:.4f} s")

    sig_r_shifted = interp1d(
        t_uniform - calculated_offset,
        sig_r,
        kind="linear",
        fill_value="extrapolate",
    )(t_uniform)

    corr_norm_peak = np.max(corr) / (np.linalg.norm(sig_v) * np.linalg.norm(sig_r) + 1e-12)
    psr = compute_psr(corr, peak_idx, guard_bins=max(1, int(0.02 / dt_resample)))
    time_metrics = {
        "offset_est_s": calculated_offset,
        "offset_err_ms": np.nan,
        "xcorr_peak_normalized": corr_norm_peak,
        "xcorr_psr": psr,
        "omega_rmse_before": rmse(sig_r - sig_v),
        "omega_rmse_after": rmse(sig_r_shifted - sig_v),
    }
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
    fig_corr_path = run_dir / "step1_cross_correlation.png"
    fig_corr.savefig(fig_corr_path, dpi=200, bbox_inches="tight")

    fig1, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(12, 8))

    title_offset = f"Known offset: {ARTIFICIAL_OFFSET}s" if args.synthetic else "Unknown offset"
    ax_b.set_title(f"Step 1: Full Sequence BEFORE Alignment ({title_offset})")
    ax_b.plot(t_uniform, sig_v, label="Vicon Omega", color="green", alpha=0.6)
    ax_b.plot(t_uniform, sig_r, "r--", label="Estimation Omega", alpha=0.6)
    ax_b.set_ylabel("Omega Norm")
    ax_b.legend(loc="upper right")
    ax_b.grid(True, linestyle=":", alpha=0.5)

    ax_a.set_title(f"Step 1: Full Sequence AFTER Alignment (Calculated: {calculated_offset:.4f}s)")
    ax_a.plot(t_uniform, sig_v, label="Vicon Omega", color="green", alpha=0.6)
    ax_a.plot(t_uniform - calculated_offset, sig_r, "b--", label="Estimation Omega (Corrected)", alpha=0.8)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Omega Norm")
    ax_a.legend(loc="upper right")
    ax_a.grid(True, linestyle=":", alpha=0.5)

    # Show a shorter window to make alignment quality visually clear.
    view_window_s = 40.0
    x0 = t_uniform[0]
    x1 = min(t_uniform[-1], x0 + view_window_s)
    ax_b.set_xlim(x0, x1)
    ax_a.set_xlim(x0, x1)

    plt.tight_layout()
    fig1_path = run_dir / "step1_time_alignment.png"
    fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")

    t_robot_sync = t_robot - calculated_offset
    interp_p = interp1d(t_robot_sync, pos_robot, axis=0, fill_value="extrapolate")
    pr_sync = interp_p(t_vicon)
    if args.quat_interp == "slerp":
        qr_sync = interpolate_quat_slerp(t_robot_sync, quat_robot, t_vicon)
    else:
        qr_sync = interpolate_quat_linear(t_robot_sync, quat_robot, t_vicon)

    print("\n--- STEP 2: SOLVING EXTRINSICS ---")
    R_calc = solve_extrinsic_rotation(quat_vicon, qr_sync)
    t_calc = solve_extrinsic_translation(pos_vicon, quat_vicon, pr_sync, qr_sync, R_calc)
    print(f"Calculated Extrinsic Rotation Matrix:\n{np.round(R_calc, 4)}")
    print(f"Calculated Translation: {np.round(t_calc, 4)} m")

    pr_corrected = np.zeros_like(pr_sync)
    Rr_mats = R.from_quat(qr_sync).as_matrix()
    for i in range(len(pr_sync)):
        pr_corrected[i] = pr_sync[i] - (Rr_mats[i] @ R_calc.T) @ t_calc

    print("\n--- STEP 3: WORLD ALIGNMENT ---")
    Rw_calc, tw_calc = solve_world_alignment(pr_corrected, pos_vicon)
    pr_final = (Rw_calc @ pr_corrected.T).T + tw_calc
    print(f"Calculated World Rotation Matrix:\n{np.round(Rw_calc, 4)}")
    print(f"Calculated World Translation: {np.round(tw_calc, 4)} m")

    R_step2_mats = np.einsum("nij,jk->nik", Rr_mats, R_calc.T)
    R_step3_mats = np.einsum("ij,njk->nik", Rw_calc, R_step2_mats)
    q_step2 = normalize_quat_array(R.from_matrix(R_step2_mats).as_quat())
    q_step3 = normalize_quat_array(R.from_matrix(R_step3_mats).as_quat())

    if args.synthetic:
        rot_ext_err_deg = np.degrees(R.from_matrix(R_calc.T @ R_ext_true).magnitude())
        t_ext_err = np.linalg.norm(t_calc - t_ext_true)
        rot_world_err_deg = np.degrees(R.from_matrix(Rw_calc.T @ Rw_true).magnitude())
        t_world_err = np.linalg.norm(tw_calc - tw_true)
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

    Rv_rot = R.from_quat(quat_vicon)
    Rr_rot = R.from_quat(qr_sync)
    dA = Rv_rot[:-1].inv() * Rv_rot[1:]
    dB = Rr_rot[:-1].inv() * Rr_rot[1:]
    Rext_rot = R.from_matrix(R_calc)
    dA_pred = Rext_rot * dB * Rext_rot.inv()
    rot_res_deg = np.degrees((dA.inv() * dA_pred).magnitude())

    C_mat, d_vec, num_constraints = build_translation_system(
        pos_vicon, quat_vicon, pr_sync, qr_sync, R_calc
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

    raw_err = np.linalg.norm(pr_sync - pos_vicon, axis=1)
    step2_err = np.linalg.norm(pr_corrected - pos_vicon, axis=1)
    step3_err = np.linalg.norm(pr_final - pos_vicon, axis=1)
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

    stage_trajs = {
        "raw": {"pos": pr_sync, "quat": qr_sync},
        "step2": {"pos": pr_corrected, "quat": q_step2},
        "step3": {"pos": pr_final, "quat": q_step3},
    }
    evo_ape = {}
    evo_rpe = {}
    for stage_name, stage_data in stage_trajs.items():
        evo_ape[stage_name] = compute_ape_evo_style(
            pos_ref=pos_vicon,
            quat_ref=quat_vicon,
            pos_est=stage_data["pos"],
            quat_est=stage_data["quat"],
        )
        evo_rpe[stage_name] = compute_rpe_evo_style(
            pos_ref=pos_vicon,
            quat_ref=quat_vicon,
            pos_est=stage_data["pos"],
            quat_est=stage_data["quat"],
            delta=args.rpe_delta,
            delta_unit=args.rpe_delta_unit,
            rel_delta_tol=args.rpe_delta_tol,
            all_pairs=args.rpe_all_pairs,
            pairs_from_reference=args.rpe_pairs_from_reference,
        )

    evo_metrics = {
        "ape": evo_ape,
        "rpe": evo_rpe,
        "rpe_config": {
            "delta": args.rpe_delta,
            "delta_unit": args.rpe_delta_unit,
            "delta_tol": args.rpe_delta_tol,
            "all_pairs": bool(args.rpe_all_pairs),
            "pairs_from_reference": bool(args.rpe_pairs_from_reference),
        },
    }

    print("\n--- EVO-STYLE METRICS (APE/RPE) ---")
    for stage_name in ("raw", "step2", "step3"):
        ape_t = evo_metrics["ape"][stage_name]["translation_part"]["rmse"]
        ape_r = evo_metrics["ape"][stage_name]["rotation_angle_deg"]["rmse"]
        rpe_t = evo_metrics["rpe"][stage_name]["translation_part"]["rmse"]
        rpe_r = evo_metrics["rpe"][stage_name]["rotation_angle_deg"]["rmse"]
        pairs = evo_metrics["rpe"][stage_name]["pair_count"]
        print(
            f"{stage_name}: "
            f"APE_trans_rmse={ape_t:.6f} m, "
            f"APE_rot_rmse={ape_r:.6f} deg, "
            f"RPE_trans_rmse={rpe_t:.6f} m, "
            f"RPE_rot_rmse={rpe_r:.6f} deg, "
            f"pairs={pairs}"
        )

    fig2 = plt.figure(figsize=(15, 5))
    titles = ["Raw (After Step-1 Sync)", "Sensor Fixed (Step 2)", "Fully Aligned (Step 3)"]
    datas = [pr_sync[::50], pr_corrected[::50], pr_final[::50]]
    for i in range(3):
        ax = fig2.add_subplot(131 + i, projection="3d")
        ax.set_title(titles[i])
        ax.plot(pos_vicon[::50, 0], pos_vicon[::50, 1], pos_vicon[::50, 2], "g", alpha=0.3)
        ax.plot(datas[i][:, 0], datas[i][:, 1], datas[i][:, 2], "r--")
    fig2.tight_layout()
    fig2_path = run_dir / "step23_trajectory_alignment_3d.png"
    fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")

    metrics_payload = {
        "time_alignment": time_metrics,
        "step2_residuals": step2_metrics,
        "trajectory": traj_metrics,
        "evo_metrics": evo_metrics,
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
            "quat_interp": args.quat_interp,
            "output_dir": str(run_dir),
        },
    }
    save_metrics(run_dir, metrics_payload)
    write_metrics_zh_report(run_dir, metrics_payload)

    print("\n--- OUTPUT FILES ---")
    print(f"Saved figure: {fig_corr_path}")
    print(f"Saved figure: {fig1_path}")
    print(f"Saved figure: {fig2_path}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print(f"Saved metrics: {run_dir / 'metrics_summary.csv'}")
    print(f"Saved report: {run_dir / 'metrics_zh.md'}")

    plt.close(fig_corr)
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    run_pipeline(parse_args())
