import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from datetime import datetime
import json

# ==========================================
# 核心功能：计算角速度模长（带时间参数）
# ==========================================
def get_angular_velocity_norm(t, quats):
    """Compute |omega| from quaternion sequence using consecutive relative rotations."""
    rot = R.from_quat(quats)
    rel_rot = rot[:-1].inv() * rot[1:]
    angles = rel_rot.magnitude()
    dt = np.diff(t)
    omega = angles / dt
    t_mid = t[:-1] + dt / 2.0
    return t_mid, omega

def rmse(x):
    return np.sqrt(np.mean(np.square(x)))

def compute_psr(corr, peak_idx, guard_bins):
    """Peak-to-sidelobe ratio, larger is better."""
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
    """
    Build linear system C * t_ext = d from relative translations.
    Returns stacked C, d and number of valid constraints.
    """
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
    return {
        "rmse": rmse(errors),
        "mean": np.mean(errors),
        "median": np.median(errors),
        "p95": np.percentile(errors, 95),
        "max": np.max(errors),
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
    """Convert numpy/scipy values to JSON-serializable python types."""
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
    """Save metrics as JSON and a flat CSV summary."""
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
        pd.DataFrame(rows).to_csv(csv_path, index=False)

def solve_extrinsic_rotation(qv, qr):
    """
    Solve R_ext from relative rotations:
        vB_k = R_ext^T * vA_k  ->  vA_k = R_ext * vB_k
    where vA/vB are rotation vectors of consecutive relative motions.
    """
    Rv = R.from_quat(qv); Rr = R.from_quat(qr)
    vA = (Rv[:-1].inv() * Rv[1:]).as_rotvec()
    vB = (Rr[:-1].inv() * Rr[1:]).as_rotvec()
    # Remove near-zero motion samples to avoid numerical noise dominating SVD.
    mask = np.linalg.norm(vA, axis=1) > 1e-3
    if np.sum(mask) < 3:
        raise ValueError("Not enough rotational excitation to estimate extrinsic rotation.")
    H = vB[mask].T @ vA[mask]
    U, S, Vt = np.linalg.svd(H)
    X = Vt.T @ U.T
    if np.linalg.det(X) < 0: Vt[2,:] *= -1; X = Vt.T @ U.T
    return X

def solve_extrinsic_translation(pv, qv, pr, qr, Rext):
    """
    Solve t_ext via linear least squares from relative translation constraints:
        (R_A_k^T R_A_{k+1} - I) * t_ext = R_ext * tB_k - tA_k
    This form uses relative motion to reduce drift/world-frame bias.
    """
    C, d, _ = build_translation_system(pv, qv, pr, qr, Rext)
    res, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
    return res

def solve_world_alignment(P, Q):
    """Rigid alignment (no scale): find Rw, tw such that Q ~= Rw * P + tw."""
    cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    Rw = Vt.T @ U.T
    if np.linalg.det(Rw) < 0: Vt[2,:] *= -1; Rw = Vt.T @ U.T
    return Rw, cQ - Rw @ cP

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_stamp
    idx = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_stamp}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving outputs to: {run_dir}")

    print("Loading EuRoC data...")
    data_path = script_dir / "data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")
    df = pd.read_csv(data_path)
    t_vicon = df['#timestamp'].values / 1e9
    pos_vicon = df[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values
    quat_vicon = df[[' q_RS_x []', ' q_RS_y []', ' q_RS_z []', ' q_RS_w []']].values
    
    # --- 1. 注入误差 ---
    # 这里构造可控的“真值外参 + 世界偏移 + 时间偏移”，便于后续 sanity check。
    ARTIFICIAL_OFFSET = 0.123 
    R_ext_true = R.from_euler('zyx', [45, -20, 30], degrees=True).as_matrix()
    t_ext_true = np.array([0.5, -0.2, 0.1])
    Rw_true = R.from_euler('z', 25, degrees=True).as_matrix()
    tw_true = np.array([1.5, -0.5, 0.3])

    # 模拟 Robot 轨迹 (带延迟)
    t_robot_fake = t_vicon + ARTIFICIAL_OFFSET
    pos_robot_fake = np.zeros_like(pos_vicon)
    quat_robot_fake = np.zeros_like(quat_vicon)
    Rv_mats = R.from_quat(quat_vicon).as_matrix()

    for i in range(len(t_vicon)):
        R_abs = Rv_mats[i] @ R_ext_true
        quat_robot_fake[i] = R.from_matrix(Rw_true.T @ R_abs).as_quat()
        p_offset = pos_vicon[i] + Rv_mats[i] @ t_ext_true
        pos_robot_fake[i] = Rw_true.T @ (p_offset - tw_true)

    # --- 2. Step 1: Time Alignment (角速度模长 + 互相关) ---
    print("--- STEP 1: TIME ALIGNMENT ---")
    t_v_mid, om_v = get_angular_velocity_norm(t_vicon, quat_vicon)
    t_r_mid, om_r = get_angular_velocity_norm(t_robot_fake, quat_robot_fake)

    dt_resample = 0.001 
    t_min, t_max = max(t_v_mid[0], t_r_mid[0]), min(t_v_mid[-1], t_r_mid[-1])
    t_uniform = np.arange(t_min, t_max, dt_resample)

    sig_v = interp1d(t_v_mid, om_v, kind='linear')(t_uniform)
    sig_r = interp1d(t_r_mid, om_r, kind='linear')(t_uniform)
    sig_v -= np.mean(sig_v); sig_r -= np.mean(sig_r)

    corr = correlate(sig_r, sig_v, mode='full')
    lags = np.arange(-len(sig_v) + 1, len(sig_r))
    peak_idx = np.argmax(corr)
    calculated_offset = lags[peak_idx] * dt_resample
    print(f"Calculated Time Offset: {calculated_offset:.4f} s")

    # Time-alignment quality metrics.
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
        "offset_err_ms": abs(calculated_offset - ARTIFICIAL_OFFSET) * 1e3,
        "xcorr_peak_normalized": corr_norm_peak,
        "xcorr_psr": psr,
        "omega_rmse_before": rmse(sig_r - sig_v),
        "omega_rmse_after": rmse(sig_r_shifted - sig_v),
    }
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

    # Step-1 correlation figure for quick review of peak sharpness.
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

    # --- Step 1 可视化：全量波形对齐图 ---
    fig1, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 1. 对齐前：显示全量数据
    ax_b.set_title(f"Step 1: Full Sequence BEFORE Alignment (Offset: {ARTIFICIAL_OFFSET}s)")
    ax_b.plot(t_uniform, sig_v, label="Vicon Omega", color='green', alpha=0.6)
    ax_b.plot(t_uniform, sig_r, 'r--', label="Robot Omega (Delayed)", alpha=0.6)
    ax_b.set_ylabel("Omega Norm")
    ax_b.legend(loc='upper right')
    ax_b.grid(True, linestyle=':', alpha=0.5)

    # 2. 对齐后：显示全量数据，验证全局一致性
    ax_a.set_title(f"Step 1: Full Sequence AFTER Alignment (Calculated: {calculated_offset:.4f}s)")
    ax_a.plot(t_uniform, sig_v, label="Vicon Omega", color='green', alpha=0.6)
    # 这里的 sig_r 已经是插值对齐后的，直接画出观察重合度
    ax_a.plot(t_uniform - calculated_offset, sig_r, 'b--', label="Robot Omega (Corrected)", alpha=0.8)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Omega Norm")
    ax_a.legend(loc='upper right')
    ax_a.grid(True, linestyle=':', alpha=0.5)

    # 如果你想看某个剧烈运动的时间段，可以取消下面这一行的注释来手动缩放 X 轴
    # ax_a.set_xlim(t_min + 10, t_min + 20) 

    plt.tight_layout()
    fig1_path = run_dir / "step1_time_alignment.png"
    fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")

    # --- 3. Step 2 & 3: Spatial Alignment ---
    # 先把时间轴纠正到同一参考，再做空间标定。
    t_robot_sync = t_robot_fake - calculated_offset
    # 位置线性插值；姿态先分量插值后归一化（轻量实现，工程上可替换为 SLERP）。
    interp_p = interp1d(t_robot_sync, pos_robot_fake, axis=0, fill_value="extrapolate")
    interp_q = interp1d(t_robot_sync, quat_robot_fake, axis=0, fill_value="extrapolate")
    
    pr_sync = interp_p(t_vicon)
    qr_sync = interp_q(t_vicon)
    qr_sync = qr_sync / np.linalg.norm(qr_sync, axis=1, keepdims=True)

    print("\n--- STEP 2: SOLVING EXTRINSICS ---")
    R_calc = solve_extrinsic_rotation(quat_vicon, qr_sync)
    t_calc = solve_extrinsic_translation(pos_vicon, quat_vicon, pr_sync, qr_sync, R_calc)
    print(f"Calculated Extrinsic Rotation Matrix:\n{np.round(R_calc, 4)}")
    print(f"Calculated Translation: {np.round(t_calc, 4)} m")

    # 轨迹修正
    pr_corrected = np.zeros_like(pr_sync)
    Rr_mats = R.from_quat(qr_sync).as_matrix()
    for i in range(len(pr_sync)):
        pr_corrected[i] = pr_sync[i] - (Rr_mats[i] @ R_calc.T) @ t_calc
    
    print("\n--- STEP 3: WORLD ALIGNMENT ---")
    Rw_calc, tw_calc = solve_world_alignment(pr_corrected, pos_vicon)
    pr_final = (Rw_calc @ pr_corrected.T).T + tw_calc
    print(f"Calculated World Rotation Matrix:\n{np.round(Rw_calc, 4)}")
    print(f"Calculated World Translation: {np.round(tw_calc, 4)} m")

    # 由于这里使用了人工注入参数，可直接打印误差验证三步法是否收敛到正确解。
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

    # Step-2 residual metrics (work without ground-truth extrinsics).
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

    # Trajectory-level metrics before/after alignment.
    raw_err = np.linalg.norm(pos_robot_fake - pos_vicon, axis=1)
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

    # 3D 可视化
    fig2 = plt.figure(figsize=(15, 5))
    titles = ["Raw (Time+Space Offset)", "Sensor Fixed (Step 2)", "Fully Aligned (Step 3)"]
    datas = [pos_robot_fake[::50], pr_corrected[::50], pr_final[::50]]
    for i in range(3):
        ax = fig2.add_subplot(131+i, projection='3d')
        ax.set_title(titles[i])
        ax.plot(pos_vicon[::50,0], pos_vicon[::50,1], pos_vicon[::50,2], 'g', alpha=0.3)
        ax.plot(datas[i][:,0], datas[i][:,1], datas[i][:,2], 'r--')
    fig2.tight_layout()
    fig2_path = run_dir / "step23_trajectory_alignment_3d.png"
    fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")

    # Persist metrics and estimated parameters.
    metrics_payload = {
        "time_alignment": time_metrics,
        "step2_residuals": step2_metrics,
        "trajectory": traj_metrics,
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
            "data_path": str(data_path),
            "output_dir": str(run_dir),
        },
    }
    save_metrics(run_dir, metrics_payload)

    print("\n--- OUTPUT FILES ---")
    print(f"Saved figure: {fig_corr_path}")
    print(f"Saved figure: {fig1_path}")
    print(f"Saved figure: {fig2_path}")
    print(f"Saved metrics: {run_dir / 'metrics.json'}")
    print(f"Saved metrics: {run_dir / 'metrics_summary.csv'}")

    plt.close(fig_corr)
    plt.close(fig1)
    plt.close(fig2)
