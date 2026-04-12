from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
    print_metric_block,
    summarize_abs_errors,
)
from .io_utils import (
    load_estimation_trajectory,
    load_reference_trajectory,
    make_output_dir,
    save_metrics,
    write_metrics_zh_report,
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


def run_pipeline_modular(args, script_dir: Path):
    run_dir = make_output_dir(script_dir)
    print(f"Saving outputs to: {run_dir}")

    gt_path = Path(args.gt_csv)
    if not gt_path.is_absolute():
        gt_path = script_dir / gt_path
    if not gt_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_path}")

    print("Loading GT trajectory...")
    gt_format = getattr(args, "gt_format", "csv")
    gt_topic = getattr(args, "gt_topic", "")
    t_vicon, pos_vicon, quat_vicon = load_reference_trajectory(
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
        t_robot, pos_robot, quat_robot = load_estimation_trajectory(
            est_path,
            est_format,
            est_topic=est_topic,
        )

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
    # Evo-compatible interpretation:
    # - evo applies t_offset to est timestamps before matching.
    # - our internal shift is t_robot_sync = t_robot - offset_est_s.
    #   Therefore, evo-equivalent t_offset is -offset_est_s.
    evo_t_offset_used_s = -calculated_offset
    evo_match_max_diff_s = 0.02
    match_ids_ref, match_ids_est = matching_time_indices(
        t_vicon, t_robot, max_diff=evo_match_max_diff_s, offset_2=evo_t_offset_used_s
    )
    time_metrics["evo_t_offset_used_s"] = evo_t_offset_used_s
    time_metrics["evo_match_max_diff_s"] = evo_match_max_diff_s
    time_metrics["evo_matches_equivalent"] = float(len(match_ids_ref))
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
        rot_ext_err_deg = float(np.degrees(R.from_matrix(R_calc.T @ R_ext_true).magnitude()))
        t_ext_err = float(np.linalg.norm(t_calc - t_ext_true))
        rot_world_err_deg = float(np.degrees(R.from_matrix(Rw_calc.T @ Rw_true).magnitude()))
        t_world_err = float(np.linalg.norm(tw_calc - tw_true))
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

    rerun_info = {
        "enabled": "false",
        "status": "disabled",
        "message": "Use --rerun to enable.",
    }
    if bool(getattr(args, "rerun", False)):
        rerun_info = log_alignment_to_rerun(
            run_dir=run_dir,
            gt_xyz=pos_vicon,
            raw_xyz=pr_sync,
            step2_xyz=pr_corrected,
            step3_xyz=pr_final,
            timestamps_s=t_vicon,
            gt_quat=quat_vicon,
            raw_quat=qr_sync,
            step2_quat=q_step2,
            step3_quat=q_step3,
            raw_error_m=raw_err,
            step2_error_m=step2_err,
            step3_error_m=step3_err,
            app_id="vicon_ws_alignment",
            spawn=not bool(getattr(args, "rerun_no_spawn", False)),
            stride=int(getattr(args, "rerun_stride", 20)),
            motion_stride=int(getattr(args, "rerun_motion_stride", 5)),
        )
        print(f"Rerun status: {rerun_info['status']} - {rerun_info['message']}")

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
            "gt_format": str(gt_format),
            "est_format": str(getattr(args, "est_format", "auto")),
            "gt_topic": str(gt_topic),
            "est_topic": str(getattr(args, "est_topic", "")),
            "quat_interp": args.quat_interp,
            "rerun": rerun_info,
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
