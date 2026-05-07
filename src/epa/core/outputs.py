from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .evaluation import normalize_pose_relation
from .io_utils import (
    compact_metrics_payload,
    save_metrics,
    write_result_bundle,
    write_run_reports,
)
from ..viz.metric_plots import generate_ape_stage_raw_plot, generate_metric_plots, generate_time_rpe_metric_plots


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


def _plot_step1_outputs(
    *,
    plots_dir: Path,
    t_uniform,
    sig_gt,
    sig_est,
    corr,
    lags,
    dt_resample: float,
    calculated_offset: float,
    title_offset: str,
):
    fig_corr, ax_corr = plt.subplots(figsize=(12, 4))
    lag_times = np.asarray(lags, dtype=float) * float(dt_resample)
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
    plt.close(fig_corr)

    fig1, (ax_b, ax_a) = plt.subplots(2, 1, figsize=(12, 8))
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
    plt.close(fig1)
    return fig_corr_path, fig1_path


def _plot_piecewise_diagnostics(*, plots_dir: Path, piecewise_detail):
    seg_t = np.asarray(piecewise_detail["segment_mid_s"], dtype=float)
    seg_g = np.asarray(piecewise_detail["segment_global_rmse_m"], dtype=float)
    seg_l = np.asarray(piecewise_detail["segment_local_rmse_m"], dtype=float)
    valid_pw = np.isfinite(seg_t) & np.isfinite(seg_g) & np.isfinite(seg_l)
    if not np.any(valid_pw):
        return None

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
    plt.close(fig_pw)
    return fig_pw_path


def _plot_stage_alignment_maps(
    *,
    plots_dir: Path,
    pos_gt,
    pr_sync,
    pr_corrected,
    pr_final,
):
    stage_order = ["raw", "step2", "step3"]
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

    fig2 = plt.figure(figsize=(18, 5.8))
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
                f"ATE translation RMSE={float(np.sqrt(np.mean(stage_err**2))):.6f} m"
            ),
        )
    fig2.tight_layout()
    fig2_path = plots_dir / "step23_trajectory_alignment_3d.png"
    fig2.savefig(fig2_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    step3_err_subset = np.linalg.norm(np.asarray(pr_final, dtype=float) - np.asarray(pos_gt, dtype=float), axis=1)
    fig_step3_map = plt.figure(figsize=(8.4, 6.8))
    ax_step3_map = fig_step3_map.add_subplot(111, projection="3d")
    _plot_alignment_map(
        fig_step3_map,
        ax_step3_map,
        pos_ref=np.asarray(pos_gt, dtype=float),
        pos_est=np.asarray(pr_final, dtype=float),
        errors_m=step3_err_subset,
        title=(
            "Step3 Alignment Map\n"
            f"ATE translation RMSE={float(np.sqrt(np.mean(step3_err_subset**2))):.6f} m"
        ),
    )
    fig_step3_map.tight_layout()
    fig_step3_map_path = plots_dir / "step3_alignment_map.png"
    fig_step3_map.savefig(fig_step3_map_path, dpi=220, bbox_inches="tight")
    plt.close(fig_step3_map)

    return fig2_path, fig_step3_map_path


def _resolve_result_bundle_path(save_results_arg: str):
    save_results_arg = str(save_results_arg or "").strip()
    if not save_results_arg:
        return None
    bundle = Path(save_results_arg).expanduser()
    if not bundle.is_absolute():
        return (Path.cwd() / bundle).resolve()
    return bundle.resolve()


def _generate_and_cleanup_metric_plots(
    *,
    metrics_payload,
    plots_dir: Path,
    args,
):
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
        plot_files.extend(
            generate_time_rpe_metric_plots(
                metrics_payload=metrics_payload,
                out_dir=plots_dir,
                relation="translation_part",
                x_dimension=str(getattr(args, "plot_x_dimension", "seconds")),
            )
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
    else:
        plot_meta = {"enabled": False, "files": []}

    metrics_payload["metadata"]["plot"] = plot_meta
    core_plot_names = {
        "piecewise_segment_rmse.png",
        "step1_cross_correlation.png",
        "step1_time_alignment.png",
        "step23_trajectory_alignment_3d.png",
        "step3_alignment_map.png",
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
    return plot_files, retained_plot_paths


def _write_outputs(
    *,
    run_dir: Path,
    plots_dir: Path,
    metrics_payload,
    args,
):
    bundle_path = _resolve_result_bundle_path(getattr(args, "save_results", ""))
    if bundle_path is not None:
        metrics_payload["metadata"]["result_bundle"] = str(bundle_path)

    _generate_and_cleanup_metric_plots(
        metrics_payload=metrics_payload,
        plots_dir=plots_dir,
        args=args,
    )

    output_metrics_payload = (
        metrics_payload
        if bool(getattr(args, "save_full_metrics", False))
        else compact_metrics_payload(metrics_payload)
    )
    output_metrics_payload["metadata"]["metrics_detail"] = (
        "full" if bool(getattr(args, "save_full_metrics", False)) else "compact"
    )

    save_metrics(run_dir, output_metrics_payload)
    report_zh_path, report_en_path = write_run_reports(run_dir, metrics_payload)
    if bundle_path is not None:
        write_result_bundle(run_dir, output_metrics_payload, bundle_path)

    return {
        "bundle_path": bundle_path,
        "report_zh_path": report_zh_path,
        "report_en_path": report_en_path,
    }
