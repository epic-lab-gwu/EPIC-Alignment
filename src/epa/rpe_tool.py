from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
import matplotlib

from epa.viz.backend_bootstrap import bootstrap_matplotlib_backend

bootstrap_matplotlib_backend(matplotlib)
try:
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401
except Exception:
    _Axes3D = None
import matplotlib.pyplot as plt
import numpy as np

from epa.config_cli import parse_args_with_config
from epa.core.evaluation import RELATION_UNITS, compute_rpe_evo_style, normalize_pose_relation
from epa.core.io_utils import save_metrics, to_builtin, write_result_bundle
from epa.metric_cli_common import (
    MetricInputs,
    align_for_eval,
    apply_time_window,
    cum_distance,
    default_out_dir,
    downsample_traj,
    load_inputs_from_args,
    load_ros_map_spec,
    motion_filter_traj,
    project_to_plane,
    resolve_eval_align_mode,
    resolve_map_tile_contextily,
    sync_trajectories,
)
from epa.viz.plot_bundle import (
    make_plot_bundle,
    make_raw_line_spec,
    make_trajectory_error_map_spec,
    save_plot_bundle,
)
from epa.viz.plot_runtime import (
    configure_plot_runtime,
    should_enable_interactive_plot,
    show_plots,
)
from epa.viz.rerun_viz import log_metric_to_rerun


def _add_time_sync_args(p: argparse.ArgumentParser, suppress_defaults: bool = False) -> None:
    def dflt(value):
        return argparse.SUPPRESS if suppress_defaults else value

    p.add_argument(
        "--t_max_diff",
        type=float,
        default=dflt(0.01),
        help="maximum timestamp difference for data association",
    )
    p.add_argument(
        "--t_offset",
        type=float,
        default=dflt(0.0),
        help="constant timestamp offset for data association",
    )
    p.add_argument(
        "--t_start",
        type=float,
        default=dflt(None),
        help="only use data with timestamps >= this start time",
    )
    p.add_argument(
        "--t_end",
        type=float,
        default=dflt(None),
        help="only use data with timestamps <= this end time",
    )


def _resolve_plot_paths(out_dir: Path, save_plot: str) -> tuple[Path, Path]:
    if str(save_plot).strip():
        base = Path(save_plot).expanduser().resolve()
        suffix = base.suffix if base.suffix else ".png"
        stem = base.stem if base.suffix else base.name
        return (
            base.with_name(f"{stem}_raw{suffix}"),
            base.with_name(f"{stem}_map{suffix}"),
        )
    return out_dir / "rpe_raw.png", out_dir / "rpe_map.png"


def _pick_x_axis(arrays: dict[str, np.ndarray], x_dimension: str, n: int) -> tuple[np.ndarray, str]:
    if x_dimension == "seconds" and "seconds_from_start" in arrays:
        x = np.asarray(arrays["seconds_from_start"], dtype=float).reshape(-1)
        return x[:n], "t (s)"
    if x_dimension == "distances" and "distances_from_start" in arrays:
        x = np.asarray(arrays["distances_from_start"], dtype=float).reshape(-1)
        return x[:n], "d (m)"
    if "delta_ids" in arrays:
        x = np.asarray(arrays["delta_ids"], dtype=float).reshape(-1)
        return x[:n], "delta_ids"
    return np.arange(n, dtype=float), "index"


def _compute_color_bounds(
    values: np.ndarray,
    cmin: float | None,
    cmax: float | None,
    cmax_percentile: float | None,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite)) if cmin is None else float(cmin)
    if cmax_percentile is not None:
        vmax = float(np.percentile(finite, float(cmax_percentile)))
    else:
        vmax = float(np.max(finite)) if cmax is None else float(cmax)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax


def _plot_raw_errors(
    errors: np.ndarray,
    x_vals: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> matplotlib.figure.Figure | None:
    fig = plt.figure(figsize=(10.8, 4.8))
    plt.plot(x_vals, errors, linewidth=1.4)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


def _plot_map(
    pos_ref: np.ndarray,
    pos_est: np.ndarray,
    errors: np.ndarray,
    delta_ids: np.ndarray,
    plot_mode: str,
    title: str,
    out_path: Path,
    cmin: float | None,
    cmax: float | None,
    cmax_percentile: float | None,
    ros_map_yaml: str,
    map_tile: str,
) -> matplotlib.figure.Figure | None:
    mode = str(plot_mode).lower()
    did = np.asarray(delta_ids, dtype=int).reshape(-1)
    valid = (did >= 0) & (did < pos_est.shape[0])
    did = did[valid]
    errs = np.asarray(errors, dtype=float).reshape(-1)[: did.size]
    vmin, vmax = _compute_color_bounds(errs, cmin, cmax, cmax_percentile)

    if mode == "xyz":
        fig = plt.figure(figsize=(9.2, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], color="black", alpha=0.35, linewidth=1.2, label="ref")
        ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2], color="gray", alpha=0.2, linewidth=0.9, label="est")
        sc = ax.scatter(
            pos_est[did, 0],
            pos_est[did, 1],
            pos_est[did, 2],
            c=errs,
            s=12,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    else:
        axis_id = {"x": 0, "y": 1, "z": 2}
        if len(mode) != 2 or mode[0] not in axis_id or mode[1] not in axis_id:
            raise ValueError(f"Unsupported plot mode: {mode}")
        i0, i1 = axis_id[mode[0]], axis_id[mode[1]]
        fig = plt.figure(figsize=(9.2, 7.0))
        ax = fig.add_subplot(111)
        ax.plot(pos_ref[:, i0], pos_ref[:, i1], color="black", alpha=0.35, linewidth=1.2, label="ref")
        ax.plot(pos_est[:, i0], pos_est[:, i1], color="gray", alpha=0.2, linewidth=0.9, label="est")
        sc = ax.scatter(
            pos_est[did, i0],
            pos_est[did, i1],
            c=errs,
            s=12,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(f"{mode[0]} (m)")
        ax.set_ylabel(f"{mode[1]} (m)")
        ax.axis("equal")
        if str(ros_map_yaml).strip():
            if mode != "xy":
                raise ValueError("--ros_map_yaml currently requires --plot_mode xy")
            spec = load_ros_map_spec(ros_map_yaml)
            img = plt.imread(str(spec.image_path))
            h, w = img.shape[:2]
            extent = [
                spec.origin_x,
                spec.origin_x + float(w) * spec.resolution,
                spec.origin_y,
                spec.origin_y + float(h) * spec.resolution,
            ]
            ax.imshow(
                img,
                extent=extent,
                origin="lower",
                cmap="gray",
                alpha=0.45,
                zorder=-10,
            )
        if str(map_tile).strip():
            if mode != "xy":
                raise ValueError("--map_tile currently requires --plot_mode xy")
            try:
                import contextily as cx  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Map tile overlay requires contextily. Install with: pip install contextily"
                ) from exc
            cx.add_basemap(ax, **resolve_map_tile_contextily(cx, str(map_tile)))
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


def _add_common_args(p: argparse.ArgumentParser, suppress_defaults: bool = False) -> None:
    def dflt(value):
        return argparse.SUPPRESS if suppress_defaults else value

    p.add_argument(
        "-c",
        "--config",
        default=dflt(""),
        help="Path to JSON config file. If set, config values override CLI flags.",
    )
    algo = p.add_argument_group("algorithm options")
    algo.add_argument(
        "-r",
        "--pose_relation",
        default=dflt("trans_part"),
        choices=[
            "full",
            "trans_part",
            "rot_part",
            "angle_deg",
            "angle_rad",
            "point_distance",
            "point_distance_error_ratio",
        ],
        help="pose relation on which RPE is based",
    )
    algo.add_argument("-a", "--align", action="store_true", default=dflt(False), help="SE(3) Umeyama alignment")
    algo.add_argument("-s", "--correct_scale", action="store_true", default=dflt(False), help="enable scale correction")
    algo.add_argument(
        "--align_origin",
        action="store_true",
        default=dflt(False),
        help="align trajectory origin to reference origin",
    )
    algo.add_argument(
        "--n_to_align",
        type=int,
        default=dflt(-1),
        help="number of leading poses used for alignment (-1 means all)",
    )
    algo.add_argument(
        "--project_to_plane",
        choices=["none", "xy", "xz", "yz"],
        default=dflt("none"),
        help="project trajectories to plane before metric evaluation",
    )
    algo.add_argument(
        "--change_unit",
        default=dflt(None),
        choices=["m", "cm", "mm", "km", "rad", "deg", "%"],
        help="Change metric output unit when compatible with pose relation.",
    )
    algo.add_argument(
        "--downsample",
        type=int,
        default=dflt(0),
        help="Downsample trajectories to max N poses before synchronization.",
    )
    algo.add_argument(
        "--motion_filter",
        nargs=2,
        type=float,
        metavar=("DISTANCE", "ANGLE_DEGREES"),
        default=dflt(None),
        help="Filter poses by accumulated motion before synchronization.",
    )
    algo.add_argument("-d", "--delta", type=float, default=dflt(1.0), help="delta between relative poses")
    algo.add_argument(
        "-t",
        "--delta_tol",
        type=float,
        default=dflt(0.1),
        help="relative delta tolerance for all_pairs mode",
    )
    algo.add_argument(
        "-u",
        "--delta_unit",
        choices=["f", "m", "d", "r", "s"],
        default=dflt("f"),
        help="unit of delta: frames/meters/degrees/radians/seconds",
    )
    algo.add_argument("--all_pairs", action="store_true", default=dflt(False), help="use all candidate pairs")
    algo.add_argument(
        "--pairs_from_reference",
        action="store_true",
        default=dflt(False),
        help="build RPE pairs from reference trajectory",
    )

    output = p.add_argument_group("output options")
    output.add_argument(
        "-p",
        "--plot",
        action="store_true",
        default=dflt(False),
        help="generate raw/map plots (evo-style: in TTY sessions also opens interactive window)",
    )
    output.add_argument(
        "--plot_mode",
        choices=["xy", "xz", "yx", "yz", "zx", "zy", "xyz"],
        default=dflt("xyz"),
        help="axes for map projection",
    )
    output.add_argument(
        "--ros_map_yaml",
        default=dflt(None),
        help="ROS 2D map yaml (.yaml/.yml) overlay in xy plot mode.",
    )
    output.add_argument(
        "--map_tile",
        default=dflt(None),
        help="Contextily map source/URL or EPSG:xxxx CRS for xy plot mode.",
    )
    output.add_argument(
        "--plot_full_ref",
        action="store_true",
        default=dflt(False),
        help="Plot full unsynchronized reference trajectory in map view.",
    )
    output.add_argument(
        "--plot_x_dimension",
        choices=["index", "seconds", "distances"],
        default=dflt("seconds"),
        help="x-axis used in raw value plot",
    )
    output.add_argument(
        "--plot_colormap_min",
        type=float,
        default=dflt(None),
        help="lower bound for colormap",
    )
    output.add_argument(
        "--plot_colormap_max",
        type=float,
        default=dflt(None),
        help="upper bound for colormap",
    )
    output.add_argument(
        "--plot_colormap_max_percentile",
        type=float,
        default=dflt(None),
        help="use percentile as colormap max (overrides --plot_colormap_max)",
    )
    output.add_argument("--save_plot", default=dflt(""), help="path stem to save plot files")
    output.add_argument(
        "--plot_interactive",
        "--plot-interactive",
        action="store_true",
        default=dflt(False),
        help="Show interactive matplotlib window after generating plots.",
    )
    output.add_argument(
        "--plot_backend",
        "--plot-backend",
        default=dflt(""),
        help="Optional matplotlib backend override (e.g. qtagg, tkagg).",
    )
    output.add_argument(
        "--serialize_plot",
        "--serialize-plot",
        default=dflt(""),
        help="path to save serialized plot bundle JSON for later re-rendering",
    )
    output.add_argument("--rerun", action="store_true", default=dflt(False), help="Log visualization data to rerun.")
    output.add_argument(
        "--rerun_rec_id",
        "--rerun-rec-id",
        default=dflt(None),
        help="Use a specific recording ID for rerun.",
    )
    output.add_argument("--out_dir", default=dflt(""), help="output directory")
    output.add_argument("--save_results", default=dflt(""), help="path to save result zip bundle")
    usability = p.add_argument_group("usability options")
    usability.add_argument("--no_warnings", action="store_true", default=dflt(False), help="reserved for compatibility")
    usability.add_argument("-v", "--verbose", action="store_true", default=dflt(False), help="reserved for compatibility")
    usability.add_argument("--silent", action="store_true", default=dflt(False), help="reserved for compatibility")
    usability.add_argument("--debug", action="store_true", default=dflt(False), help="reserved for compatibility")
    usability.add_argument("--logfile", default=dflt(None), help="reserved for compatibility")
    _add_time_sync_args(p, suppress_defaults=suppress_defaults)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Relative pose error (RPE) tool.")
    _add_common_args(p)

    shared = argparse.ArgumentParser(add_help=False)
    _add_common_args(shared, suppress_defaults=True)

    subs = p.add_subparsers(dest="subcommand")
    subs.required = True

    kitti = subs.add_parser("kitti", help="KITTI poses", parents=[shared])
    kitti.add_argument("ref_file", help="reference pose file")
    kitti.add_argument("est_file", help="estimated pose file")

    tum = subs.add_parser("tum", help="TUM trajectories", parents=[shared])
    tum.add_argument("ref_file", help="reference trajectory file")
    tum.add_argument("est_file", help="estimated trajectory file")

    euroc = subs.add_parser("euroc", help="EuRoC files", parents=[shared])
    euroc.add_argument("state_gt_csv", help="ground truth csv")
    euroc.add_argument("est_file", help="estimated trajectory file in TUM format")

    bag = subs.add_parser("bag", help="ROS1 bag", parents=[shared])
    bag.add_argument("bag", help="bag path")
    bag.add_argument("ref_topic", help="reference trajectory topic")
    bag.add_argument("est_topic", help="estimated trajectory topic")

    bag2 = subs.add_parser("bag2", aliases=["mcap"], help="ROS2 bag / MCAP", parents=[shared])
    bag2.add_argument("bag", help="bag path")
    bag2.add_argument("ref_topic", help="reference trajectory topic")
    bag2.add_argument("est_topic", help="estimated trajectory topic")
    return p


def _prepare_inputs(args) -> tuple[MetricInputs, int]:
    data = load_inputs_from_args(args)
    t_ref, pos_ref, quat_ref = data.t_ref, data.pos_ref, data.quat_ref
    t_est, pos_est, quat_est = data.t_est, data.pos_est, data.quat_est

    t_start = getattr(args, "t_start", None)
    t_end = getattr(args, "t_end", None)
    if t_start is not None or t_end is not None:
        t_ref, pos_ref, quat_ref = apply_time_window(t_ref, pos_ref, quat_ref, t_start, t_end)
        t_est, pos_est, quat_est = apply_time_window(t_est, pos_est, quat_est, t_start, t_end)
    data.full_ref_t = np.asarray(t_ref, dtype=float).copy()
    data.full_ref_pos = np.asarray(pos_ref, dtype=float).copy()
    data.full_ref_quat = np.asarray(quat_ref, dtype=float).copy()

    ds = int(getattr(args, "downsample", 0) or 0)
    if ds > 0:
        t_ref, pos_ref, quat_ref = downsample_traj(t_ref, pos_ref, quat_ref, ds)
        t_est, pos_est, quat_est = downsample_traj(t_est, pos_est, quat_est, ds)

    mf = getattr(args, "motion_filter", None)
    if mf is not None:
        t_ref, pos_ref, quat_ref = motion_filter_traj(
            t_ref, pos_ref, quat_ref, float(mf[0]), float(mf[1])
        )
        t_est, pos_est, quat_est = motion_filter_traj(
            t_est, pos_est, quat_est, float(mf[0]), float(mf[1])
        )

    matched = min(t_ref.size, t_est.size)
    if data.subcommand != "kitti":
        t_ref, pos_ref, quat_ref, t_est, pos_est, quat_est = sync_trajectories(
            t_ref=t_ref,
            pos_ref=pos_ref,
            quat_ref=quat_ref,
            t_est=t_est,
            pos_est=pos_est,
            quat_est=quat_est,
            max_diff=float(getattr(args, "t_max_diff", 0.01)),
            offset=float(getattr(args, "t_offset", 0.0)),
        )
        matched = int(t_ref.size)
    else:
        if t_ref.size != t_est.size:
            raise ValueError(
                "KITTI mode requires equal pose counts (no timestamp sync available). "
                f"Got ref={t_ref.size}, est={t_est.size}."
            )

    data.t_ref = t_ref
    data.pos_ref = pos_ref
    data.quat_ref = quat_ref
    data.t_est = t_est
    data.pos_est = pos_est
    data.quat_est = quat_est
    return data, int(matched)


def run(args: argparse.Namespace) -> int:
    if bool(getattr(args, "align_origin", False)) and bool(getattr(args, "align", False)):
        raise ValueError("--align and --align_origin cannot be used together.")

    data, matched = _prepare_inputs(args)
    align_mode = resolve_eval_align_mode(args)
    n_to_align = int(getattr(args, "n_to_align", -1))
    pose_relation = normalize_pose_relation("rpe", getattr(args, "pose_relation", "trans_part"))
    project_plane = str(getattr(args, "project_to_plane", "none"))

    est_aligned_pos, est_aligned_quat = align_for_eval(
        pos_ref=data.pos_ref,
        quat_ref=data.quat_ref,
        pos_est=data.pos_est,
        quat_est=data.quat_est,
        mode=align_mode,
        n_to_align=n_to_align,
    )
    ref_eval_pos, ref_eval_quat = project_to_plane(data.pos_ref, data.quat_ref, project_plane)
    est_eval_pos, est_eval_quat = project_to_plane(est_aligned_pos, est_aligned_quat, project_plane)

    rpe_block = compute_rpe_evo_style(
        pos_ref=ref_eval_pos,
        quat_ref=ref_eval_quat,
        pos_est=est_eval_pos,
        quat_est=est_eval_quat,
        delta=float(getattr(args, "delta", 1.0)),
        delta_unit=str(getattr(args, "delta_unit", "f")),
        rel_delta_tol=float(getattr(args, "delta_tol", 0.1)),
        all_pairs=bool(getattr(args, "all_pairs", False)),
        pairs_from_reference=bool(getattr(args, "pairs_from_reference", False)),
        timestamps=np.asarray(data.t_ref, dtype=float),
        include_raw=True,
    )
    seconds_from_start = np.asarray(data.t_ref, dtype=float) - float(data.t_ref[0])
    distances_from_start = cum_distance(ref_eval_pos)
    delta_ids = np.asarray(rpe_block["_x_axis"]["delta_ids"], dtype=int).reshape(-1)
    valid = (delta_ids >= 0) & (delta_ids < seconds_from_start.size)
    rpe_block["_x_axis"]["seconds_from_start"] = seconds_from_start[delta_ids[valid]]
    rpe_block["_x_axis"]["distances_from_start"] = distances_from_start[delta_ids[valid]]

    selected_unit = RELATION_UNITS.get(pose_relation, "")
    target_unit = getattr(args, "change_unit", None)
    if target_unit:
        factor = None
        if pose_relation in {"translation_part", "point_distance"}:
            table = {"m": 1.0, "cm": 100.0, "mm": 1000.0, "km": 0.001}
            if target_unit in table:
                factor = table[target_unit]
                selected_unit = target_unit
        elif pose_relation == "rotation_angle_rad":
            if target_unit == "deg":
                factor = float(180.0 / np.pi)
                selected_unit = "deg"
            elif target_unit == "rad":
                factor = 1.0
                selected_unit = "rad"
        elif pose_relation == "rotation_angle_deg":
            if target_unit == "rad":
                factor = float(np.pi / 180.0)
                selected_unit = "rad"
            elif target_unit == "deg":
                factor = 1.0
                selected_unit = "deg"
        elif pose_relation == "point_distance_error_ratio":
            if target_unit == "%":
                factor = 1.0
                selected_unit = "%"
        if factor is None:
            raise ValueError(
                f"--change_unit {target_unit} is incompatible with pose relation {pose_relation}"
            )
        for k in ("rmse", "mean", "median", "std", "min", "max", "sse"):
            rpe_block[pose_relation][k] = float(rpe_block[pose_relation][k]) * factor
        rpe_block["_error_arrays"][pose_relation] = (
            np.asarray(rpe_block["_error_arrays"][pose_relation], dtype=float) * factor
        )

    stat = rpe_block[pose_relation]
    unit = selected_unit
    unit_suffix = f" {unit}" if unit else ""
    pair_count = int(rpe_block["pair_count"])
    print(
        f"RPE {pose_relation}: rmse={float(stat['rmse']):.6f}{unit_suffix}, "
        f"mean={float(stat['mean']):.6f}{unit_suffix}, "
        f"median={float(stat['median']):.6f}{unit_suffix}, "
        f"pairs={pair_count}"
    )
    print(f"Aligned pairs: {matched}")

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else default_out_dir("rpe", Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metric_summary": {
            f"rpe_{pose_relation}_rmse": float(stat["rmse"]),
            f"rpe_{pose_relation}_mean": float(stat["mean"]),
            f"rpe_{pose_relation}_median": float(stat["median"]),
            "pair_count": float(pair_count),
            "matched_pairs": float(matched),
        },
        "pose_metrics": {
            "ape": {},
            "rpe": {"raw": rpe_block},
            "rpe_config": {
                "delta": float(getattr(args, "delta", 1.0)),
                "delta_unit": str(getattr(args, "delta_unit", "f")),
                "delta_tol": float(getattr(args, "delta_tol", 0.1)),
                "all_pairs": bool(getattr(args, "all_pairs", False)),
                "pairs_from_reference": bool(getattr(args, "pairs_from_reference", False)),
            },
            "eval_config": {
                "metric": "rpe",
                "pose_relation": pose_relation,
                "align": align_mode,
                "n_to_align": n_to_align,
                "project_to_plane": project_plane,
                "change_unit": target_unit,
                "ros_map_yaml": None if not getattr(args, "ros_map_yaml", None) else str(args.ros_map_yaml),
                "map_tile": None if not getattr(args, "map_tile", None) else str(args.map_tile),
                "t_max_diff": None if data.subcommand == "kitti" else float(getattr(args, "t_max_diff", 0.01)),
                "t_offset": None if data.subcommand == "kitti" else float(getattr(args, "t_offset", 0.0)),
                "t_start": None if getattr(args, "t_start", None) is None else float(args.t_start),
                "t_end": None if getattr(args, "t_end", None) is None else float(args.t_end),
            },
        },
        "metadata": {
            "tool": "epa_rpe",
            "subcommand": data.subcommand,
            "ref_name": data.ref_name,
            "est_name": data.est_name,
            "output_dir": str(out_dir),
        },
    }

    metrics_json = out_dir / "metrics.json"
    save_metrics(out_dir, payload)
    print(f"Saved metrics: {metrics_json}")

    plot_files: list[Path] = []
    raw_spec: dict | None = None
    map_spec: dict | None = None
    interactive_enabled = False
    need_plot_data = (
        bool(getattr(args, "plot", False))
        or str(getattr(args, "save_plot", "")).strip()
        or str(getattr(args, "serialize_plot", "")).strip()
    )
    if need_plot_data:
        interactive_requested = should_enable_interactive_plot(
            plot=bool(getattr(args, "plot", False)),
            plot_interactive=bool(getattr(args, "plot_interactive", False)),
        )
        interactive_enabled, active_backend = configure_plot_runtime(
            plot_interactive=interactive_requested,
            plot_backend=str(getattr(args, "plot_backend", "") or ""),
        )
        if interactive_requested:
            if interactive_enabled:
                print(f"[plot] Interactive backend: {active_backend}")
            else:
                print("[plot] Interactive mode requested but no interactive backend is available.")
        errs = np.asarray(rpe_block["_error_arrays"][pose_relation], dtype=float).reshape(-1)
        x_vals, x_label = _pick_x_axis(
            rpe_block["_x_axis"],
            str(getattr(args, "plot_x_dimension", "seconds")),
            errs.size,
        )
        delta_ids = np.asarray(rpe_block["_x_axis"]["delta_ids"], dtype=int).reshape(-1)
        valid = (delta_ids >= 0) & (delta_ids < est_eval_pos.shape[0])
        scatter_pos = est_eval_pos[delta_ids[valid]]
        n_scatter = min(scatter_pos.shape[0], errs.size)
        scatter_pos = scatter_pos[:n_scatter]
        errs_scatter = errs[:n_scatter]
        ylabel = f"error ({unit})" if unit else "error"
        map_ref = (
            project_to_plane(
                np.asarray(data.full_ref_pos, dtype=float),
                np.asarray(data.full_ref_quat, dtype=float),
                project_plane,
            )[0]
            if bool(getattr(args, "plot_full_ref", False))
            and data.full_ref_pos is not None
            and data.full_ref_quat is not None
            else ref_eval_pos
        )
        raw_spec = make_raw_line_spec(
            name="raw",
            title=f"RPE raw ({pose_relation})",
            x=x_vals,
            y=errs,
            x_label=x_label,
            y_label=ylabel,
        )
        map_spec = make_trajectory_error_map_spec(
            name="map",
            title=f"RPE map ({pose_relation})",
            plot_mode=str(getattr(args, "plot_mode", "xyz")),
            ref_positions=map_ref,
            est_positions=est_eval_pos,
            scatter_positions=scatter_pos,
            errors=errs_scatter,
            cmin=getattr(args, "plot_colormap_min", None),
            cmax=getattr(args, "plot_colormap_max", None),
            cmax_percentile=getattr(args, "plot_colormap_max_percentile", None),
            ros_map_yaml=str(getattr(args, "ros_map_yaml", "") or ""),
            map_tile=str(getattr(args, "map_tile", "") or ""),
        )
        map_spec["scatter_indices"] = delta_ids[valid][:n_scatter].tolist()

    if bool(getattr(args, "plot", False)) or str(getattr(args, "save_plot", "")).strip():
        open_figures: list[matplotlib.figure.Figure] = []
        raw_plot, map_plot = _resolve_plot_paths(
            out_dir=out_dir,
            save_plot=str(getattr(args, "save_plot", "")),
        )
        raw_plot.parent.mkdir(parents=True, exist_ok=True)
        map_plot.parent.mkdir(parents=True, exist_ok=True)
        assert raw_spec is not None and map_spec is not None
        raw_fig = _plot_raw_errors(
            errors=np.asarray(raw_spec["y"], dtype=float),
            x_vals=np.asarray(raw_spec["x"], dtype=float),
            x_label=str(raw_spec["x_label"]),
            y_label=str(raw_spec["y_label"]),
            title=str(raw_spec["title"]),
            out_path=raw_plot,
        )
        if raw_fig is not None:
            open_figures.append(raw_fig)
        full_est = np.asarray(map_spec["est_positions"], dtype=float)
        delta_ids = np.asarray(map_spec.get("scatter_indices", []), dtype=int).reshape(-1)
        map_fig = _plot_map(
            pos_ref=np.asarray(map_spec["ref_positions"], dtype=float),
            pos_est=full_est,
            errors=np.asarray(map_spec["errors"], dtype=float),
            delta_ids=delta_ids,
            plot_mode=str(map_spec["plot_mode"]),
            title=str(map_spec["title"]),
            out_path=map_plot,
            cmin=map_spec.get("cmin", None),
            cmax=map_spec.get("cmax", None),
            cmax_percentile=map_spec.get("cmax_percentile", None),
            ros_map_yaml=str(map_spec.get("ros_map_yaml", "") or ""),
            map_tile=str(map_spec.get("map_tile", "") or ""),
        )
        if map_fig is not None:
            open_figures.append(map_fig)
        plot_files = [raw_plot, map_plot]
        payload["metadata"]["plot_files"] = [str(p) for p in plot_files]
        for p in plot_files:
            print(f"Saved figure: {p}")
        if interactive_enabled and open_figures:
            show_plots()
        for fig in open_figures:
            plt.close(fig)

    if str(getattr(args, "serialize_plot", "")).strip():
        assert raw_spec is not None and map_spec is not None
        bundle_path = save_plot_bundle(
            make_plot_bundle(
                tool="epa_rpe",
                figures=[raw_spec, map_spec],
                metadata={
                    "metric": "rpe",
                    "pose_relation": pose_relation,
                    "ref_name": data.ref_name,
                    "est_name": data.est_name,
                },
            ),
            str(getattr(args, "serialize_plot")),
        )
        payload["metadata"]["serialize_plot"] = str(bundle_path)
        print(f"Saved serialized plot: {bundle_path}")

    rerun_info = {
        "enabled": "false",
        "status": "disabled",
        "message": "Use --rerun to enable.",
    }
    if bool(getattr(args, "rerun", False)):
        errs = np.asarray(rpe_block["_error_arrays"][pose_relation], dtype=float).reshape(-1)
        delta_ids = np.asarray(rpe_block["_x_axis"]["delta_ids"], dtype=int).reshape(-1)
        valid = (delta_ids >= 0) & (delta_ids < est_eval_pos.shape[0])
        idx = delta_ids[valid][: errs.size]
        x_vals, _ = _pick_x_axis(
            rpe_block["_x_axis"],
            str(getattr(args, "plot_x_dimension", "seconds")),
            idx.size,
        )
        time_is_seconds = str(getattr(args, "plot_x_dimension", "seconds")) == "seconds"
        rerun_info = log_metric_to_rerun(
            metric_name="rpe",
            relation=pose_relation,
            ref_positions=ref_eval_pos,
            est_positions=est_eval_pos,
            errors=errs[: idx.size],
            ref_quaternions=ref_eval_quat,
            est_quaternions=est_eval_quat,
            sample_indices=idx,
            timeline_values=x_vals,
            timeline_is_seconds=time_is_seconds,
            app_id="epa_rpe",
            spawn=True,
            recording_id=(
                None
                if getattr(args, "rerun_rec_id", None) in {None, ""}
                else str(getattr(args, "rerun_rec_id"))
            ),
        )
        print(f"Rerun status: {rerun_info['status']} - {rerun_info['message']}")
    payload["metadata"]["rerun"] = rerun_info
    metrics_json.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")

    if str(getattr(args, "save_results", "")).strip():
        bundle_path = Path(str(args.save_results)).expanduser().resolve()
        write_result_bundle(out_dir, payload, bundle_path)
        print(f"Saved results: {bundle_path}")

    return 0


def main() -> int:
    parser = _build_parser()
    args = parse_args_with_config(parser, config_dest="config", tool_name="epa_rpe")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
