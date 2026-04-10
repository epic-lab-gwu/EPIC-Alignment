from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.cm
from matplotlib.colors import Normalize, rgb2hex


def _as_line_strip(points_xyz: np.ndarray, stride: int) -> np.ndarray:
    if points_xyz.shape[0] <= 1:
        return points_xyz
    return points_xyz[:: max(1, int(stride))]


def _as_motion_samples(points_xyz: np.ndarray, stride: int) -> np.ndarray:
    return points_xyz[:: max(1, int(stride))]


def _set_timeline(rr, t_val: float, use_time_seconds: bool) -> None:
    # rerun >= 0.31 uses a unified rr.set_time API.
    if hasattr(rr, "set_time"):
        if use_time_seconds:
            rr.set_time("time", duration=float(t_val))
        else:
            rr.set_time("index", sequence=int(t_val))
        return

    # Backward compatibility with older APIs.
    if use_time_seconds and hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("time", float(t_val))
    elif (not use_time_seconds) and hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("index", int(t_val))
    else:
        raise AttributeError(
            "Rerun SDK has no compatible timeline setter (need set_time or legacy set_time_*)."
        )


def _timeline_column(rr, timeline_name: str, values: np.ndarray):
    values = np.asarray(values)
    if timeline_name == "time":
        return rr.TimeColumn("time", timestamp=values.astype(float))
    return rr.TimeColumn("index", sequence=values.astype(np.int64))


def _rgb_to_u32(colors_rgb: np.ndarray) -> list[int]:
    rgb = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
    return [int((int(r) << 24) | (int(g) << 16) | (int(b) << 8) | 255) for r, g, b in rgb]


def _mapped_colors(cmap_name: str, values: np.ndarray) -> list[int]:
    vals = np.asarray(values, dtype=float).reshape(-1)
    if vals.size == 0:
        return []
    norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()), clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_name)
    mapper.set_array(vals)
    return [
        int(f"0x{rgb2hex(tuple(mapper.to_rgba(v)), keep_alpha=True).strip('#')}", base=16)
        for v in vals
    ]


def _send_timed_points(
    rr,
    *,
    entity_path: str,
    positions: np.ndarray,
    timeline_name: str,
    timeline_values: np.ndarray,
    color_u32: int,
    radius: float,
) -> None:
    pos = np.asarray(positions, dtype=float)
    tvals = np.asarray(timeline_values, dtype=float).reshape(-1)
    if pos.shape[0] == 0 or tvals.size == 0:
        return
    n = min(pos.shape[0], tvals.size)
    pos = pos[:n]
    tvals = tvals[:n]
    rr.send_columns(
        entity_path,
        indexes=[_timeline_column(rr, timeline_name, tvals)],
        columns=[*rr.Points3D.columns(positions=pos, colors=[color_u32] * n)],
    )
    rr.log(
        entity_path,
        rr.Points3D.from_fields(radii=[radius]),
        static=True,
    )


def _send_timed_line_strips(
    rr,
    *,
    entity_path: str,
    positions: np.ndarray,
    timeline_name: str,
    timeline_values: np.ndarray,
    radii: float,
    colors_u32: Optional[list[int]] = None,
    static_color_rgba: Optional[list[int]] = None,
) -> None:
    pos = np.asarray(positions, dtype=float)
    tvals = np.asarray(timeline_values, dtype=float).reshape(-1)
    if pos.shape[0] < 2 or tvals.size < 2:
        return
    n = min(pos.shape[0], tvals.size)
    pos = pos[:n]
    tvals = tvals[:n]
    strips = [[a, b] for a, b in zip(pos[:-1], pos[1:])]
    if colors_u32 is not None:
        seg_colors = colors_u32[: len(strips)]
    else:
        seg_colors = None

    columns = rr.LineStrips3D.columns(strips=strips, colors=seg_colors)
    rr.send_columns(
        entity_path,
        indexes=[_timeline_column(rr, timeline_name, tvals[1:])],
        columns=[*columns],
    )
    if static_color_rgba is not None:
        rr.log(
            entity_path,
            rr.LineStrips3D.from_fields(colors=[static_color_rgba], radii=[radii]),
            static=True,
        )
    else:
        rr.log(
            entity_path,
            rr.LineStrips3D.from_fields(radii=[radii]),
            static=True,
        )


def _send_timed_transforms(
    rr,
    *,
    entity_path: str,
    positions: np.ndarray,
    quats_xyzw: Optional[np.ndarray],
    timeline_name: str,
    timeline_values: np.ndarray,
    axis_length: float = 0.04,
) -> None:
    pos = np.asarray(positions, dtype=float)
    tvals = np.asarray(timeline_values, dtype=float).reshape(-1)
    if pos.shape[0] == 0 or tvals.size == 0:
        return
    n = min(pos.shape[0], tvals.size)
    pos = pos[:n]
    tvals = tvals[:n]
    if quats_xyzw is not None:
        quat = np.asarray(quats_xyzw, dtype=float)
        if quat.shape[0] >= n:
            rr.send_columns(
                entity_path,
                indexes=[_timeline_column(rr, timeline_name, tvals)],
                columns=rr.Transform3D.columns(
                    translation=pos,
                    quaternion=quat[:n],
                ),
            )
        else:
            rr.send_columns(
                entity_path,
                indexes=[_timeline_column(rr, timeline_name, tvals)],
                columns=rr.Transform3D.columns(translation=pos),
            )
    else:
        rr.send_columns(
            entity_path,
            indexes=[_timeline_column(rr, timeline_name, tvals)],
            columns=rr.Transform3D.columns(translation=pos),
        )
    rr.log(entity_path, rr.TransformAxes3D.from_fields(axis_length=axis_length), static=True)


def _send_scalar_series(
    rr,
    *,
    entity_path: str,
    values: np.ndarray,
    timeline_name: str,
    timeline_values: np.ndarray,
    color_rgba,
    label: str,
) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    timeline_values = np.asarray(timeline_values).reshape(-1)
    if values.size == 0 or timeline_values.size == 0:
        return
    if values.size != timeline_values.size:
        min_size = min(values.size, timeline_values.size)
        values = values[:min_size]
        timeline_values = timeline_values[:min_size]

    rr.send_columns(
        entity_path,
        indexes=[_timeline_column(rr, timeline_name, timeline_values)],
        columns=rr.Scalars.columns(scalars=values),
    )
    rr.log(
        entity_path,
        rr.SeriesLines(colors=[color_rgba], names=[label]),
        static=True,
    )


def _send_xyz_rpy_speed(
    rr,
    *,
    base_path: str,
    name: str,
    positions: np.ndarray,
    quats_xyzw: Optional[np.ndarray],
    timeline_name: str,
    timeline_values: np.ndarray,
    color_rgba,
) -> None:
    pos = np.asarray(positions, dtype=float)
    t = np.asarray(timeline_values)

    _send_scalar_series(
        rr,
        entity_path=f"{base_path}/x/{name}",
        values=pos[:, 0],
        timeline_name=timeline_name,
        timeline_values=t,
        color_rgba=color_rgba,
        label=f"{name}.x",
    )
    _send_scalar_series(
        rr,
        entity_path=f"{base_path}/y/{name}",
        values=pos[:, 1],
        timeline_name=timeline_name,
        timeline_values=t,
        color_rgba=color_rgba,
        label=f"{name}.y",
    )
    _send_scalar_series(
        rr,
        entity_path=f"{base_path}/z/{name}",
        values=pos[:, 2],
        timeline_name=timeline_name,
        timeline_values=t,
        color_rgba=color_rgba,
        label=f"{name}.z",
    )

    if quats_xyzw is not None:
        quat = np.asarray(quats_xyzw, dtype=float)
        if quat.shape[0] == pos.shape[0]:
            euler_deg = R.from_quat(quat).as_euler("xyz", degrees=True)
            _send_scalar_series(
                rr,
                entity_path=f"{base_path}/roll/{name}",
                values=euler_deg[:, 0],
                timeline_name=timeline_name,
                timeline_values=t,
                color_rgba=color_rgba,
                label=f"{name}.roll",
            )
            _send_scalar_series(
                rr,
                entity_path=f"{base_path}/pitch/{name}",
                values=euler_deg[:, 1],
                timeline_name=timeline_name,
                timeline_values=t,
                color_rgba=color_rgba,
                label=f"{name}.pitch",
            )
            _send_scalar_series(
                rr,
                entity_path=f"{base_path}/yaw/{name}",
                values=euler_deg[:, 2],
                timeline_name=timeline_name,
                timeline_values=t,
                color_rgba=color_rgba,
                label=f"{name}.yaw",
            )

    if pos.shape[0] > 1 and t.size > 1:
        dt = np.diff(t.astype(float))
        dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.abs(dt)
        _send_scalar_series(
            rr,
            entity_path=f"{base_path}/speed/{name}",
            values=speed,
            timeline_name=timeline_name,
            timeline_values=t[1:],
            color_rgba=color_rgba,
            label=f"{name}.speed",
        )


def _send_statistics_bars(rr, stats_path: str, stage_stats: Dict[str, Dict[str, float]]) -> None:
    keys = ["rmse", "mean", "median", "std", "p95", "max"]
    colors = {
        "raw": [235, 70, 70, 255],
        "step2": [150, 150, 150, 255],
        "step3": [120, 155, 205, 255],
    }
    for stage, values in stage_stats.items():
        for idx, key in enumerate(keys):
            if key not in values:
                continue
            val = float(values[key])
            if not np.isfinite(val):
                continue
            rr.log(
                f"{stats_path}/{stage}_{key}",
                rr.BarChart(
                    values=np.array([val], dtype=float),
                    abscissa=np.array([idx], dtype=float),
                    color=colors.get(stage, [180, 180, 180, 255]),
                ),
                static=True,
            )


def _gradient_rgb(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    if vals.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if np.allclose(vals.max(), vals.min()):
        norm = np.linspace(0.0, 1.0, vals.size)
    else:
        norm = (vals - vals.min()) / (vals.max() - vals.min())

    anchors = np.array(
        [
            [59.0, 76.0, 192.0],
            [64.0, 142.0, 255.0],
            [42.0, 191.0, 127.0],
            [248.0, 209.0, 70.0],
            [220.0, 53.0, 69.0],
        ]
    )
    anchor_x = np.linspace(0.0, 1.0, anchors.shape[0])
    rgb = np.stack(
        [
            np.interp(norm, anchor_x, anchors[:, 0]),
            np.interp(norm, anchor_x, anchors[:, 1]),
            np.interp(norm, anchor_x, anchors[:, 2]),
        ],
        axis=1,
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _send_blueprint(timeline_name: str):
    import rerun as rr
    import rerun.blueprint as rrb

    history_range = rrb.VisibleTimeRange(
        timeline=timeline_name,
        start=rrb.TimeRangeBoundary.infinite(),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=0.0 if timeline_name == "time" else 0),
    )
    xyz_views = []
    for label in ("x", "y", "z"):
        xyz_views.append(
            rrb.TimeSeriesView(
                name=label.upper(),
                time_ranges=history_range,
                plot_legend=rrb.PlotLegend(visible=False),
                contents=[f"/vicon_ws/time_series/{label}/**"],
                axis_x=rrb.TimeAxis(link="LinkToGlobal"),
            )
        )

    rpy_views = []
    for label in ("roll", "pitch", "yaw"):
        rpy_views.append(
            rrb.TimeSeriesView(
                name=label.capitalize(),
                time_ranges=history_range,
                plot_legend=rrb.PlotLegend(visible=False),
                contents=[f"/vicon_ws/time_series/{label}/**"],
                axis_x=rrb.TimeAxis(link="LinkToGlobal"),
            )
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Tabs(
                contents=[
                    rrb.Grid(
                        name="Visualization",
                        contents=[
                            rrb.Spatial3DView(
                                name="Trajectories",
                                time_ranges=history_range,
                                contents=["/vicon_ws/replay/**"],
                            ),
                            rrb.Grid(
                                name="Metrics",
                                contents=[
                                    rrb.TimeSeriesView(
                                        name="ATE Error",
                                        time_ranges=history_range,
                                        plot_legend=rrb.Corner2D.RightTop,
                                        contents=["/vicon_ws/error/scalars/**"],
                                        axis_x=rrb.TimeAxis(link="LinkToGlobal"),
                                    ),
                                    rrb.BarChartView(
                                        name="Statistics",
                                        origin="/vicon_ws/error/statistics",
                                        plot_legend=rrb.PlotLegend(None, visible=False),
                                    ),
                                ],
                                column_shares=[1, 1],
                            ),
                        ],
                        column_shares=None,
                        row_shares=[2.5, 1.0],
                        grid_columns=1,
                    ),
                    rrb.Grid(
                        name="Trajectory Series",
                        contents=[
                            rrb.Vertical(name="Position", contents=xyz_views),
                            rrb.Vertical(name="Orientation", contents=rpy_views),
                            rrb.TimeSeriesView(
                                name="Speed",
                                time_ranges=history_range,
                                plot_legend=rrb.Corner2D.RightTop,
                                contents=["/vicon_ws/time_series/speed/**"],
                                axis_x=rrb.TimeAxis(link="LinkToGlobal"),
                            ),
                        ],
                        column_shares=1,
                    ),
                    rrb.DataframeView(
                        name="Raw Data",
                        origin="/vicon_ws",
                        contents=[
                            "$origin/trajectory/**",
                            "$origin/replay/**",
                            "$origin/error/scalars/**",
                            "$origin/time_series/**",
                        ],
                    ),
                ]
            ),
            rrb.SelectionPanel(expanded=False),
            rrb.TimePanel(expanded=False),
        )
    )


def log_alignment_to_rerun(
    *,
    run_dir: Path,
    gt_xyz: np.ndarray,
    raw_xyz: np.ndarray,
    step2_xyz: np.ndarray,
    step3_xyz: np.ndarray,
    timestamps_s: Optional[np.ndarray] = None,
    gt_quat: Optional[np.ndarray] = None,
    raw_quat: Optional[np.ndarray] = None,
    step2_quat: Optional[np.ndarray] = None,
    step3_quat: Optional[np.ndarray] = None,
    raw_error_m: Optional[np.ndarray] = None,
    step2_error_m: Optional[np.ndarray] = None,
    step3_error_m: Optional[np.ndarray] = None,
    app_id: str = "vicon_ws_alignment",
    spawn: bool = True,
    stride: int = 20,
    motion_stride: int = 5,
) -> Dict[str, str]:
    """
    Send alignment trajectories and metrics to Rerun if rerun-sdk is installed.
    Returns a status dict that can be added to metrics metadata.
    """
    try:
        import rerun as rr
    except Exception as exc:
        return {
            "enabled": "false",
            "status": "rerun_sdk_missing",
            "message": f"Install rerun-sdk to enable visualization ({exc})",
        }

    try:
        # Initialize recording first, then spawn an SDK-matching viewer binary when requested.
        rr.init(app_id, spawn=False)
        if spawn:
            env_rerun = Path(sys.executable).resolve().parent / "rerun"
            if env_rerun.exists():
                rr.spawn(executable_path=str(env_rerun))
            else:
                rr.spawn()

        gt = np.asarray(gt_xyz, dtype=float)
        raw = np.asarray(raw_xyz, dtype=float)
        step2 = np.asarray(step2_xyz, dtype=float)
        step3 = np.asarray(step3_xyz, dtype=float)

        if not (len(gt) == len(raw) == len(step2) == len(step3)):
            raise ValueError("All trajectories must have the same number of poses for replay.")

        if timestamps_s is None:
            t_full = np.arange(len(gt), dtype=float)
            use_time_seconds = False
        else:
            t_full = np.asarray(timestamps_s, dtype=float)
            if len(t_full) != len(gt):
                raise ValueError("timestamps_s length must match trajectory length.")
            t_full = t_full - t_full[0]
            use_time_seconds = True

        timeline_name = "time" if use_time_seconds else "index"
        try:
            _send_blueprint(timeline_name)
        except Exception:
            # Keep logging robust even if blueprint APIs change.
            pass

        # evo-style time-indexed replay streams.
        replay_step = max(1, int(motion_stride))
        t_replay = t_full[::replay_step]
        gt_replay = gt[::replay_step]
        raw_replay = raw[::replay_step]
        step3_replay = step3[::replay_step]

        gt_quat_replay = (
            np.asarray(gt_quat, dtype=float)[::replay_step]
            if gt_quat is not None
            else None
        )
        raw_quat_replay = (
            np.asarray(raw_quat, dtype=float)[::replay_step]
            if raw_quat is not None
            else None
        )
        step3_quat_replay = (
            np.asarray(step3_quat, dtype=float)[::replay_step]
            if step3_quat is not None
            else None
        )

        # evo-like playback: trajectory grows by time; a moving pose marker indicates current frame.
        _send_timed_transforms(
            rr,
            entity_path="vicon_ws/replay/gt/pose",
            positions=gt_replay,
            quats_xyzw=gt_quat_replay,
            timeline_name=timeline_name,
            timeline_values=t_replay,
            axis_length=0.035,
        )
        _send_timed_transforms(
            rr,
            entity_path="vicon_ws/replay/raw/pose",
            positions=raw_replay,
            quats_xyzw=raw_quat_replay,
            timeline_name=timeline_name,
            timeline_values=t_replay,
            axis_length=0.035,
        )
        _send_timed_transforms(
            rr,
            entity_path="vicon_ws/replay/step3/pose",
            positions=step3_replay,
            quats_xyzw=step3_quat_replay,
            timeline_name=timeline_name,
            timeline_values=t_replay,
            axis_length=0.04,
        )

        _send_timed_line_strips(
            rr,
            entity_path="vicon_ws/replay/gt/lines",
            positions=gt_replay,
            timeline_name=timeline_name,
            timeline_values=t_replay,
            radii=0.0011,
            static_color_rgba=[0, 210, 0, 210],
        )
        _send_timed_line_strips(
            rr,
            entity_path="vicon_ws/replay/raw/lines",
            positions=raw_replay,
            timeline_name=timeline_name,
            timeline_values=t_replay,
            radii=0.0010,
            static_color_rgba=[235, 70, 70, 170],
        )

        # evo-like visual style: same mapping idea as evo_ape/evo_rpe in rerun:
        # map per-pose error values through the configured trajectory colormap (jet by default).
        if step3_error_m is not None:
            err_replay = np.asarray(step3_error_m, dtype=float).reshape(-1)[::replay_step]
            if err_replay.size >= 2:
                # line segments are between pose i-1 -> i, so use error at i
                step3_seg_colors = _mapped_colors("jet", err_replay[1:])
            else:
                step3_progress = np.linspace(0.0, 1.0, max(1, len(step3_replay) - 1))
                step3_seg_colors = _mapped_colors("jet", step3_progress)
        else:
            step3_progress = np.linspace(0.0, 1.0, max(1, len(step3_replay) - 1))
            step3_seg_colors = _mapped_colors("jet", step3_progress)

        _send_timed_line_strips(
            rr,
            entity_path="vicon_ws/replay/step3/lines",
            positions=step3_replay,
            timeline_name=timeline_name,
            timeline_values=t_replay,
            radii=0.0017,
            colors_u32=step3_seg_colors,
        )

        # Error time-series (evo-like metrics viewer section).
        if raw_error_m is not None:
            _send_scalar_series(
                rr,
                entity_path="vicon_ws/error/scalars/raw_ate_m",
                values=np.asarray(raw_error_m, dtype=float),
                timeline_name=timeline_name,
                timeline_values=t_full,
                color_rgba=[235, 70, 70, 255],
                label="raw ATE (m)",
            )
        if step2_error_m is not None:
            _send_scalar_series(
                rr,
                entity_path="vicon_ws/error/scalars/step2_ate_m",
                values=np.asarray(step2_error_m, dtype=float),
                timeline_name=timeline_name,
                timeline_values=t_full,
                color_rgba=[150, 150, 150, 255],
                label="step2 ATE (m)",
            )
        if step3_error_m is not None:
            _send_scalar_series(
                rr,
                entity_path="vicon_ws/error/scalars/step3_ate_m",
                values=np.asarray(step3_error_m, dtype=float),
                timeline_name=timeline_name,
                timeline_values=t_full,
                color_rgba=[120, 155, 205, 255],
                label="step3 ATE (m)",
            )

        # Trajectory x/y/z, roll/pitch/yaw and speed (evo_traj-like plotting support).
        _send_xyz_rpy_speed(
            rr,
            base_path="vicon_ws/time_series",
            name="gt",
            positions=gt,
            quats_xyzw=gt_quat,
            timeline_name=timeline_name,
            timeline_values=t_full,
            color_rgba=[0, 210, 0, 255],
        )
        _send_xyz_rpy_speed(
            rr,
            base_path="vicon_ws/time_series",
            name="raw",
            positions=raw,
            quats_xyzw=raw_quat,
            timeline_name=timeline_name,
            timeline_values=t_full,
            color_rgba=[235, 70, 70, 255],
        )
        _send_xyz_rpy_speed(
            rr,
            base_path="vicon_ws/time_series",
            name="step2",
            positions=step2,
            quats_xyzw=step2_quat,
            timeline_name=timeline_name,
            timeline_values=t_full,
            color_rgba=[150, 150, 150, 255],
        )
        _send_xyz_rpy_speed(
            rr,
            base_path="vicon_ws/time_series",
            name="step3",
            positions=step3,
            quats_xyzw=step3_quat,
            timeline_name=timeline_name,
            timeline_values=t_full,
            color_rgba=[120, 155, 205, 255],
        )

        # Static statistics bars.
        stage_stats = {}
        if raw_error_m is not None:
            raw_vals = np.asarray(raw_error_m, dtype=float)
            stage_stats["raw"] = {
                "rmse": float(np.sqrt(np.mean(raw_vals ** 2))),
                "mean": float(np.mean(raw_vals)),
                "median": float(np.median(raw_vals)),
                "std": float(np.std(raw_vals)),
                "p95": float(np.percentile(raw_vals, 95)),
                "max": float(np.max(raw_vals)),
            }
        if step2_error_m is not None:
            step2_vals = np.asarray(step2_error_m, dtype=float)
            stage_stats["step2"] = {
                "rmse": float(np.sqrt(np.mean(step2_vals ** 2))),
                "mean": float(np.mean(step2_vals)),
                "median": float(np.median(step2_vals)),
                "std": float(np.std(step2_vals)),
                "p95": float(np.percentile(step2_vals, 95)),
                "max": float(np.max(step2_vals)),
            }
        if step3_error_m is not None:
            step3_vals = np.asarray(step3_error_m, dtype=float)
            stage_stats["step3"] = {
                "rmse": float(np.sqrt(np.mean(step3_vals ** 2))),
                "mean": float(np.mean(step3_vals)),
                "median": float(np.median(step3_vals)),
                "std": float(np.std(step3_vals)),
                "p95": float(np.percentile(step3_vals, 95)),
                "max": float(np.max(step3_vals)),
            }
        if stage_stats:
            _send_statistics_bars(rr, "vicon_ws/error/statistics", stage_stats)

        return {
            "enabled": "true",
            "status": "ok",
            "message": (
                "Rerun logged (not persisted to .rrd, evo-style) "
                f"(timeline={timeline_name}, replay_step={replay_step}, replay_poses={len(t_replay)})"
            ),
        }
    except Exception as exc:
        return {
            "enabled": "true",
            "status": "rerun_runtime_error",
            "message": str(exc),
        }
