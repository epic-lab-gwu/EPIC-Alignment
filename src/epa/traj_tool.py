from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.config_cli import parse_args_with_config
from epa.core.io_utils import (
    SUPPORTED_ROS_MSGS,
    load_estimation_csv,
    load_estimation_trajectory,
    load_reference_trajectory,
)
from epa.core.time_alignment import matching_time_indices
from epa.metric_cli_common import (
    align_for_eval_with_info,
    load_ros_map_spec,
    project_to_plane,
    resolve_map_tile_contextily,
)
from epa.viz.plot_runtime import (
    configure_plot_runtime,
    should_enable_interactive_plot,
    show_plots,
)
from epa.viz.rerun_viz import log_trajectories_to_rerun

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
import matplotlib

from epa.viz.backend_bootstrap import bootstrap_matplotlib_backend

bootstrap_matplotlib_backend(matplotlib)
try:
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401
except Exception:
    _Axes3D = None
import matplotlib.pyplot as plt


@dataclass
class Trajectory:
    label: str
    source: Path
    topic: str
    t: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    synced: bool = False
    ref_label: str = ""
    matched_samples: int = 0


def _poses_from_traj(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    mats = R.from_quat(np.asarray(quat, dtype=float)).as_matrix()
    n = np.asarray(pos, dtype=float).shape[0]
    out = np.repeat(np.eye(4, dtype=float)[None, :, :], n, axis=0)
    out[:, :3, :3] = mats
    out[:, :3, 3] = np.asarray(pos, dtype=float)
    return out


def _closest_rotation(mat3: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(mat3, dtype=float))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _traj_from_poses(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=float)
    pos = poses[:, :3, 3].copy()
    rot = np.zeros((poses.shape[0], 3, 3), dtype=float)
    for i in range(poses.shape[0]):
        rot[i] = _closest_rotation(poses[i, :3, :3])
    quat = R.from_matrix(rot).as_quat()
    return pos, quat


def _parse_spec(spec: str, default_topic: str) -> tuple[Path, str]:
    text = str(spec).strip()
    if "::" in text:
        p, topic = text.split("::", 1)
        return Path(p).expanduser().resolve(), topic.strip()
    return Path(text).expanduser().resolve(), default_topic


def _normalize_evo_bag_topics(specs: list[str], fmt: str, all_topics: bool) -> list[str]:
    fmt_norm = str(fmt).strip().lower()
    if fmt_norm not in {"bag", "bag2", "mcap"}:
        return specs
    if len(specs) < 2:
        return specs
    first = str(specs[0]).strip()
    if not first or "::" in first:
        return specs
    bag_path = Path(first).expanduser().resolve()
    if not bag_path.exists():
        return specs
    rest = [str(x).strip() for x in specs[1:]]
    if any((not x) or ("::" in x) or Path(x).expanduser().exists() for x in rest):
        return specs
    if all_topics:
        return specs
    return [f"{bag_path}::{topic}" for topic in rest]


def _path_length(pos: np.ndarray) -> float:
    if pos.shape[0] <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def _apply_time_window(
    t: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    t_start: float | None,
    t_end: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t_start is None and t_end is None:
        return t, pos, quat
    mask = np.ones(t.shape[0], dtype=bool)
    if t_start is not None:
        mask &= t >= float(t_start)
    if t_end is not None:
        mask &= t <= float(t_end)
    if int(np.sum(mask)) < 2:
        raise ValueError("Time window kept fewer than 2 trajectory samples.")
    t_new = t[mask]
    t_new = t_new - float(t_new[0])
    return t_new, pos[mask], quat[mask]


def _align_origin(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pos.shape[0] == 0:
        return pos, quat
    q0 = quat[0]
    r0 = R.from_quat(q0)
    pos_new = (r0.inv().as_matrix() @ (pos - pos[0]).T).T
    quat_new = (r0.inv() * R.from_quat(quat)).as_quat()
    return pos_new, quat_new


def _load_transform(path: Path) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = ("x", "y", "z", "qx", "qy", "qz", "qw")
        if not isinstance(data, dict) or not all(k in data for k in keys):
            raise ValueError(f"Invalid transform json: {path}")
        xyz = np.array([float(data["x"]), float(data["y"]), float(data["z"])], dtype=float)
        quat = np.array(
            [float(data["qx"]), float(data["qy"]), float(data["qz"]), float(data["qw"])], dtype=float
        )
        scale = float(data.get("scale", 1.0))
        T = np.eye(4, dtype=float)
        T[:3, :3] = scale * R.from_quat(quat).as_matrix()
        T[:3, 3] = xyz
        return T
    if path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path), dtype=float)
    else:
        arr = np.asarray(np.loadtxt(path), dtype=float)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=float)
        out[:3, :4] = arr
        return out
    if arr.shape == (4, 4):
        return arr
    if arr.size == 12:
        arr = arr.reshape(3, 4)
        out = np.eye(4, dtype=float)
        out[:3, :4] = arr
        return out
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"Transform must be 4x4 or 3x4: {path}")


def _invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    Rm = T[:3, :3]
    t = T[:3, 3]
    R_inv = np.linalg.pinv(Rm)
    out = np.eye(4, dtype=float)
    out[:3, :3] = R_inv
    out[:3, 3] = -R_inv @ t
    return out


def _apply_transform(
    pos: np.ndarray,
    quat: np.ndarray,
    T: np.ndarray,
    right_mul: bool = False,
    propagate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    poses = _poses_from_traj(pos, quat)
    if right_mul and not propagate:
        poses_new = np.einsum("nij,jk->nik", poses, T)
    elif right_mul and propagate:
        n = poses.shape[0]
        rel = []
        for i in range(n - 1):
            rel_i = np.linalg.inv(poses[i]) @ poses[i + 1] @ T
            rel.append(rel_i)
        poses_new = [poses[0]]
        for i in range(n - 1):
            poses_new.append(poses_new[-1] @ rel[i])
        poses_new = np.asarray(poses_new, dtype=float)
    else:
        poses_new = np.einsum("ij,njk->nik", T, poses)
    return _traj_from_poses(poses_new)


def _umeyama_transform(src_xyz: np.ndarray, dst_xyz: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src_xyz, dtype=float)
    dst = np.asarray(dst_xyz, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Umeyama alignment requires at least 3 paired 3D points.")
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / float(src.shape[0])
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3, dtype=float)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0
    r_align = u @ s @ vt
    scale = 1.0
    if with_scale:
        var_src = float(np.mean(np.sum(src_c**2, axis=1)))
        if var_src < 1e-12:
            raise ValueError("Degenerate source trajectory for scale alignment.")
        scale = float(np.trace(np.diag(d) @ s) / var_src)
    t_align = mu_dst - scale * (r_align @ mu_src)
    return scale, r_align, t_align


def _align_to_reference(
    trajs: list[Trajectory],
    ref_index: int,
    align_mode: str,
    n_to_align: int,
) -> list[Trajectory]:
    ref = trajs[ref_index]
    out = list(trajs)
    for i, tr in enumerate(trajs):
        if i == ref_index:
            continue
        n = min(ref.pos.shape[0], tr.pos.shape[0])
        if n < 2:
            continue
        pos_new, quat_new, _ = align_for_eval_with_info(
            pos_ref=ref.pos[:n],
            quat_ref=ref.quat[:n],
            pos_est=tr.pos,
            quat_est=tr.quat,
            mode=align_mode,
            n_to_align=int(n_to_align),
        )
        out[i] = Trajectory(
            label=tr.label,
            source=tr.source,
            topic=tr.topic,
            t=tr.t,
            pos=pos_new,
            quat=quat_new,
            synced=tr.synced,
            ref_label=tr.ref_label,
            matched_samples=tr.matched_samples,
        )
    return out


def _downsample_traj(t: np.ndarray, pos: np.ndarray, quat: np.ndarray, num_poses: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_poses < 1:
        raise ValueError("--downsample must be >= 1.")
    n = t.shape[0]
    if n <= num_poses:
        return t, pos, quat
    ids = np.linspace(0, n - 1, num_poses, dtype=int)
    return t[ids], pos[ids], quat[ids]


def _motion_filter_traj(
    t: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    distance_threshold: float,
    angle_threshold_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t.shape[0] <= 1:
        return t, pos, quat
    ang_thr = np.deg2rad(float(angle_threshold_deg))
    ids = [0]
    accum_dist = 0.0
    accum_ang = 0.0
    rot = R.from_quat(quat)
    for i in range(1, t.shape[0]):
        accum_dist += float(np.linalg.norm(pos[i] - pos[i - 1]))
        accum_ang += float((rot[i - 1].inv() * rot[i]).magnitude())
        if accum_dist >= float(distance_threshold) or accum_ang >= ang_thr:
            ids.append(i)
            accum_dist = 0.0
            accum_ang = 0.0
    if ids[-1] != t.shape[0] - 1:
        ids.append(t.shape[0] - 1)
    ids_arr = np.asarray(ids, dtype=int)
    return t[ids_arr], pos[ids_arr], quat[ids_arr]


def _merge_trajs(trajs: list[Trajectory]) -> list[Trajectory]:
    if len(trajs) <= 1:
        return trajs
    all_t = np.concatenate([tr.t for tr in trajs])
    all_pos = np.concatenate([tr.pos for tr in trajs], axis=0)
    all_quat = np.concatenate([tr.quat for tr in trajs], axis=0)
    order = np.argsort(all_t)
    merged = Trajectory(
        label="merged_trajectory",
        source=trajs[0].source,
        topic=trajs[0].topic,
        t=all_t[order],
        pos=all_pos[order],
        quat=all_quat[order],
    )
    return [merged]


def _list_bag_topics(path: Path, fmt: str) -> list[str]:
    try:
        from rosbags.rosbag1 import Reader as Rosbag1Reader
        from rosbags.rosbag2 import Reader as Rosbag2Reader
    except Exception as exc:
        raise ImportError("Bag topic listing requires rosbags. Install with: pip install -e .[ros]") from exc

    bag_kind = str(fmt).lower()
    if bag_kind == "auto":
        bag_kind = "bag2" if path.is_dir() else ("bag" if path.suffix.lower() == ".bag" else "mcap")
    reader = Rosbag1Reader(str(path)) if bag_kind == "bag" else Rosbag2Reader(str(path))
    topics: list[str] = []
    try:
        reader.open()
        for c in reader.connections:
            msgtype = str(getattr(c, "msgtype", ""))
            if str(getattr(c, "topic", "")).strip() == "":
                continue
            if int(getattr(c, "msgcount", 0)) <= 0:
                continue
            if msgtype in SUPPORTED_ROS_MSGS and msgtype != "tf2_msgs/msg/TFMessage":
                topics.append(c.topic)
    finally:
        reader.close()
    return sorted(set(topics))


def _write_bag_trajectory(
    writer,
    t: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    topic_name: str,
) -> None:
    from rosbags.rosbag1 import Writer as Rosbag1Writer
    from rosbags.typesys import Stores, get_typestore

    is_ros1 = isinstance(writer, Rosbag1Writer)
    typestore = get_typestore(Stores.ROS1_NOETIC if is_ros1 else Stores.LATEST)
    Time = typestore.types["builtin_interfaces/msg/Time"]
    Header = typestore.types["std_msgs/msg/Header"]
    Point = typestore.types["geometry_msgs/msg/Point"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    Pose = typestore.types["geometry_msgs/msg/Pose"]
    PoseStamped = typestore.types["geometry_msgs/msg/PoseStamped"]
    msgtype = PoseStamped.__msgtype__  # type: ignore
    conn = writer.add_connection(topic_name, msgtype, typestore=typestore)

    seq = 0
    t_shift = np.asarray(t, dtype=float) - float(t[0])
    for ts, p, q in zip(t_shift, pos, quat):
        sec = int(ts // 1.0)
        nanosec = int((ts - sec) * 1e9)
        time = Time(sec, nanosec)
        if is_ros1:
            header = Header(seq, time, "")
            seq += 1
        else:
            header = Header(time, "")
        point = Point(float(p[0]), float(p[1]), float(p[2]))
        qmsg = Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        msg = PoseStamped(header, Pose(point, qmsg))
        if is_ros1:
            raw = typestore.serialize_ros1(msg, msgtype)
        else:
            raw = typestore.serialize_cdr(msg, msgtype)
        writer.write(conn, int(ts * 1e9), raw)


def _write_bag_export(trajs: list[Trajectory], out_path: Path, bag_kind: str) -> None:
    try:
        from rosbags.rosbag1 import Writer as Rosbag1Writer
        from rosbags.rosbag2 import Writer as Rosbag2Writer
    except Exception as exc:
        raise ImportError("Bag export requires rosbags. Install with: pip install -e .[ros]") from exc

    if bag_kind == "bag":
        writer = Rosbag1Writer(str(out_path))
    else:
        writer = Rosbag2Writer(str(out_path))
    try:
        writer.open()
        for tr in trajs:
            topic = str(tr.topic).strip()
            if not topic or ":" in topic:
                topic = "/" + _safe_name(tr.label)
            if not topic.startswith("/"):
                topic = "/" + topic
            _write_bag_trajectory(writer, tr.t, tr.pos, tr.quat, topic_name=topic)
    finally:
        writer.close()


def _load_traj(path: Path, fmt: str, topic: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmt = str(fmt).lower()
    if fmt in {"bag", "bag2", "mcap", "kitti", "tum", "euroc"}:
        return load_estimation_trajectory(path, fmt, est_topic=topic)
    if fmt == "csv":
        try:
            return load_estimation_trajectory(path, fmt, est_topic=topic)
        except Exception:
            return load_reference_trajectory(path, gt_format=fmt, gt_topic=topic)
    if fmt == "auto":
        suffix = path.suffix.lower()
        if path.is_dir() or suffix in {".bag", ".mcap"}:
            return load_estimation_trajectory(path, fmt, est_topic=topic)
        if suffix == ".csv":
            try:
                return load_reference_trajectory(path, gt_format="csv", gt_topic=topic)
            except Exception:
                return load_estimation_csv(path)
        return load_estimation_trajectory(path, fmt, est_topic=topic)
    raise ValueError(f"Unsupported format: {fmt}")


def _resolve_ref_index(ref: str, trajs: list[Trajectory]) -> int:
    text = str(ref).strip()
    if text == "":
        return 0
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(trajs):
            return idx
        raise ValueError(f"Invalid --ref index: {text} (valid range: 1..{len(trajs)})")
    for i, tr in enumerate(trajs):
        if tr.label == text:
            return i
    labels = ", ".join(tr.label for tr in trajs)
    raise ValueError(f"Invalid --ref label: {text}. Available labels: {labels}")


def _sync_trajectories(
    trajs: list[Trajectory],
    ref_index: int,
    max_diff: float,
    offset: float,
) -> list[Trajectory]:
    if len(trajs) < 2:
        raise ValueError("--sync requires at least two trajectories.")
    ref = trajs[ref_index]
    synced = list(trajs)
    for i, tr in enumerate(trajs):
        if i == ref_index:
            continue
        idx_ref, idx_cur = matching_time_indices(ref.t, tr.t, max_diff=max_diff, offset_2=offset)
        if len(idx_ref) < 2:
            raise ValueError(
                f"Sync failed for {tr.label}: only {len(idx_ref)} matches "
                f"(max_diff={max_diff}, offset={offset})."
            )
        idx_ref_arr = np.asarray(idx_ref, dtype=int)
        idx_cur_arr = np.asarray(idx_cur, dtype=int)
        t_new = ref.t[idx_ref_arr]
        t_new = t_new - float(t_new[0])
        synced[i] = Trajectory(
            label=tr.label,
            source=tr.source,
            topic=tr.topic,
            t=t_new,
            pos=tr.pos[idx_cur_arr],
            quat=tr.quat[idx_cur_arr],
            synced=True,
            ref_label=ref.label,
            matched_samples=int(idx_ref_arr.size),
        )
    return synced


def _safe_name(name: str) -> str:
    text = str(name).strip().replace("/", "_").replace(" ", "_")
    return text if text else "traj"


def _build_labels(paths: list[Path], labels: list[str], show_full_names: bool = False) -> list[str]:
    if labels:
        if len(labels) != len(paths):
            raise ValueError("--labels count must match trajectory count.")
        base = [_safe_name(x) for x in labels]
    elif bool(show_full_names):
        base = [_safe_name(str(p)) for p in paths]
    else:
        base = [_safe_name(p.stem) for p in paths]
    seen: dict[str, int] = {}
    out: list[str] = []
    for b in base:
        cnt = seen.get(b, 0)
        seen[b] = cnt + 1
        out.append(b if cnt == 0 else f"{b}_{cnt+1}")
    return out


def _write_tum(path: Path, t: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ts, p, q in zip(t, pos, quat):
            f.write(
                f"{float(ts):.9f} {float(p[0]):.9f} {float(p[1]):.9f} {float(p[2]):.9f} "
                f"{float(q[0]):.9f} {float(q[1]):.9f} {float(q[2]):.9f} {float(q[3]):.9f}\n"
            )


def _write_kitti(path: Path, pos: np.ndarray, quat: np.ndarray) -> None:
    mats = R.from_quat(quat).as_matrix()
    with path.open("w", encoding="utf-8") as f:
        for p, m in zip(pos, mats):
            row = np.hstack([m, np.asarray(p, dtype=float).reshape(3, 1)]).reshape(-1)
            f.write(" ".join(f"{float(v):.9f}" for v in row) + "\n")


def _full_check_line(tr: Trajectory) -> str:
    if tr.t.size == 0:
        return f"[full_check] {tr.label}: empty trajectory"
    pos = np.asarray(tr.pos, dtype=float)
    quat = np.asarray(tr.quat, dtype=float)
    qnorm = np.linalg.norm(quat, axis=1)
    lo = np.min(pos, axis=0)
    hi = np.max(pos, axis=0)
    return (
        f"[full_check] {tr.label}: "
        f"bbox=([{lo[0]:.3f},{lo[1]:.3f},{lo[2]:.3f}] -> "
        f"[{hi[0]:.3f},{hi[1]:.3f},{hi[2]:.3f}]), "
        f"quat_norm=[min:{np.min(qnorm):.6f}, mean:{np.mean(qnorm):.6f}, max:{np.max(qnorm):.6f}]"
    )


def _plot_trajectories(
    trajs: list[Trajectory],
    mode: str,
    stride: int,
    out_path: Path,
    ros_map_yaml: str = "",
    map_tile: str = "",
) -> matplotlib.figure.Figure:
    stride = max(1, int(stride))
    mode = str(mode).lower()
    if mode == "xyz":
        fig = plt.figure(figsize=(8.5, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        for tr in trajs:
            pts = tr.pos[::stride]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=tr.label, linewidth=1.4)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    else:
        axes = {
            "xy": (0, 1),
            "yx": (1, 0),
            "xz": (0, 2),
            "zx": (2, 0),
            "yz": (1, 2),
            "zy": (2, 1),
        }
        if mode not in axes:
            raise ValueError(f"Unsupported plot mode: {mode}")
        i0, i1 = axes[mode]
        fig = plt.figure(figsize=(8.5, 6.2))
        ax = fig.add_subplot(111)
        if str(ros_map_yaml).strip():
            if mode != "xy":
                raise ValueError("--ros_map_yaml currently requires --plot-mode xy")
            spec = load_ros_map_spec(ros_map_yaml)
            img = plt.imread(str(spec.image_path))
            h, w = img.shape[:2]
            extent = (
                spec.origin_x,
                spec.origin_x + float(w) * spec.resolution,
                spec.origin_y,
                spec.origin_y + float(h) * spec.resolution,
            )
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
                raise ValueError("--map_tile currently requires --plot-mode xy")
            try:
                import contextily as cx  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Map tile overlay requires contextily. Install with: pip install contextily"
                ) from exc
            cx.add_basemap(ax, **resolve_map_tile_contextily(cx, str(map_tile)))
        for tr in trajs:
            pts = tr.pos[::stride]
            ax.plot(pts[:, i0], pts[:, i1], label=tr.label, linewidth=1.4)
        ax.set_xlabel(f"{'xyz'[i0]} (m)")
        ax.set_ylabel(f"{'xyz'[i1]} (m)")
        ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


def _plot_xyz_series(trajs: list[Trajectory], out_path: Path, relative_time: bool) -> matplotlib.figure.Figure:
    fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(10.2, 7.0))
    labels = ("x (m)", "y (m)", "z (m)")
    for tr in trajs:
        tvals = tr.t - float(tr.t[0]) if bool(relative_time) and tr.t.size > 0 else tr.t
        for i in range(3):
            axarr[i].plot(tvals, tr.pos[:, i], linewidth=1.2, label=tr.label if i == 0 else None)
            axarr[i].set_ylabel(labels[i])
            axarr[i].grid(True, linestyle=":", alpha=0.5)
    axarr[-1].set_xlabel("t (s)")
    handles, _ = axarr[0].get_legend_handles_labels()
    if handles:
        axarr[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


def _plot_rpy_series(trajs: list[Trajectory], out_path: Path, relative_time: bool) -> matplotlib.figure.Figure:
    fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(10.2, 7.0))
    labels = ("roll (deg)", "pitch (deg)", "yaw (deg)")
    for tr in trajs:
        tvals = tr.t - float(tr.t[0]) if bool(relative_time) and tr.t.size > 0 else tr.t
        rpy = R.from_quat(np.asarray(tr.quat, dtype=float)).as_euler("xyz", degrees=True)
        for i in range(3):
            axarr[i].plot(tvals, rpy[:, i], linewidth=1.2, label=tr.label if i == 0 else None)
            axarr[i].set_ylabel(labels[i])
            axarr[i].grid(True, linestyle=":", alpha=0.5)
    axarr[-1].set_xlabel("t (s)")
    handles, _ = axarr[0].get_legend_handles_labels()
    if handles:
        axarr[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


def _plot_speed_series(trajs: list[Trajectory], out_path: Path, relative_time: bool) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(10.2, 4.2))
    ax = fig.add_subplot(111)
    for tr in trajs:
        if tr.t.size <= 1:
            continue
        dt = np.diff(tr.t)
        valid = dt > 1e-12
        if not np.any(valid):
            continue
        speed = np.linalg.norm(np.diff(tr.pos, axis=0), axis=1)
        speed = speed[valid] / dt[valid]
        t_mid = 0.5 * (tr.t[1:] + tr.t[:-1])
        t_mid = t_mid[valid]
        if bool(relative_time) and t_mid.size > 0:
            t_mid = t_mid - float(tr.t[0])
        ax.plot(t_mid, speed, linewidth=1.2, label=tr.label)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("speed (m/s)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return fig


def _resolve_plot_targets(save_plot: str, out_dir: Path | None) -> dict[str, list[Path]]:
    if out_dir is None:
        out_dir = _default_out_dir(Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: dict[str, list[Path]] = {
        "trajectories": [],
        "xyz": [],
        "rpy": [],
        "speeds": [],
    }
    if str(save_plot).strip():
        base = Path(save_plot).expanduser().resolve()
        if base.suffix:
            suffix = base.suffix
            stem = base.stem
            targets["trajectories"].append(base.with_name(f"{stem}_trajectories{suffix}"))
            targets["xyz"].append(base.with_name(f"{stem}_xyz{suffix}"))
            targets["rpy"].append(base.with_name(f"{stem}_rpy{suffix}"))
            targets["speeds"].append(base.with_name(f"{stem}_speeds{suffix}"))
        else:
            base.mkdir(parents=True, exist_ok=True)
            targets["trajectories"].append(base / "trajectories.png")
            targets["xyz"].append(base / "xyz.png")
            targets["rpy"].append(base / "rpy.png")
            targets["speeds"].append(base / "speeds.png")
    else:
        targets["trajectories"].append(out_dir / "trajectories.png")
        targets["xyz"].append(out_dir / "xyz.png")
        targets["rpy"].append(out_dir / "rpy.png")
        targets["speeds"].append(out_dir / "speeds.png")

    # Keep backward-compatible default name.
    if not str(save_plot).strip():
        targets["trajectories"].append(out_dir / "traj_plot.png")
    return targets


def _default_out_dir(cwd: Path) -> Path:
    root = cwd / "outputs" / "traj_tool"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = root / stamp
    idx = 1
    while out.exists():
        out = root / f"{stamp}_{idx:02d}"
        idx += 1
    out.mkdir(parents=True, exist_ok=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trajectory utility: load, inspect, plot, and export one or multiple trajectories."
    )
    p.add_argument(
        "-c",
        "--config",
        default="",
        help="Path to JSON config file. If set, config values override CLI flags.",
    )
    p.add_argument(
        "-f",
        "--full-check",
        "--full_check",
        action="store_true",
        help="Run all checks and print additional trajectory stats.",
    )
    p.add_argument(
        "--format",
        choices=["auto", "csv", "euroc", "tum", "kitti", "bag", "bag2", "mcap"],
        default="auto",
        help="Trajectory format for all inputs.",
    )
    p.add_argument(
        "--topic",
        default="",
        help="Default topic for bag inputs. Can be overridden by each spec: path::topic.",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Optional labels, count must match number of trajectories.",
    )
    p.add_argument(
        "--align-origin",
        "--align_origin",
        action="store_true",
        help="Transform each trajectory so first pose is at origin.",
    )
    p.add_argument(
        "--align",
        action="store_true",
        help="Align each non-reference trajectory to the reference using Umeyama (SE3).",
    )
    p.add_argument(
        "--correct-scale",
        action="store_true",
        help="Enable scale correction during --align (or scale-only if --align is not set).",
    )
    p.add_argument(
        "--eval-align",
        choices=[
            "",
            "none",
            "epa_step3",
            "se3",
            "epa_se3",
            "epa_se3_eval",
            "posyaw",
            "epa_posyaw",
            "sim3",
            "ov_sim3",
            "epa_sim3",
            "scale",
        ],
        default="",
        help="Explicit EPA alignment mode; overrides --align/--correct-scale.",
    )
    p.add_argument(
        "--n-to-align",
        "--n_to_align",
        type=int,
        default=-1,
        help="Number of leading poses used for alignment (-1 means all).",
    )
    p.add_argument("--ref", default="", help="Reference label or 1-based index used for --sync.")
    p.add_argument("--sync", action="store_true", help="Associate each non-reference trajectory to the reference by timestamp.")
    p.add_argument(
        "--sync-max-diff",
        "--t_max_diff",
        type=float,
        default=0.01,
        help="Maximum timestamp difference used for --sync association.",
    )
    p.add_argument(
        "--sync-offset",
        "--t_offset",
        type=float,
        default=0.0,
        help="Constant offset added to non-reference timestamps before --sync matching.",
    )
    p.add_argument("--t-start", type=float, default=None, help="Keep samples with t >= t_start.")
    p.add_argument("--t-end", type=float, default=None, help="Keep samples with t <= t_end.")
    p.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="Downsample each trajectory to max N poses (0 disables).",
    )
    p.add_argument(
        "--motion-filter",
        "--motion_filter",
        nargs=2,
        type=float,
        metavar=("DISTANCE", "ANGLE_DEGREES"),
        default=None,
        help="Filter poses by accumulated motion thresholds.",
    )
    p.add_argument("--merge", action="store_true", help="Merge trajectories into one timestamp-sorted trajectory.")
    p.add_argument(
        "--project-to-plane",
        "--project_to_plane",
        choices=["none", "xy", "xz", "yz"],
        default="none",
        help="Projects trajectories to the selected 2D plane after alignment/transforms.",
    )
    p.add_argument(
        "--transform-left",
        "--transform_left",
        default="",
        help="Path to transformation matrix/json applied left-multiplicative.",
    )
    p.add_argument(
        "--transform-right",
        "--transform_right",
        default="",
        help="Path to transformation matrix/json applied right-multiplicative.",
    )
    p.add_argument(
        "--propagate-transform",
        "--propagate_transform",
        action="store_true",
        help="With --transform-right, propagate resulting drift to subsequent poses.",
    )
    p.add_argument(
        "--invert-transform",
        "--invert_transform",
        action="store_true",
        help="Invert transform loaded from --transform-left/--transform-right.",
    )
    p.add_argument(
        "--all-topics",
        "--all_topics",
        "--all_channels",
        action="store_true",
        help="For bag inputs, load all supported topics from each bag path.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate trajectory plots (in TTY sessions also opens interactive window).",
    )
    p.add_argument(
        "--plot-interactive",
        "--plot_interactive",
        action="store_true",
        help="Show interactive matplotlib window after generating plots.",
    )
    p.add_argument(
        "--plot-backend",
        "--plot_backend",
        default="",
        help="Optional matplotlib backend override (e.g. qtagg, tkagg).",
    )
    p.add_argument(
        "--plot-relative-time",
        "--plot_relative_time",
        action="store_true",
        help="Use relative timestamps for time-series plots.",
    )
    p.add_argument(
        "--plot-mode",
        "--plot_mode",
        choices=["xy", "xz", "yx", "yz", "zx", "zy", "xyz"],
        default="xyz",
        help="Plot projection mode.",
    )
    p.add_argument("--plot-stride", type=int, default=1, help="Downsample stride for plotting.")
    p.add_argument(
        "--ros_map_yaml",
        default="",
        help="ROS occupancy map yaml to render under XY trajectory plots.",
    )
    p.add_argument(
        "--map_tile",
        default="",
        help="Map tile source (provider name/url) or EPSG:xxxx CRS for contextily basemap.",
    )
    p.add_argument(
        "--save-plot",
        "--save_plot",
        default="",
        help="Path prefix or directory to save generated plots.",
    )
    p.add_argument("--save-table", "--save_table", default="", help="Path to save summary table as CSV.")
    p.add_argument(
        "--save-as",
        "--save_as",
        choices=["", "tum", "kitti", "bag", "bag2"],
        default="",
        help="Export each loaded trajectory to this format.",
    )
    p.add_argument("--save_as_tum", action="store_true", help="Save trajectories as TUM files.")
    p.add_argument("--save_as_kitti", action="store_true", help="Save trajectories as KITTI files.")
    p.add_argument("--save_as_bag", action="store_true", help="Save trajectories in ROS1 bag.")
    p.add_argument("--save_as_bag2", action="store_true", help="Save trajectories in ROS2 bag.")
    p.add_argument("--out-dir", "--out_dir", default="", help="Output dir for auto-generated files.")
    p.add_argument("--show_full_names", action="store_true", help="Use full source paths as labels.")
    p.add_argument("--rerun", action="store_true", help="Log visualization data to rerun.")
    p.add_argument(
        "--rerun_rec_id",
        "--rerun-rec-id",
        default=None,
        help="Use a specific recording ID for rerun.",
    )
    p.add_argument("--logfile", default=None, help="Reserved for compatibility.")
    p.add_argument("--no_warnings", action="store_true", help="Reserved for compatibility.")
    p.add_argument("-v", "--verbose", action="store_true", help="Reserved for compatibility.")
    p.add_argument("--silent", action="store_true", help="Reserved for compatibility.")
    p.add_argument("--debug", action="store_true", help="Reserved for compatibility.")
    p.add_argument("trajectories", nargs="+", help="Trajectory specs: /path/file or /path/file::topic")
    return p


def run(args: argparse.Namespace) -> int:
    # Support format-first subcommand invocation:
    #   epa_traj tum a.tum b.tum ...
    # while keeping current --format based usage.
    if bool(getattr(args, "trajectories", None)) and str(getattr(args, "format", "auto")) == "auto":
        first = str(args.trajectories[0]).strip()
        fmts = {"csv", "euroc", "tum", "kitti", "bag", "bag2", "mcap"}
        if (
            first in fmts
            and "::" not in first
            and not Path(first).expanduser().exists()
            and len(args.trajectories) > 1
        ):
            args.format = first
            args.trajectories = args.trajectories[1:]

    args.trajectories = _normalize_evo_bag_topics(
        [str(x) for x in args.trajectories],
        fmt=str(getattr(args, "format", "auto")),
        all_topics=bool(getattr(args, "all_topics", False)),
    )

    if bool(args.transform_left) and bool(args.transform_right):
        raise ValueError("--transform-left and --transform-right are mutually exclusive.")
    if bool(args.propagate_transform) and not bool(args.transform_right):
        raise ValueError("--propagate-transform requires --transform-right.")
    if bool(args.align_origin) and bool(args.align):
        raise ValueError("--align-origin and --align cannot be used together.")

    specs = [str(x) for x in args.trajectories]
    parsed = [_parse_spec(s, default_topic=str(args.topic)) for s in specs]
    if bool(args.all_topics):
        if args.labels:
            raise ValueError("--all-topics cannot be combined with --labels.")
        expanded: list[tuple[Path, str]] = []
        seen: set[tuple[str, str]] = set()
        for path, topic in parsed:
            if topic.strip():
                key = (str(path), topic.strip())
                if key not in seen:
                    seen.add(key)
                    expanded.append((path, topic.strip()))
            for tp in _list_bag_topics(path, fmt=str(args.format)):
                key = (str(path), tp)
                if key not in seen:
                    seen.add(key)
                    expanded.append((path, tp))
        parsed = expanded
        if not parsed:
            raise ValueError("--all-topics found no supported trajectory topics.")

    paths = [p for p, _ in parsed]
    labels = _build_labels(paths, list(args.labels), show_full_names=bool(getattr(args, "show_full_names", False)))

    trajs: list[Trajectory] = []
    for idx, ((path, topic), label) in enumerate(zip(parsed, labels), start=1):
        if not path.exists():
            raise FileNotFoundError(f"Trajectory not found: {path}")
        t, pos, quat = _load_traj(path, fmt=args.format, topic=topic)
        t, pos, quat = _apply_time_window(t, pos, quat, t_start=args.t_start, t_end=args.t_end)
        if int(getattr(args, "downsample", 0) or 0) > 0:
            t, pos, quat = _downsample_traj(t, pos, quat, int(args.downsample))
        if args.motion_filter is not None:
            t, pos, quat = _motion_filter_traj(
                t, pos, quat, distance_threshold=float(args.motion_filter[0]), angle_threshold_deg=float(args.motion_filter[1])
            )
        if bool(args.align_origin):
            pos, quat = _align_origin(pos, quat)
        trajs.append(Trajectory(label=label, source=path, topic=topic, t=t, pos=pos, quat=quat))
        duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
        print(
            f"[{idx}] {label}: samples={t.size}, duration_s={duration:.3f}, "
            f"path_m={_path_length(pos):.3f}, src={path}"
        )
        if bool(getattr(args, "full_check", False)):
            print(_full_check_line(trajs[-1]))

    if bool(args.merge):
        trajs = _merge_trajs(trajs)
        print("Merged trajectories into: merged_trajectory")

    explicit_align = str(getattr(args, "eval_align", "") or "").strip().lower()
    if explicit_align == "epa_sim3":
        explicit_align = "sim3"
    align_mode = explicit_align
    if not align_mode and bool(args.align):
        align_mode = "sim3" if bool(args.correct_scale) else "se3"
    elif not align_mode and bool(args.correct_scale):
        align_mode = "scale"
    needs_sync = bool(args.sync or align_mode)
    if needs_sync:
        ref_index = _resolve_ref_index(str(args.ref), trajs)
        trajs = _sync_trajectories(
            trajs,
            ref_index=ref_index,
            max_diff=float(args.sync_max_diff),
            offset=float(args.sync_offset),
        )
        ref_label = trajs[ref_index].label
        print(
            f"Sync reference: {ref_label} (max_diff={float(args.sync_max_diff):.3f}, "
            f"offset={float(args.sync_offset):.3f})"
        )
        for tr in trajs:
            if tr.synced:
                print(f"Synced {tr.label} <- {tr.ref_label}: matches={tr.matched_samples}")

    if align_mode:
        ref_index = _resolve_ref_index(str(args.ref), trajs)
        trajs = _align_to_reference(
            trajs,
            ref_index=ref_index,
            align_mode=align_mode,
            n_to_align=int(args.n_to_align),
        )
        print(
            f"Aligned trajectories to reference: {trajs[ref_index].label} "
            f"(mode={align_mode}, n_to_align={int(args.n_to_align)})"
        )

    if bool(args.transform_left or args.transform_right):
        tf_path = args.transform_left or args.transform_right
        transform = _load_transform(Path(tf_path))
        if bool(args.invert_transform):
            transform = _invert_transform(transform)
        right_mul = bool(args.transform_right)
        for i, tr in enumerate(trajs):
            pos_new, quat_new = _apply_transform(
                tr.pos,
                tr.quat,
                transform,
                right_mul=right_mul,
                propagate=bool(args.propagate_transform),
            )
            trajs[i] = Trajectory(
                label=tr.label,
                source=tr.source,
                topic=tr.topic,
                t=tr.t,
                pos=pos_new,
                quat=quat_new,
                synced=tr.synced,
                ref_label=tr.ref_label,
                matched_samples=tr.matched_samples,
            )
        print(
            f"Applied {'right' if right_mul else 'left'} transform: {Path(tf_path).expanduser().resolve()}"
        )

    proj = str(getattr(args, "project_to_plane", "none") or "none").lower()
    if proj != "none":
        for i, tr in enumerate(trajs):
            pos_new, quat_new = project_to_plane(tr.pos, tr.quat, proj)
            trajs[i] = Trajectory(
                label=tr.label,
                source=tr.source,
                topic=tr.topic,
                t=tr.t,
                pos=pos_new,
                quat=quat_new,
                synced=tr.synced,
                ref_label=tr.ref_label,
                matched_samples=tr.matched_samples,
            )
        print(f"Projected trajectories to plane: {proj}")

    save_as_flags = {
        "tum": bool(getattr(args, "save_as_tum", False)),
        "kitti": bool(getattr(args, "save_as_kitti", False)),
        "bag": bool(getattr(args, "save_as_bag", False)),
        "bag2": bool(getattr(args, "save_as_bag2", False)),
    }
    enabled_save_as = [k for k, v in save_as_flags.items() if v]
    if str(args.save_as).strip() and enabled_save_as:
        raise ValueError("Use either --save-as or one of --save_as_* flags, not both.")
    if len(enabled_save_as) > 1:
        raise ValueError("Only one of --save_as_tum/--save_as_kitti/--save_as_bag/--save_as_bag2 may be set.")
    save_kind = str(args.save_as).strip() if str(args.save_as).strip() else (enabled_save_as[0] if enabled_save_as else "")

    need_out_dir = bool(args.plot or save_kind or args.save_table) and not (
        args.save_plot or args.save_table or args.out_dir
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if need_out_dir and out_dir is None:
        out_dir = _default_out_dir(Path.cwd())
    elif out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot):
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
        open_figures: list[matplotlib.figure.Figure] = []
        targets = _resolve_plot_targets(str(args.save_plot), out_dir)
        primary_traj_plot = targets["trajectories"][0]
        primary_traj_plot.parent.mkdir(parents=True, exist_ok=True)
        fig_traj = _plot_trajectories(
            trajs,
            mode=args.plot_mode,
            stride=args.plot_stride,
            out_path=primary_traj_plot,
            ros_map_yaml=str(getattr(args, "ros_map_yaml", "") or ""),
            map_tile=str(getattr(args, "map_tile", "") or ""),
        )
        open_figures.append(fig_traj)
        for pth in targets["trajectories"][1:]:
            pth.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(primary_traj_plot, pth)
        for idx, pth in enumerate(targets["xyz"]):
            pth.parent.mkdir(parents=True, exist_ok=True)
            fig = _plot_xyz_series(trajs, pth, relative_time=bool(getattr(args, "plot_relative_time", False)))
            if idx == 0:
                open_figures.append(fig)
            else:
                plt.close(fig)
        for idx, pth in enumerate(targets["rpy"]):
            pth.parent.mkdir(parents=True, exist_ok=True)
            fig = _plot_rpy_series(trajs, pth, relative_time=bool(getattr(args, "plot_relative_time", False)))
            if idx == 0:
                open_figures.append(fig)
            else:
                plt.close(fig)
        for idx, pth in enumerate(targets["speeds"]):
            pth.parent.mkdir(parents=True, exist_ok=True)
            fig = _plot_speed_series(trajs, pth, relative_time=bool(getattr(args, "plot_relative_time", False)))
            if idx == 0:
                open_figures.append(fig)
            else:
                plt.close(fig)
        all_plots = targets["trajectories"] + targets["xyz"] + targets["rpy"] + targets["speeds"]
        print(f"Saved plots: {', '.join(str(p) for p in all_plots)}")
        if interactive_enabled and open_figures:
            show_plots()
        for fig in open_figures:
            plt.close(fig)

    if str(save_kind):
        export_dir = out_dir if out_dir is not None else _default_out_dir(Path.cwd())
        export_dir.mkdir(parents=True, exist_ok=True)
        for tr in trajs:
            if save_kind == "tum":
                out = export_dir / f"{tr.label}.tum"
                _write_tum(out, tr.t, tr.pos, tr.quat)
                print(f"Saved export: {out}")
            elif save_kind == "kitti":
                out = export_dir / f"{tr.label}.kitti"
                _write_kitti(out, tr.pos, tr.quat)
                print(f"Saved export: {out}")
        if save_kind in {"bag", "bag2"}:
            bag_out = export_dir / ("trajectories.bag" if save_kind == "bag" else "trajectories_bag2")
            if bag_out.exists():
                raise FileExistsError(f"Export destination already exists: {bag_out}")
            _write_bag_export(trajs, bag_out, bag_kind=str(save_kind))
            print(f"Saved export: {bag_out}")

    table_path = Path(args.save_table).expanduser().resolve() if args.save_table else (
        (out_dir / "traj_summary.csv") if out_dir is not None else None
    )
    if table_path is not None:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with table_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "label",
                    "source",
                    "topic",
                    "samples",
                    "duration_s",
                    "path_length_m",
                    "synced",
                    "ref_label",
                    "matched_samples",
                    "start_x",
                    "start_y",
                    "start_z",
                    "end_x",
                    "end_y",
                    "end_z",
                ],
            )
            writer.writeheader()
            for tr in trajs:
                t = tr.t
                pos = tr.pos
                writer.writerow(
                    {
                        "label": tr.label,
                        "source": str(tr.source),
                        "topic": tr.topic,
                        "samples": int(t.size),
                        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
                        "path_length_m": _path_length(pos),
                        "synced": str(bool(tr.synced)).lower(),
                        "ref_label": tr.ref_label,
                        "matched_samples": int(tr.matched_samples),
                        "start_x": float(pos[0, 0]),
                        "start_y": float(pos[0, 1]),
                        "start_z": float(pos[0, 2]),
                        "end_x": float(pos[-1, 0]),
                        "end_y": float(pos[-1, 1]),
                        "end_z": float(pos[-1, 2]),
                    }
                )
        print(f"Saved table: {table_path}")

    if bool(getattr(args, "rerun", False)):
        rerun_info = log_trajectories_to_rerun(
            trajectories=[
                {
                    "name": tr.label,
                    "positions": tr.pos,
                    "quaternions": tr.quat,
                    "timestamps": tr.t,
                }
                for tr in trajs
            ],
            app_id="epa_traj",
            spawn=True,
            recording_id=(
                None
                if getattr(args, "rerun_rec_id", None) in {None, ""}
                else str(getattr(args, "rerun_rec_id"))
            ),
        )
        print(f"Rerun status: {rerun_info['status']} - {rerun_info['message']}")

    return 0


def main() -> int:
    parser = build_parser()
    args = parse_args_with_config(parser, config_dest="config", tool_name="epa_traj")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
