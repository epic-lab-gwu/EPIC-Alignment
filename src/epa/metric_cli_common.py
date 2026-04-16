from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.io_utils import load_estimation_trajectory, load_reference_trajectory
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


def align_for_eval(
    pos_ref: np.ndarray,
    quat_ref: np.ndarray,
    pos_est: np.ndarray,
    quat_est: np.ndarray,
    mode: str,
    n_to_align: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(mode).lower()
    if mode == "none":
        return np.asarray(pos_est, dtype=float), np.asarray(quat_est, dtype=float)

    n = pos_ref.shape[0]
    n_use = n if int(n_to_align) <= 0 else min(n, int(n_to_align))
    if n_use < 2:
        return np.asarray(pos_est, dtype=float), np.asarray(quat_est, dtype=float)

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
