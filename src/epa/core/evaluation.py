from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .math_utils import compute_error_statistics, poses_se3_from_traj, relative_se3

APE_RELATION_ALIASES = {
    "full": "full_transformation",
    "trans_part": "translation_part",
    "rot_part": "rotation_part",
    "angle_rad": "rotation_angle_rad",
    "angle_deg": "rotation_angle_deg",
    "point_distance": "point_distance",
}

RPE_RELATION_ALIASES = {
    **APE_RELATION_ALIASES,
    "point_distance_error_ratio": "point_distance_error_ratio",
}

RELATION_UNITS = {
    "full_transformation": "",
    "translation_part": "m",
    "rotation_part": "",
    "rotation_angle_rad": "rad",
    "rotation_angle_deg": "deg",
    "point_distance": "m",
    "point_distance_error_ratio": "%",
}


def summarize_abs_errors(errors):
    stats = compute_error_statistics(errors)
    values = np.asarray(errors, dtype=float).reshape(-1)
    stats["p95"] = np.percentile(values, 95) if values.size > 0 else np.nan
    return stats


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


def rpe_pairs_by_time(timestamps, delta, tol=0.0, all_pairs=False):
    stamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if stamps.size < 2:
        return []
    if float(delta) <= 0.0:
        raise ValueError("RPE delta must be > 0 for seconds unit.")
    if np.any(np.diff(stamps) <= 0.0):
        raise ValueError("RPE timestamps must be strictly increasing for seconds unit.")

    tol = float(tol)

    def nearest_after(start_idx: int, target: float) -> int | None:
        k = int(np.searchsorted(stamps, target, side="left"))
        candidates = [idx for idx in (k - 1, k, k + 1) if start_idx < idx < stamps.size]
        if not candidates:
            return None
        best = min(candidates, key=lambda idx: abs(float(stamps[idx]) - float(target)))
        if abs(float(stamps[best]) - float(target)) > tol:
            return None
        return int(best)

    if all_pairs:
        id_pairs = []
        for i in range(stamps.size - 1):
            j = nearest_after(i, float(stamps[i]) + float(delta))
            if j is not None:
                id_pairs.append((int(i), int(j)))
        return id_pairs

    id_pairs = []
    i = 0
    while i < stamps.size - 1:
        j = nearest_after(i, float(stamps[i]) + float(delta))
        if j is None:
            i += 1
            continue
        id_pairs.append((int(i), int(j)))
        i = int(j)
    return id_pairs


def build_rpe_pairs(poses, delta, delta_unit="f", rel_delta_tol=0.1, all_pairs=False, timestamps=None):
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
    if delta_unit == "s":
        if timestamps is None:
            raise ValueError("RPE delta unit seconds ('s') requires timestamps.")
        if len(timestamps) != len(poses):
            raise ValueError("RPE timestamps must have the same length as poses for seconds unit.")
        tol = float(delta) * float(rel_delta_tol)
        return rpe_pairs_by_time(timestamps, float(delta), tol=tol, all_pairs=all_pairs)
    raise ValueError(f"Unsupported RPE delta unit: {delta_unit}")


def normalize_pose_relation(metric_kind, relation):
    relation = str(relation)
    if relation == "all":
        return "all"
    aliases = APE_RELATION_ALIASES if metric_kind == "ape" else RPE_RELATION_ALIASES
    if relation in aliases:
        return aliases[relation]
    if relation in aliases.values():
        return relation
    supported = ", ".join(sorted(list(aliases.keys()) + list(aliases.values())))
    raise ValueError(f"Unsupported {metric_kind.upper()} pose relation: {relation}. Supported: {supported}")


def compute_ape(pos_ref, quat_ref, pos_est, quat_est, include_raw=False):
    if len(pos_ref) != len(pos_est):
        raise ValueError("APE requires trajectories with the same number of poses.")

    pos_ref = np.asarray(pos_ref, dtype=float)
    pos_est = np.asarray(pos_est, dtype=float)
    T_ref = poses_se3_from_traj(pos_ref, quat_ref)
    T_est = poses_se3_from_traj(pos_est, quat_est)
    R_est = T_est[:, :3, :3]
    R_ref = T_ref[:, :3, :3]
    t_est = T_est[:, :3, 3]
    t_ref = T_ref[:, :3, 3]
    E_rot = np.einsum("nij,njk->nik", np.swapaxes(R_est, 1, 2), R_ref)
    E_trans = np.einsum("nij,nj->ni", np.swapaxes(R_est, 1, 2), t_ref - t_est)

    I3 = np.eye(3)
    trans_err = np.linalg.norm(pos_est - pos_ref, axis=1)
    rot_part_err = np.linalg.norm(E_rot - I3, axis=(1, 2))
    full_err = np.sqrt(rot_part_err * rot_part_err + np.sum(E_trans * E_trans, axis=1))
    rot_angle_rad = np.abs(R.from_matrix(E_rot).magnitude())
    rot_angle_deg = np.degrees(rot_angle_rad)
    error_arrays = {
        "translation_part": trans_err,
        "point_distance": trans_err,
        "rotation_part": rot_part_err,
        "full_transformation": full_err,
        "rotation_angle_rad": rot_angle_rad,
        "rotation_angle_deg": rot_angle_deg,
    }

    result = {
        "translation_part": compute_error_statistics(trans_err),
        "point_distance": compute_error_statistics(trans_err),
        "rotation_part": compute_error_statistics(rot_part_err),
        "full_transformation": compute_error_statistics(full_err),
        "rotation_angle_rad": compute_error_statistics(rot_angle_rad),
        "rotation_angle_deg": compute_error_statistics(rot_angle_deg),
    }
    if include_raw:
        result["_error_arrays"] = {k: np.asarray(v, dtype=float) for k, v in error_arrays.items()}
        result["_x_axis"] = {"index": np.arange(trans_err.size, dtype=float)}
    return result


def compute_rpe(
    pos_ref,
    quat_ref,
    pos_est,
    quat_est,
    delta=1.0,
    delta_unit="f",
    rel_delta_tol=0.1,
    all_pairs=False,
    pairs_from_reference=False,
    timestamps=None,
    include_raw=False,
    max_pairs=50000,
):
    if len(pos_ref) != len(pos_est):
        raise ValueError("RPE requires trajectories with the same number of poses.")

    pos_ref = np.asarray(pos_ref, dtype=float)
    pos_est = np.asarray(pos_est, dtype=float)
    T_ref = poses_se3_from_traj(pos_ref, quat_ref)
    T_est = poses_se3_from_traj(pos_est, quat_est)
    pair_source = T_ref if pairs_from_reference else T_est
    id_pairs = build_rpe_pairs(
        pair_source,
        delta=delta,
        delta_unit=delta_unit,
        rel_delta_tol=rel_delta_tol,
        all_pairs=all_pairs,
        timestamps=timestamps,
    )
    if int(max_pairs) > 0 and len(id_pairs) > int(max_pairs):
        keep = np.linspace(0, len(id_pairs) - 1, int(max_pairs), dtype=int)
        id_pairs = [id_pairs[int(idx)] for idx in keep]
    delta_ids = [int(j) for _, j in id_pairs]

    if len(id_pairs) == 0:
        empty = compute_error_statistics(np.array([]))
        result = {
            "pair_count": 0,
            "translation_part": empty,
            "point_distance": empty,
            "point_distance_error_ratio": empty,
            "rotation_part": empty,
            "full_transformation": empty,
            "rotation_angle_rad": empty,
            "rotation_angle_deg": empty,
        }
        if include_raw:
            result["_error_arrays"] = {
                "translation_part": np.array([], dtype=float),
                "point_distance": np.array([], dtype=float),
                "point_distance_error_ratio": np.array([], dtype=float),
                "rotation_part": np.array([], dtype=float),
                "full_transformation": np.array([], dtype=float),
                "rotation_angle_rad": np.array([], dtype=float),
                "rotation_angle_deg": np.array([], dtype=float),
            }
            result["_x_axis"] = {
                "index": np.array([], dtype=float),
                "delta_ids": np.array([], dtype=float),
            }
            result["_pair_ids"] = np.array([], dtype=int).reshape(0, 2)
        return result

    pair_ids = np.asarray(id_pairs, dtype=int).reshape(-1, 2)
    pair_i = pair_ids[:, 0]
    pair_j = pair_ids[:, 1]

    ref_distances = np.linalg.norm(pos_ref[pair_i] - pos_ref[pair_j], axis=1)
    est_distances = np.linalg.norm(pos_est[pair_i] - pos_est[pair_j], axis=1)
    point_distance_err = np.abs(ref_distances - est_distances)
    ratio_mask = ref_distances != 0.0
    point_ratio = np.full(len(id_pairs), np.nan, dtype=float)
    point_ratio[ratio_mask] = np.divide(point_distance_err[ratio_mask], ref_distances[ratio_mask]) * 100.0

    R_ref_i = T_ref[pair_i, :3, :3]
    R_ref_j = T_ref[pair_j, :3, :3]
    t_ref_i = T_ref[pair_i, :3, 3]
    t_ref_j = T_ref[pair_j, :3, 3]
    R_est_i = T_est[pair_i, :3, :3]
    R_est_j = T_est[pair_j, :3, :3]
    t_est_i = T_est[pair_i, :3, 3]
    t_est_j = T_est[pair_j, :3, 3]

    R_ref_rel = np.einsum("nij,njk->nik", np.swapaxes(R_ref_i, 1, 2), R_ref_j)
    t_ref_rel = np.einsum("nij,nj->ni", np.swapaxes(R_ref_i, 1, 2), t_ref_j - t_ref_i)
    R_est_rel = np.einsum("nij,njk->nik", np.swapaxes(R_est_i, 1, 2), R_est_j)
    t_est_rel = np.einsum("nij,nj->ni", np.swapaxes(R_est_i, 1, 2), t_est_j - t_est_i)

    E_rot = np.einsum("nij,njk->nik", np.swapaxes(R_ref_rel, 1, 2), R_est_rel)
    E_trans = np.einsum(
        "nij,nj->ni",
        np.swapaxes(R_ref_rel, 1, 2),
        t_est_rel - t_ref_rel,
    )

    I3 = np.eye(3)
    translation_part_err = np.linalg.norm(E_trans, axis=1)
    rotation_part_err = np.linalg.norm(E_rot - I3, axis=(1, 2))
    full_err = np.sqrt(rotation_part_err * rotation_part_err + translation_part_err * translation_part_err)
    rot_angle_rad = np.abs(R.from_matrix(E_rot).magnitude())
    rot_angle_deg = np.degrees(rot_angle_rad)
    error_arrays = {
        "translation_part": translation_part_err,
        "point_distance": point_distance_err,
        "point_distance_error_ratio": point_ratio,
        "rotation_part": rotation_part_err,
        "full_transformation": full_err,
        "rotation_angle_rad": rot_angle_rad,
        "rotation_angle_deg": rot_angle_deg,
    }

    result = {
        "pair_count": int(len(id_pairs)),
        "translation_part": compute_error_statistics(translation_part_err),
        "point_distance": compute_error_statistics(point_distance_err),
        "point_distance_error_ratio": compute_error_statistics(point_ratio),
        "rotation_part": compute_error_statistics(rotation_part_err),
        "full_transformation": compute_error_statistics(full_err),
        "rotation_angle_rad": compute_error_statistics(rot_angle_rad),
        "rotation_angle_deg": compute_error_statistics(rot_angle_deg),
    }
    if include_raw:
        result["_error_arrays"] = {k: np.asarray(v, dtype=float) for k, v in error_arrays.items()}
        result["_x_axis"] = {
            "index": np.arange(len(id_pairs), dtype=float),
            "delta_ids": np.asarray(delta_ids, dtype=float),
        }
        result["_pair_ids"] = pair_ids
    return result

def _empty_metric_block(include_raw=False):
    empty = compute_error_statistics(np.array([]))
    result = {
        "translation_part": empty,
        "point_distance": empty,
        "rotation_part": empty,
        "full_transformation": empty,
        "rotation_angle_rad": empty,
        "rotation_angle_deg": empty,
    }
    if include_raw:
        result["_error_arrays"] = {
            "translation_part": np.array([], dtype=float),
            "point_distance": np.array([], dtype=float),
            "rotation_part": np.array([], dtype=float),
            "full_transformation": np.array([], dtype=float),
            "rotation_angle_rad": np.array([], dtype=float),
            "rotation_angle_deg": np.array([], dtype=float),
        }
        result["_x_axis"] = {"index": np.array([], dtype=float)}
    return result


def _empty_rpe_block(include_raw=True):
    empty = compute_error_statistics(np.array([]))
    result = {
        "pair_count": 0,
        "translation_part": empty,
        "point_distance": empty,
        "point_distance_error_ratio": empty,
        "rotation_part": empty,
        "full_transformation": empty,
        "rotation_angle_rad": empty,
        "rotation_angle_deg": empty,
    }
    if include_raw:
        result["_error_arrays"] = {
            "translation_part": np.array([], dtype=float),
            "point_distance": np.array([], dtype=float),
            "point_distance_error_ratio": np.array([], dtype=float),
            "rotation_part": np.array([], dtype=float),
            "full_transformation": np.array([], dtype=float),
            "rotation_angle_rad": np.array([], dtype=float),
            "rotation_angle_deg": np.array([], dtype=float),
        }
        result["_x_axis"] = {"index": np.array([], dtype=float), "delta_ids": np.array([], dtype=float)}
        result["_pair_ids"] = np.array([], dtype=int).reshape(0, 2)
    return result


def _filter_ape_block(ape_block, valid_sample_mask, include_raw=True):
    valid = np.asarray(valid_sample_mask, dtype=bool).reshape(-1)
    arrays = ape_block.get("_error_arrays", {}) if isinstance(ape_block, dict) else {}
    if not arrays or valid.size == 0:
        return _empty_metric_block(include_raw=include_raw)

    result = {}
    filtered_arrays = {}
    for relation in (
        "translation_part",
        "point_distance",
        "rotation_part",
        "full_transformation",
        "rotation_angle_rad",
        "rotation_angle_deg",
    ):
        values = np.asarray(arrays.get(relation, []), dtype=float).reshape(-1)
        n = min(values.size, valid.size)
        filtered = values[:n][valid[:n]]
        result[relation] = compute_error_statistics(filtered)
        filtered_arrays[relation] = filtered
    if include_raw:
        result["_error_arrays"] = filtered_arrays
        result["_x_axis"] = {"index": np.arange(filtered_arrays["translation_part"].size, dtype=float)}
    return result


def filter_ape_block_by_valid_samples(ape_block, valid_sample_mask, include_raw=True):
    return _filter_ape_block(ape_block, valid_sample_mask, include_raw=include_raw)


def _filter_rpe_block(rpe_block, valid_segment_mask, include_raw=True):
    valid_segments = np.asarray(valid_segment_mask, dtype=bool).reshape(-1)
    pair_ids = np.asarray(rpe_block.get("_pair_ids", []), dtype=int).reshape(-1, 2)
    arrays = rpe_block.get("_error_arrays", {}) if isinstance(rpe_block, dict) else {}
    if pair_ids.size == 0 or not arrays:
        return _empty_rpe_block(include_raw=include_raw)

    starts = pair_ids[:, 0].astype(int)
    ends = pair_ids[:, 1].astype(int)
    valid_pair = (ends > starts) & (starts >= 0) & ((ends - 1) < valid_segments.size)
    invalid_prefix = np.concatenate(
        [[0], np.cumsum((~valid_segments).astype(int), dtype=int)]
    )
    keep = np.zeros(pair_ids.shape[0], dtype=bool)
    if np.any(valid_pair):
        keep[valid_pair] = (
            invalid_prefix[ends[valid_pair]] - invalid_prefix[starts[valid_pair]]
        ) == 0

    result = {"pair_count": int(np.count_nonzero(keep))}
    filtered_arrays = {}
    for relation in (
        "translation_part",
        "point_distance",
        "point_distance_error_ratio",
        "rotation_part",
        "full_transformation",
        "rotation_angle_rad",
        "rotation_angle_deg",
    ):
        values = np.asarray(arrays.get(relation, []), dtype=float).reshape(-1)
        n = min(values.size, keep.size)
        filtered = values[:n][keep[:n]]
        result[relation] = compute_error_statistics(filtered)
        filtered_arrays[relation] = filtered

    if include_raw:
        kept_pairs = pair_ids[keep]
        result["_error_arrays"] = filtered_arrays
        result["_x_axis"] = {
            "index": np.arange(int(np.count_nonzero(keep)), dtype=float),
            "delta_ids": kept_pairs[:, 1].astype(float) if kept_pairs.size else np.array([], dtype=float),
        }
        result["_pair_ids"] = kept_pairs.astype(int).reshape(-1, 2)
    return result


def filter_rpe_block_by_valid_segments(rpe_block, valid_segment_mask, include_raw=True):
    return _filter_rpe_block(rpe_block, valid_segment_mask, include_raw=include_raw)


def estimate_knee_threshold(
    errors,
    *,
    min_threshold_m=1.0,
    max_threshold_m=100.0,
    fallback_percentile=95.0,
    trim_percentile=95.0,
):
    values = np.asarray(errors, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if values.size >= 3 and float(trim_percentile) < 100.0:
        cap = float(np.percentile(values, float(trim_percentile)))
        trimmed = values[values <= cap]
        if trimmed.size >= 3:
            values = trimmed

    sorted_values = np.sort(values)
    if sorted_values.size < 3 or float(sorted_values[-1] - sorted_values[0]) <= 1e-12:
        threshold = float(np.percentile(sorted_values, float(fallback_percentile)))
    else:
        x = np.linspace(0.0, 1.0, sorted_values.size)
        y = (sorted_values - sorted_values[0]) / (sorted_values[-1] - sorted_values[0])
        line = x
        knee_idx = int(np.argmax(line - y))
        threshold_idx = min(knee_idx + 1, sorted_values.size - 1)
        threshold = float(sorted_values[threshold_idx])
        if knee_idx <= 0 or knee_idx >= sorted_values.size - 1:
            threshold = float(np.percentile(sorted_values, float(fallback_percentile)))

    return float(np.clip(threshold, float(min_threshold_m), float(max_threshold_m)))


def resolve_success_threshold(
    errors,
    *,
    mode="fixed",
    fixed_threshold_m=10.0,
    min_threshold_m=1.0,
    max_threshold_m=100.0,
    trim_percentile=95.0,
):
    mode_norm = str(mode).lower()
    if mode_norm == "fixed":
        return float(fixed_threshold_m), {"mode": "fixed", "threshold_m": float(fixed_threshold_m)}
    if mode_norm in {"adaptive", "knee", "adaptive_knee"}:
        threshold = estimate_knee_threshold(
            errors,
            min_threshold_m=float(min_threshold_m),
            max_threshold_m=float(max_threshold_m),
            trim_percentile=float(trim_percentile),
        )
        return threshold, {
            "mode": "adaptive_knee",
            "threshold_m": float(threshold),
            "min_threshold_m": float(min_threshold_m),
            "max_threshold_m": float(max_threshold_m),
            "trim_percentile": float(trim_percentile),
        }
    raise ValueError(f"Unsupported success threshold mode: {mode}")


def compute_path_length(pos_ref) -> float:
    pos = np.asarray(pos_ref, dtype=float)
    if pos.ndim != 2 or pos.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def resolve_global_gate(
    pos_ref,
    *,
    mode="fixed",
    fixed_m=30.0,
    path_ratio=0.05,
    min_m=2.0,
    max_m=100.0,
):
    path_length_m = compute_path_length(pos_ref)
    mode_norm = str(mode).lower()
    if mode_norm == "fixed":
        gate_m = float(fixed_m)
        return gate_m, {
            "mode": "fixed",
            "effective_m": gate_m,
            "fixed_m": float(fixed_m),
            "path_length_m": path_length_m,
        }
    if mode_norm in {"scale_aware", "scale-aware", "relative"}:
        raw_m = path_length_m * float(path_ratio)
        gate_m = float(np.clip(raw_m, float(min_m), float(max_m)))
        return gate_m, {
            "mode": "scale_aware",
            "effective_m": gate_m,
            "path_length_m": path_length_m,
            "path_ratio": float(path_ratio),
            "raw_m": float(raw_m),
            "min_m": float(min_m),
            "max_m": float(max_m),
        }
    raise ValueError(f"Unsupported global gate mode: {mode}")


def _sample_mask_from_pair_mask(n, pair_ids, pair_mask):
    sample_mask = np.ones(int(n), dtype=bool)
    pairs = np.asarray(pair_ids, dtype=int)
    keep_pair = np.asarray(pair_mask, dtype=bool).reshape(-1)
    if pairs.size == 0:
        return sample_mask
    pairs = pairs.reshape(-1, 2)
    m = min(pairs.shape[0], keep_pair.size)
    bad_pairs = pairs[:m][~keep_pair[:m]]
    if bad_pairs.size == 0:
        return sample_mask
    lo = np.clip(np.minimum(bad_pairs[:, 0], bad_pairs[:, 1]), 0, int(n) - 1)
    hi = np.clip(np.maximum(bad_pairs[:, 0], bad_pairs[:, 1]), 0, int(n) - 1)
    diff = np.zeros(int(n) + 1, dtype=int)
    np.add.at(diff, lo, 1)
    np.add.at(diff, hi + 1, -1)
    sample_mask[np.cumsum(diff[:-1]) > 0] = False
    return sample_mask


def compute_drift_regions(
    timestamps,
    pos_ref,
    ape_translation_errors,
    rpe_time_1s_block,
    *,
    drift_rpe_1s_m=2.0,
    drift_ape_slope_mps=1.0,
    drift_ape_jump_m=5.0,
):
    t = np.asarray(timestamps, dtype=float).reshape(-1)
    pos = np.asarray(pos_ref, dtype=float)
    ape = np.asarray(ape_translation_errors, dtype=float).reshape(-1)
    n = min(t.size, pos.shape[0], ape.size)
    if n == 0:
        return compute_success_regions(t, pos, ape, valid_sample_mask=np.array([], dtype=bool))

    t = t[:n]
    pos = pos[:n]
    ape = ape[:n]
    valid_sample = np.isfinite(ape)

    pair_ids = np.asarray(rpe_time_1s_block.get("_pair_ids", np.empty((0, 2), dtype=int)), dtype=int)
    rpe_1s = np.asarray(
        rpe_time_1s_block.get("_error_arrays", {}).get("translation_part", []),
        dtype=float,
    ).reshape(-1)
    if pair_ids.size > 0 and rpe_1s.size > 0:
        pair_ids = pair_ids.reshape(-1, 2)
        m = min(pair_ids.shape[0], rpe_1s.size)
        pair_ok = np.isfinite(rpe_1s[:m]) & (rpe_1s[:m] <= float(drift_rpe_1s_m))
        valid_sample &= _sample_mask_from_pair_mask(n, pair_ids[:m], pair_ok)

    if n > 1:
        dt = np.diff(t)
        dape = np.diff(ape)
        slope = np.full(dape.size, np.nan, dtype=float)
        positive = dt > 0.0
        slope[positive] = dape[positive] / dt[positive]
        drift_step = (
            np.isfinite(slope)
            & (slope > float(drift_ape_slope_mps))
            & np.isfinite(dape)
            & (dape > float(drift_ape_jump_m))
        )
        drift_sample = np.zeros(n, dtype=bool)
        drift_sample[1:] |= drift_step
        active_level = float("inf")
        jump_m = float(drift_ape_jump_m)
        for idx, is_drift in enumerate(drift_step):
            if is_drift and np.isfinite(ape[idx]):
                active_level = min(active_level, float(ape[idx]) + jump_m)
            j = idx + 1
            if np.isfinite(ape[j]) and float(ape[j]) > active_level:
                drift_sample[j] = True
            else:
                active_level = float("inf")
        valid_sample &= ~drift_sample

    return compute_success_regions(t, pos, ape, valid_sample_mask=valid_sample)


def compute_success_regions(timestamps, pos_ref, ape_translation_errors, threshold_m=10.0, valid_sample_mask=None):
    t = np.asarray(timestamps, dtype=float).reshape(-1)
    pos = np.asarray(pos_ref, dtype=float)
    err = np.asarray(ape_translation_errors, dtype=float).reshape(-1)
    n = min(t.size, pos.shape[0], err.size)
    if n == 0:
        return {
            "threshold_m": float(threshold_m),
            "valid_sample_mask": np.array([], dtype=bool),
            "valid_segment_mask": np.array([], dtype=bool),
            "fail_segments": [],
            "success_rate_distance": np.nan,
            "success_rate_time": np.nan,
            "valid_distance_m": 0.0,
            "total_distance_m": 0.0,
            "valid_time_s": 0.0,
            "total_time_s": 0.0,
            "valid_sample_count": 0,
            "total_sample_count": 0,
            "fail_segment_count": 0,
        }

    t = t[:n]
    pos = pos[:n]
    err = err[:n]
    if valid_sample_mask is None:
        valid_sample = np.isfinite(err) & (err <= float(threshold_m))
    else:
        valid_sample = np.asarray(valid_sample_mask, dtype=bool).reshape(-1)[:n] & np.isfinite(err)
    valid_segment = valid_sample[:-1] & valid_sample[1:] if n > 1 else np.array([], dtype=bool)
    seg_dist = np.linalg.norm(np.diff(pos, axis=0), axis=1) if n > 1 else np.array([], dtype=float)
    seg_dt = np.diff(t) if n > 1 else np.array([], dtype=float)
    positive_dt = np.where(seg_dt > 0.0, seg_dt, 0.0)

    total_distance = float(np.sum(seg_dist))
    valid_distance = float(np.sum(seg_dist[valid_segment])) if valid_segment.size else 0.0
    total_time = float(np.sum(positive_dt))
    valid_time = float(np.sum(positive_dt[valid_segment])) if valid_segment.size else 0.0

    fail_segments = []
    fail = ~valid_sample
    i = 0
    distances_from_start = np.zeros(n, dtype=float)
    if n > 1:
        distances_from_start[1:] = np.cumsum(seg_dist)
    while i < n:
        if not fail[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and fail[i + 1]:
            i += 1
        end = i
        fail_errors = err[start : end + 1]
        finite_fail_errors = fail_errors[np.isfinite(fail_errors)]
        max_error = float(np.max(finite_fail_errors)) if finite_fail_errors.size else float("nan")
        mean_error = float(np.mean(finite_fail_errors)) if finite_fail_errors.size else float("nan")
        fail_segments.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "start_time_s": float(t[start] - t[0]),
                "end_time_s": float(t[end] - t[0]),
                "start_distance_m": float(distances_from_start[start]),
                "end_distance_m": float(distances_from_start[end]),
                "duration_s": float(max(0.0, t[end] - t[start])),
                "distance_m": float(max(0.0, distances_from_start[end] - distances_from_start[start])),
                "max_error_m": max_error,
                "mean_error_m": mean_error,
            }
        )
        i += 1

    return {
        "threshold_m": float(threshold_m),
        "valid_sample_mask": valid_sample,
        "valid_segment_mask": valid_segment,
        "fail_segments": fail_segments,
        "success_rate_distance": valid_distance / total_distance if total_distance > 0.0 else np.nan,
        "success_rate_time": valid_time / total_time if total_time > 0.0 else np.nan,
        "valid_distance_m": valid_distance,
        "total_distance_m": total_distance,
        "valid_time_s": valid_time,
        "total_time_s": total_time,
        "valid_sample_count": int(np.count_nonzero(valid_sample)),
        "total_sample_count": int(n),
        "fail_segment_count": int(len(fail_segments)),
    }


def compute_valid_segment_metrics(
    *,
    timestamps,
    pos_ref,
    ape_block,
    rpe_block,
    rpe_time_1s_block,
    threshold_m=10.0,
    threshold_info=None,
    global_gate_mode="fixed",
    global_gate_m=30.0,
    global_gate_path_ratio=0.05,
    global_gate_min_m=2.0,
    global_gate_max_m=100.0,
    global_gate_percentile=5.0,
    drift_rpe_1s_m=2.0,
    drift_ape_slope_mps=1.0,
    drift_ape_jump_m=5.0,
    include_raw=True,
    include_masks=False,
):
    ape_errors = np.asarray(ape_block.get("_error_arrays", {}).get("translation_part", []), dtype=float)
    finite_errors = ape_errors[np.isfinite(ape_errors)]
    gate_value = float(np.percentile(finite_errors, float(global_gate_percentile))) if finite_errors.size else np.nan
    effective_gate_m, gate_info = resolve_global_gate(
        pos_ref,
        mode=global_gate_mode,
        fixed_m=float(global_gate_m),
        path_ratio=float(global_gate_path_ratio),
        min_m=float(global_gate_min_m),
        max_m=float(global_gate_max_m),
    )
    global_gate_failed = bool(not np.isfinite(gate_value) or gate_value > float(effective_gate_m))
    regions = compute_drift_regions(
        timestamps=timestamps,
        pos_ref=pos_ref,
        ape_translation_errors=ape_errors,
        rpe_time_1s_block=rpe_time_1s_block,
        drift_rpe_1s_m=float(drift_rpe_1s_m),
        drift_ape_slope_mps=float(drift_ape_slope_mps),
        drift_ape_jump_m=float(drift_ape_jump_m),
    )
    valid_sample = np.asarray(regions["valid_sample_mask"], dtype=bool)
    valid_segment = np.asarray(regions["valid_segment_mask"], dtype=bool)
    valid_metric_sample = np.zeros(valid_sample.size, dtype=bool)
    if valid_segment.size > 0:
        valid_metric_sample[:-1] |= valid_segment
        valid_metric_sample[1:] |= valid_segment
    success = {
        key: value
        for key, value in regions.items()
        if key not in {"valid_sample_mask", "valid_segment_mask"}
    }
    has_valid_segments = bool(np.count_nonzero(valid_segment) > 0)
    if global_gate_failed:
        success["case_status"] = "globally_unstable" if has_valid_segments else "globally_failed"
    else:
        success["case_status"] = "valid_segment"
    success["global_gate_failed"] = bool(global_gate_failed)
    success["global_gate_m"] = float(effective_gate_m)
    success["global_gate_info"] = gate_info
    success["global_gate_mode"] = str(gate_info["mode"])
    success["global_gate_fixed_m"] = float(global_gate_m)
    success["global_gate_path_length_m"] = float(gate_info["path_length_m"])
    success["global_gate_percentile"] = float(global_gate_percentile)
    success["global_gate_value_m"] = float(gate_value)
    if global_gate_failed:
        success["global_gate_warning"] = (
            "global gate failed; success rate and valid metrics are computed from local valid segments"
        )
    success["drift_rpe_1s_m"] = float(drift_rpe_1s_m)
    success["drift_ape_slope_mps"] = float(drift_ape_slope_mps)
    success["drift_ape_jump_m"] = float(drift_ape_jump_m)
    if threshold_info is not None:
        success["threshold"] = threshold_info
    if include_masks:
        success["valid_sample_mask"] = valid_sample
        success["valid_metric_sample_mask"] = valid_metric_sample
        success["valid_segment_mask"] = valid_segment
    return {
        "success": success,
        "ape": _filter_ape_block(ape_block, valid_metric_sample, include_raw=include_raw),
        "rpe": _filter_rpe_block(rpe_block, valid_segment, include_raw=include_raw),
        "rpe_time_1s": _filter_rpe_block(rpe_time_1s_block, valid_segment, include_raw=include_raw),
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
