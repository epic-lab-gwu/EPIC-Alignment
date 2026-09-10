from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .math_utils import compute_error_statistics, poses_se3_from_traj

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
        if distances.size < 2:
            return []
        if not np.isfinite(delta) or not np.isfinite(tol):
            for i in range(distances.size - 1):
                offset = i + 1
                distances_from_here = distances[offset:] - distances[i]
                candidate_index = int(np.argmin(np.abs(distances_from_here - delta)))
                if np.abs(distances_from_here[candidate_index] - delta) > tol:
                    continue
                id_pairs.append((i, candidate_index + offset))
            return id_pairs

        starts = np.arange(distances.size - 1, dtype=int)
        targets = distances[starts] + float(delta)
        insert_ids = np.searchsorted(distances, targets, side="left")
        right_ids = np.maximum(insert_ids, starts + 1)
        left_ids = right_ids - 1
        left_valid = left_ids >= starts + 1
        right_valid = right_ids < distances.size
        left_safe = np.clip(left_ids, 0, distances.size - 1)
        right_safe = np.clip(right_ids, 0, distances.size - 1)
        left_error = np.where(left_valid, np.abs(distances[left_safe] - targets), np.inf)
        right_error = np.where(right_valid, np.abs(distances[right_safe] - targets), np.inf)
        choose_left = left_error <= right_error
        candidate_ids = np.where(choose_left, left_safe, right_safe)
        candidate_error = np.where(choose_left, left_error, right_error)

        first_duplicate_ids = np.searchsorted(
            distances, distances[candidate_ids], side="left"
        )
        candidate_ids = np.maximum(first_duplicate_ids, starts + 1)
        accepted = candidate_error <= float(tol)
        return list(zip(starts[accepted].tolist(), candidate_ids[accepted].tolist()))
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
        if np.isfinite(delta) and np.isfinite(tol):
            starts = np.arange(stamps.size - 1, dtype=int)
            targets = stamps[starts] + float(delta)
            insert_ids = np.searchsorted(stamps, targets, side="left")
            candidates = insert_ids[:, None] + np.array([-1, 0, 1], dtype=int)
            valid = (candidates > starts[:, None]) & (candidates < stamps.size)
            safe_candidates = np.clip(candidates, 0, stamps.size - 1)
            errors = np.abs(stamps[safe_candidates] - targets[:, None])
            errors[~valid] = np.inf
            best_columns = np.argmin(errors, axis=1)
            rows = np.arange(starts.size, dtype=int)
            best_errors = errors[rows, best_columns]
            accepted = best_errors <= tol
            return list(
                zip(
                    starts[accepted].tolist(),
                    candidates[rows[accepted], best_columns[accepted]].tolist(),
                )
            )

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


def rpe_pairs_chained(timestamps, delta=1.0, valid_segment_mask=None):
    """Cover each supported interval once with contiguous nearest-duration pairs.

    Each endpoint starts the next pair. Include the final short tail; restart
    only after explicitly blocked intervals. Ties choose the earlier endpoint.
    """
    t = np.asarray(timestamps, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Chained RPE requires finite, strictly increasing timestamps")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Chained RPE duration must be positive")
    valid = np.ones(max(0, len(t)-1), dtype=bool) if valid_segment_mask is None else np.asarray(valid_segment_mask, dtype=bool)
    if valid.shape != (max(0, len(t)-1),):
        raise ValueError("RPE interval mask must match trajectory intervals")
    edges = np.diff(np.r_[False, valid, False].astype(int))
    pairs = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        i = int(start)
        while i < end:
            target = t[i] + delta
            hi = int(np.clip(np.searchsorted(t, target), i+1, end))
            lo = max(i+1, hi-1)
            j = lo if abs(t[lo]-target) <= abs(t[hi]-target) else hi
            pairs.append((i, j))
            i = j
    return pairs


def rpe_pairs_per_pose(timestamps, delta=1.0, valid_segment_mask=None):
    """One owned RPE check per pose, overlapping in time.

    Prefer the forward endpoint nearest delta, shortening near each supported
    fragment's end. Its final pose owns a backward check nearest delta. Isolated
    poses have no check. No pair crosses an explicitly blocked interval.
    """
    t = np.asarray(timestamps, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Per-pose RPE requires finite, strictly increasing timestamps")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Per-pose RPE duration must be positive")
    valid = np.ones(max(0, len(t)-1), dtype=bool) if valid_segment_mask is None else np.asarray(valid_segment_mask, dtype=bool)
    if valid.shape != (max(0, len(t)-1),):
        raise ValueError("RPE interval mask must match trajectory intervals")
    edges = np.diff(np.r_[False, valid, False].astype(int))
    pairs, owners = [], []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ids = np.arange(start, end)
        target = t[ids] + delta
        hi = np.minimum(np.maximum(np.searchsorted(t, target), ids+1), end)
        lo = np.maximum(ids+1, hi-1)
        ends = np.where(abs(t[lo]-target) <= abs(t[hi]-target), lo, hi)
        pairs.extend(zip(ids.tolist(), ends.tolist()))
        owners.extend(ids.tolist())
        target = t[end] - delta
        hi = int(np.clip(np.searchsorted(t, target), start, end-1))
        lo = max(int(start), hi-1)
        begin = lo if abs(t[lo]-target) <= abs(t[hi]-target) else hi
        pairs.append((begin, int(end)))
        owners.append(int(end))
    return pairs, np.asarray(owners, dtype=int)


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
    valid_segment_mask=None,
    pair_indices=None,
):
    if len(pos_ref) != len(pos_est):
        raise ValueError("RPE requires trajectories with the same number of poses.")

    pos_ref = np.asarray(pos_ref, dtype=float)
    pos_est = np.asarray(pos_est, dtype=float)
    T_ref = poses_se3_from_traj(pos_ref, quat_ref)
    T_est = poses_se3_from_traj(pos_est, quat_est)
    pair_source = T_ref if pairs_from_reference else T_est
    id_pairs = list(pair_indices) if pair_indices is not None else build_rpe_pairs(
        pair_source,
        delta=delta,
        delta_unit=delta_unit,
        rel_delta_tol=rel_delta_tol,
        all_pairs=all_pairs,
        timestamps=timestamps,
    )
    if any(i < 0 or j >= len(pos_ref) or j <= i for i, j in id_pairs):
        raise ValueError("Invalid RPE pair indices")
    if valid_segment_mask is not None:
        valid = np.asarray(valid_segment_mask, dtype=bool)
        if valid.shape != (max(0, len(pos_ref)-1),):
            raise ValueError("RPE interval mask must match trajectory intervals")
        invalid_prefix = np.r_[0, np.cumsum(~valid)]
        id_pairs = [(i, j) for i, j in id_pairs if invalid_prefix[j] == invalid_prefix[i]]
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
        result["_reference_motion"] = {
            "translation_m": np.linalg.norm(t_ref_rel, axis=1),
            "rotation_deg": np.degrees(R.from_matrix(R_ref_rel).magnitude()),
        }
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
        if "_reference_motion" in rpe_block:
            result["_reference_motion"] = {
                key: np.asarray(values)[keep]
                for key, values in rpe_block["_reference_motion"].items()
            }
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


def _bbox_diag(pos_ref) -> float:
    pos = np.asarray(pos_ref, dtype=float)
    if pos.ndim != 2 or pos.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(np.max(pos, axis=0) - np.min(pos, axis=0)))


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


def _finite_percentile(values, percentile: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.percentile(finite, float(percentile)))


def resolve_drift_thresholds(
    timestamps,
    pos_ref,
    rpe_time_1s_block=None,
    *,
    mode="adaptive",
    fixed_rpe_1s_m=2.0,
    fixed_ape_slope_mps=1.0,
    fixed_ape_jump_m=5.0,
):
    """Resolve local drift thresholds from the reference trajectory scale.

    The fixed values are kept as fallbacks for reproducibility. In adaptive
    mode, thresholds are derived from the case's reference 1s motion, spatial
    extent, path length, and speed distribution.
    """

    mode_norm = str(mode).lower()
    if mode_norm == "fixed":
        return {
            "mode": "fixed",
            "rpe_1s_m": float(fixed_rpe_1s_m),
            "ape_slope_mps": float(fixed_ape_slope_mps),
            "ape_jump_m": float(fixed_ape_jump_m),
        }
    if mode_norm not in {"adaptive", "case_adaptive", "case-aware", "case_aware"}:
        raise ValueError(f"Unsupported drift threshold mode: {mode}")

    t = np.asarray(timestamps, dtype=float).reshape(-1)
    pos = np.asarray(pos_ref, dtype=float)
    n = min(t.size, pos.shape[0])
    if n < 2:
        return {
            "mode": "adaptive",
            "rpe_1s_m": float(fixed_rpe_1s_m),
            "ape_slope_mps": float(fixed_ape_slope_mps),
            "ape_jump_m": float(fixed_ape_jump_m),
            "fallback": True,
        }
    t = t[:n]
    pos = pos[:n]
    seg_dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    seg_dt = np.diff(t)
    positive = seg_dt > 0.0
    speeds = np.divide(seg_dist, seg_dt, out=np.full_like(seg_dist, np.nan), where=positive)
    path_length = float(np.sum(seg_dist))
    duration = float(np.sum(np.where(positive, seg_dt, 0.0)))
    bbox_diag = _bbox_diag(pos)
    median_speed = _finite_percentile(speeds, 50.0, default=path_length / max(duration, 1e-9))
    p95_speed = _finite_percentile(speeds, 95.0, default=median_speed)

    ref_1s_motion = []
    if isinstance(rpe_time_1s_block, dict):
        pair_ids = np.asarray(rpe_time_1s_block.get("_pair_ids", np.empty((0, 2), dtype=int)), dtype=int)
        if pair_ids.size > 0:
            pair_ids = pair_ids.reshape(-1, 2)
            valid = (pair_ids[:, 0] >= 0) & (pair_ids[:, 1] >= 0) & (pair_ids[:, 0] < n) & (pair_ids[:, 1] < n)
            if np.any(valid):
                ids = pair_ids[valid]
                ref_1s_motion = np.linalg.norm(pos[ids[:, 1]] - pos[ids[:, 0]], axis=1)
    ref_1s_p90 = _finite_percentile(ref_1s_motion, 90.0, default=median_speed)

    spatial_floor = max(0.03 * bbox_diag, 0.003 * path_length, 0.05)
    rpe_1s_m = max(2.0 * ref_1s_p90, spatial_floor)
    rpe_1s_m = float(np.clip(rpe_1s_m, 0.05, max(float(fixed_rpe_1s_m), 2.0 * ref_1s_p90, 0.05 * path_length, 0.10)))

    ape_jump_m = max(2.5 * rpe_1s_m, 0.05 * bbox_diag, 0.005 * path_length, 0.10)
    ape_jump_m = float(np.clip(ape_jump_m, 0.10, max(float(fixed_ape_jump_m), 0.10 * path_length, 0.25)))

    ape_slope_mps = max(0.50 * rpe_1s_m, 0.25 * median_speed, 0.05)
    ape_slope_mps = float(np.clip(ape_slope_mps, 0.05, max(float(fixed_ape_slope_mps), 0.50 * p95_speed, 0.10)))

    return {
        "mode": "adaptive",
        "rpe_1s_m": rpe_1s_m,
        "ape_slope_mps": ape_slope_mps,
        "ape_jump_m": ape_jump_m,
        "fixed_rpe_1s_m": float(fixed_rpe_1s_m),
        "fixed_ape_slope_mps": float(fixed_ape_slope_mps),
        "fixed_ape_jump_m": float(fixed_ape_jump_m),
        "ref_path_length_m": path_length,
        "ref_duration_s": duration,
        "ref_bbox_diag_m": float(bbox_diag),
        "ref_1s_motion_p90_m": float(ref_1s_p90),
        "ref_speed_median_mps": float(median_speed),
        "ref_speed_p95_mps": float(p95_speed),
    }


def compute_drift_regions(
    timestamps,
    pos_ref,
    ape_translation_errors,
    rpe_time_1s_block,
    *,
    ape_threshold_m=None,
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
    if ape_threshold_m is not None and np.isfinite(float(ape_threshold_m)) and float(ape_threshold_m) > 0.0:
        valid_sample &= ape <= float(ape_threshold_m)

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


def apply_input_coverage_to_success(
    success: dict,
    input_coverage: dict | None,
    *,
    set_primary: bool = True,
) -> dict:
    """Add complete-reference SR without discarding overlap-local SR.

    ``compute_success_regions`` can only score the temporally supported overlap.
    This helper treats unsupported portions of the complete reference as
    unsuccessful, so a short but locally accurate estimate cannot be reported
    as a 100% complete-run success.
    """
    if not isinstance(success, dict):
        raise TypeError("success must be a dictionary")
    coverage = input_coverage if isinstance(input_coverage, dict) else {}
    if not coverage:
        return success

    def finite(value, default=np.nan) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if np.isfinite(out) else float(default)

    def ratio(numerator, denominator) -> float:
        num = finite(numerator)
        den = finite(denominator)
        if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
            return float("nan")
        return float(np.clip(num / den, 0.0, 1.0))

    local_distance = finite(
        success.get("local_success_rate_distance", success.get("success_rate_distance"))
    )
    local_time = finite(success.get("local_success_rate_time", success.get("success_rate_time")))
    raw_local_distance = finite(
        success.get("raw_local_success_rate_distance", success.get("raw_success_rate_distance"))
    )
    raw_local_time = finite(
        success.get("raw_local_success_rate_time", success.get("raw_success_rate_time"))
    )
    local_total_distance = finite(success.get("local_total_distance_m", success.get("total_distance_m")))
    local_total_time = finite(success.get("local_total_time_s", success.get("total_time_s")))
    complete_total_distance = finite(coverage.get("reference_path_length_m"))
    complete_total_time = finite(coverage.get("reference_duration_s"))

    complete_distance = ratio(success.get("valid_distance_m"), complete_total_distance)
    complete_time = ratio(success.get("valid_time_s"), complete_total_time)
    raw_complete_distance = ratio(success.get("raw_valid_distance_m"), complete_total_distance)
    raw_complete_time = ratio(success.get("raw_valid_time_s"), complete_total_time)

    success.update(
        {
            "success_rate_scope": "complete_reference" if set_primary else "overlap_local",
            "local_success_rate_distance": local_distance,
            "local_success_rate_time": local_time,
            "raw_local_success_rate_distance": raw_local_distance,
            "raw_local_success_rate_time": raw_local_time,
            "complete_success_rate_distance": complete_distance,
            "complete_success_rate_time": complete_time,
            "raw_complete_success_rate_distance": raw_complete_distance,
            "raw_complete_success_rate_time": raw_complete_time,
            "local_total_distance_m": local_total_distance,
            "local_total_time_s": local_total_time,
            "complete_total_distance_m": complete_total_distance,
            "complete_total_time_s": complete_total_time,
            "temporal_coverage_ratio": finite(coverage.get("temporal_coverage_ratio")),
            "path_coverage_ratio": finite(coverage.get("path_coverage_ratio")),
            "input_coverage_status": str(coverage.get("coverage_status", "ok")),
            "input_coverage_hard_reasons": list(coverage.get("coverage_hard_reasons", [])),
            "input_coverage_soft_reasons": list(coverage.get("coverage_soft_reasons", [])),
        }
    )
    if set_primary:
        success["success_rate_distance"] = complete_distance
        success["success_rate_time"] = complete_time
        success["raw_success_rate_distance"] = raw_complete_distance
        success["raw_success_rate_time"] = raw_complete_time

    coverage_status = str(coverage.get("coverage_status", "ok"))
    previous_status = str(success.get("sr_reliability_status", "ok"))
    rank = {"ok": 0, "warning": 1, "failed": 2}
    success["sr_reliability_status"] = max(
        (previous_status, coverage_status), key=lambda item: rank.get(item, 0)
    )
    if coverage_status in {"warning", "failed"}:
        note = (
            "Reference support is incomplete; complete-reference SR counts "
            "the unsupported part as unsuccessful."
        )
        previous_note = str(success.get("sr_warning_explanation", "") or "")
        if note not in previous_note:
            success["sr_warning_explanation"] = " ".join(
                part for part in (previous_note, note) if part
            )
    return success


def _motion_relative_pair_pass(block):
    """Apply the same pose-motion limits to time or consecutive pose pairs."""
    pairs = np.asarray(block.get("_pair_ids", []), dtype=int).reshape(-1, 2)
    errors = block.get("_error_arrays", {})
    motion = block.get("_reference_motion", {})
    trans_error = np.asarray(errors.get("translation_part", []), dtype=float)
    rot_error = np.asarray(errors.get("rotation_angle_deg", []), dtype=float)
    trans_motion = np.asarray(motion.get("translation_m", []), dtype=float)
    rot_motion = np.asarray(motion.get("rotation_deg", []), dtype=float)
    if any(len(a) != len(pairs) for a in (trans_error, rot_error, trans_motion, rot_motion)):
        raise ValueError("SR requires translation/rotation errors and reference motion for each pair")
    trans_limit = np.where(trans_motion < 0.1, 0.3, 3.0 * trans_motion)
    rot_limit = np.where(rot_motion < 1.0, 3.0, 3.0 * rot_motion)
    return pairs, (np.isfinite(trans_error) & np.isfinite(rot_error)
                   & np.isfinite(trans_motion) & np.isfinite(rot_motion)
                   & (trans_error < trans_limit) & (rot_error < rot_limit))


def _longest_successful_group(timestamps, valid_segment, max_gap_s=10.0, hard_boundaries=None):
    """Keep the group with most successful time; short separating gaps stay false.

    Groups are joined only across gaps strictly shorter than max_gap_s. Ties
    select the earliest group. Neither gap duration nor sample count earns credit.
    """
    t = np.asarray(timestamps, dtype=float).reshape(-1)
    valid = np.asarray(valid_segment, dtype=bool).reshape(-1)
    if valid.size != max(0, t.size - 1):
        raise ValueError("Interval mask must have one entry per timestamp interval")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
        raise ValueError("Group selection requires finite, nondecreasing timestamps")
    barriers = np.zeros_like(valid) if hard_boundaries is None else np.asarray(hard_boundaries, dtype=bool)
    if barriers.shape != valid.shape:
        raise ValueError("Hard boundaries must match interval mask")
    valid = valid & ~barriers
    edges = np.diff(np.r_[False, valid, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    groups = []
    for start, end in zip(starts, ends):
        duration = float(t[end] - t[start])
        if (groups and t[start] - t[groups[-1]["end_index"]] < max_gap_s
                and not np.any(barriers[groups[-1]["end_index"]:start])):
            groups[-1]["end_index"] = int(end)
            groups[-1]["successful_duration_s"] += duration
        else:
            groups.append(dict(start_index=int(start), end_index=int(end),
                               successful_duration_s=duration))
    result = np.zeros_like(valid)
    selected = max(groups, key=lambda g: g["successful_duration_s"]) if groups else None
    if selected is not None:
        start, end = selected["start_index"], selected["end_index"]
        result[start:end] = valid[start:end]
    return result, dict(policy="longest_successful_duration", gap_threshold_s=float(max_gap_s),
                        gap_comparison="strictly_less_than", tie_break="earliest",
                        groups=groups, selected_group=selected,
                        discarded_successful_intervals=int(np.count_nonzero(valid & ~result)))


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
    drift_threshold_mode="adaptive",
    include_raw=True,
    include_masks=False,
    quat_ref=None,
    pos_est=None,
    quat_est=None,
    input_coverage=None,
):
    """Score every pose with overlapping RPE; require both interval endpoints.

    A passing check validates only its owner pose, never its interior. A failed
    pose cannot enter valid ATE through an adjacent interval. SR is frozen before
    optional valid-only world alignment and never depends on absolute APE.
    """
    t = np.asarray(timestamps, dtype=float)
    pos = np.asarray(pos_ref, dtype=float).reshape(-1, 3)
    n = len(t)
    from .adaptive_association import reset_crossing_mask
    resets = (input_coverage or {}).get("reset_intervals", [])
    blocked = reset_crossing_mask(t, resets)
    per_pose = all(v is not None for v in (quat_ref, pos_est, quat_est))
    if per_pose:
        check_pairs, owners = rpe_pairs_per_pose(t, valid_segment_mask=~blocked)
        rpe_time_1s_block = compute_rpe(
            pos, quat_ref, pos_est, quat_est, include_raw=True, max_pairs=0,
            pair_indices=check_pairs,
        )
    elif np.any(blocked):
        rpe_time_1s_block = _filter_rpe_block(rpe_time_1s_block, ~blocked)
    if np.any(blocked):
        rpe_block = _filter_rpe_block(rpe_block, ~blocked)
    pairs, pair_ok = _motion_relative_pair_pass(rpe_time_1s_block)
    m = len(pairs)
    if np.any((pairs[:, 0] < 0) | (pairs[:, 1] >= n) | (pairs[:, 1] <= pairs[:, 0])):
        raise ValueError("Invalid per-pose RPE pair indices")
    if not per_pose:
        owners = pairs[:, 0]
    checks = np.bincount(owners, minlength=n)
    failures = np.bincount(owners[~pair_ok], minlength=n)
    pose_checked = checks > 0
    pose_pass = pose_checked & (failures == 0)
    fallback_pairs = np.empty((0, 2), dtype=int)
    fallback_ok = np.array([], dtype=bool)
    if not per_pose:
        # Compatibility for callers supplying error blocks without pose arrays.
        seq_pairs, seq_ok = _motion_relative_pair_pass(rpe_block)
        eligible = (seq_pairs[:, 0] >= 0) & (seq_pairs[:, 1] < n)
        eligible &= seq_pairs[:, 1] == seq_pairs[:, 0] + 1
        ids = np.flatnonzero(eligible)
        ids = ids[~pose_checked[seq_pairs[ids, 0]]]
        fallback_pairs, fallback_ok = seq_pairs[ids], seq_ok[ids]
        pose_checked[fallback_pairs[:, 0]] = True
        pose_pass[fallback_pairs[:, 0]] = fallback_ok
        # The final pose has no outgoing pair; use its closest available incoming check.
        if n and not pose_checked[-1]:
            all_pairs = np.concatenate((pairs, fallback_pairs))
            all_ok = np.r_[pair_ok, fallback_ok]
            ids = np.flatnonzero(all_pairs[:, 1] == n-1)
            if ids.size:
                k = ids[np.argmin(abs(t[-1]-t[all_pairs[ids, 0]]-1.0))]
                pose_checked[-1], pose_pass[-1] = True, all_ok[k]
    covered = pose_checked[:-1] & pose_checked[1:] & ~blocked
    valid_segment = pose_pass[:-1] & pose_pass[1:] & ~blocked
    unfiltered_segment = valid_segment.copy()
    valid_segment, group_selection = _longest_successful_group(t, valid_segment)
    valid_sample = np.zeros(n, dtype=bool)
    valid_sample[:-1] |= valid_segment
    valid_sample[1:] |= valid_segment
    valid_sample &= pose_pass
    # Use the common reporting shape, then replace totals with the pair-based
    # segment mask (adjacent passing intervals need not share passing pairs).
    success = compute_success_regions(t, pos, np.zeros(n), valid_sample_mask=valid_sample)
    dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.maximum(np.diff(t), 0)
    valid_dist, valid_time = float(np.sum(dist[valid_segment])), float(np.sum(dt[valid_segment]))
    # Report failures by intervals, including gaps between successful runs.
    # A shared boundary pose can belong to a successful adjacent interval.
    boundaries = np.diff(np.r_[False, ~valid_segment, False].astype(int))
    starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
    distances = np.r_[0.0, np.cumsum(dist)]
    ape_errors = np.asarray(ape_block.get("_error_arrays", {}).get("translation_part", []))
    failed_regions = []
    for start, end in zip(starts, ends):
        region_errors = ape_errors[start:end + 1]
        finite_errors = region_errors[np.isfinite(region_errors)]
        failed_regions.append({
            "start_index": int(start), "end_index": int(end),
            "start_time_s": float(t[start] - t[0]), "end_time_s": float(t[end] - t[0]),
            "start_distance_m": float(distances[start]), "end_distance_m": float(distances[end]),
            "duration_s": float(t[end] - t[start]),
            "distance_m": float(distances[end] - distances[start]),
            "max_error_m": float(np.max(finite_errors)) if finite_errors.size else float("nan"),
            "mean_error_m": float(np.mean(finite_errors)) if finite_errors.size else float("nan"),
        })
    local_dist = valid_dist / np.sum(dist) if np.sum(dist) > 0 else float("nan")
    local_time = valid_time / np.sum(dt) if np.sum(dt) > 0 else float("nan")
    success.update({
        "policy": "rpe_1s_motion_relative",
        "pairing_policy": "overlapping_per_pose_nearest_1s" if per_pose else "supplied_per_pose_pairs",
        "interval_policy": "both_endpoint_poses_pass",
        "tail_policy": "shortened_forward_then_backward_at_fragment_end",
        "checked_pose_count": int(np.count_nonzero(pose_checked)),
        "passing_pose_count": int(np.count_nonzero(pose_pass)),
        "nominal_pair_duration_s": 1.0,
        "reset_intervals": resets,
        "reset_blocked_interval_count": int(np.count_nonzero(blocked)),
        "group_selection": group_selection,
        "fail_segments": failed_regions, "fail_segment_count": len(failed_regions),
        "threshold": {"mode": "rpe_1s_motion_relative", "relative_ratio": 3.0,
                      "small_translation_m": 0.1, "small_rotation_deg": 1.0,
                      "absolute_translation_m": 0.3, "absolute_rotation_deg": 3.0},
        "valid_distance_m": valid_dist, "valid_time_s": valid_time,
        "success_rate_distance": local_dist, "success_rate_time": local_time,
        "local_success_rate_distance": local_dist, "local_success_rate_time": local_time,
        "local_total_distance_m": float(np.sum(dist)), "local_total_time_s": float(np.sum(dt)),
        "success_rate_scope": "overlap_local", "sr_reliability_status": "ok",
        "sr_warning_explanation": "", "case_status": "valid_segment" if np.any(valid_segment) else "no_valid_segments",
        "global_gate_failed": False, "global_gate_mode": "disabled",
        "scored_pair_count": m + len(fallback_pairs),
        "passing_pair_count": int(np.count_nonzero(pair_ok) + np.count_nonzero(fallback_ok)),
        "one_second_pair_count": m,
        "one_second_passing_pair_count": int(np.count_nonzero(pair_ok)),
        "sequential_fallback_pair_count": len(fallback_pairs),
        "sequential_fallback_passing_pair_count": int(np.count_nonzero(fallback_ok)),
        "unscored_segment_count": int(np.count_nonzero(~covered)),
        "unscored_interval_policy": "unsuccessful",
    })
    success.pop("threshold_m", None)
    raw_dist = float(np.sum(dist[unfiltered_segment]))
    raw_time = float(np.sum(dt[unfiltered_segment]))
    raw_local_dist = raw_dist / np.sum(dist) if np.sum(dist) > 0 else float("nan")
    raw_local_time = raw_time / np.sum(dt) if np.sum(dt) > 0 else float("nan")
    success.update(raw_valid_distance_m=raw_dist, raw_valid_time_s=raw_time,
                   raw_success_rate_distance=raw_local_dist, raw_success_rate_time=raw_local_time,
                   raw_local_success_rate_distance=raw_local_dist,
                   raw_local_success_rate_time=raw_local_time)
    apply_input_coverage_to_success(success, input_coverage, set_primary=True)
    alignment = {"applied": False, "reason": "sr_at_least_half", "mode": "se3",
                 "sr_scope": success["success_rate_scope"], "sr_metric": "distance"}
    trigger_sr = success["success_rate_distance"]
    if not np.isfinite(trigger_sr):
        trigger_sr = success["success_rate_time"]
        alignment["sr_metric"] = "time"
    if not np.isfinite(trigger_sr):
        alignment["reason"] = "sr_unavailable"
    if np.isfinite(trigger_sr) and trigger_sr < 0.5:
        if not np.any(valid_sample):
            alignment["reason"] = "no_valid_segments"
        elif any(v is None for v in (quat_ref, pos_est, quat_est)):
            alignment["reason"] = "poses_unavailable"
        else:
            from .trajectory_alignment import _solve_extrinsic_and_world_alignment
            ref = pos[valid_sample]
            est = np.asarray(pos_est)[valid_sample]
            q_ref = np.asarray(quat_ref)[valid_sample]
            q_est = np.asarray(quat_est)[valid_sample]
            # Reuse the original SE3 solver, including rotation-first fitting,
            # robust translation fitting and world-candidate selection. These
            # poses already include the calibrated extrinsics; only refit world.
            fitted = _solve_extrinsic_and_world_alignment(
                pr_sync=est, qr_sync=q_est,
                pos_gt_solve=ref, quat_gt_solve=q_ref,
                pr_solve=est, qr_solve=q_est,
                global_align_mode="se3",
                disable_extrinsic_calibration=True,
            )
            rw = np.asarray(fitted["Rw_calc"], dtype=float)
            tw = np.asarray(fitted["tw_calc"], dtype=float)
            choice = fitted["step3_choice"]
            method = str(choice["step3_alignment_mode"])
            realigned_pos = np.asarray(pos_est) @ rw.T + tw
            realigned_quat = (R.from_matrix(rw) * R.from_quat(quat_est)).as_quat()
            original_axes = ape_block.get("_x_axis")
            ape_block = compute_ape(pos, quat_ref, realigned_pos, realigned_quat, include_raw=True)
            if original_axes is not None:
                ape_block["_x_axis"] = original_axes
            alignment.update(applied=True, reason="sr_below_half", method=method,
                             solver="_solve_extrinsic_and_world_alignment",
                             solver_info=choice,
                             rotation=rw.tolist(), translation_m=tw.tolist(),
                             sample_count=int(np.count_nonzero(valid_sample)))
            # Relative pose errors are invariant to a common left SE3 transform;
            # retain the original RPE pairs/errors, then filter by the frozen mask.
    success["valid_only_world_alignment"] = alignment
    success["valid_sample_count"] = int(np.count_nonzero(valid_sample))
    success.pop("valid_sample_mask", None)
    success.pop("valid_segment_mask", None)
    if include_masks:
        success["valid_sample_mask"] = valid_sample
        success["valid_metric_sample_mask"] = valid_sample
        success["valid_segment_mask"] = valid_segment
        success["unfiltered_valid_segment_mask"] = unfiltered_segment
        success["rpe_pair_pass_mask"] = pair_ok
        success["pose_check_pair_ids"] = pairs
        success["pose_check_owner_ids"] = owners
        success["pose_check_pass_mask"] = pose_pass
        success["pose_checked_mask"] = pose_checked
        success["sequential_fallback_pair_ids"] = fallback_pairs
        success["sequential_fallback_pair_pass_mask"] = fallback_ok
    return {
        "success": success,
        "ape": _filter_ape_block(ape_block, valid_sample, include_raw=include_raw),
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
