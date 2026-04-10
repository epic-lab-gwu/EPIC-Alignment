import numpy as np
from scipy.spatial.transform import Rotation as R

from .math_utils import compute_error_statistics, poses_se3_from_traj, relative_se3


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

