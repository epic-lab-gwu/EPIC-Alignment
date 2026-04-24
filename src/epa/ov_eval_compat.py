from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.evaluation import compute_ape_evo_style, compute_rpe_evo_style
from epa.core.math_utils import compute_error_statistics, normalize_quat_array
from epa.metric_cli_common import project_to_plane
from epa.traj_tool import build_parser as build_traj_parser
from epa.traj_tool import run as run_traj


_VALID_ALIGN_MODES = {"posyaw", "posyawsingle", "se3", "se3single", "sim3", "none"}
_DEFAULT_ASSOC_MAX_DIFF = 0.02


def _fmt(v: float, nd: int = 3) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{float(v):.{nd}f}"


def _load_ov_eval_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    pos: list[list[float]] = []
    quat: list[list[float]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 8:
                continue
            vals = [float(x) for x in parts[:8]]
            times.append(vals[0])
            pos.append(vals[1:4])
            quat.append(vals[4:8])

    if len(times) == 0:
        raise ValueError(f"Could not parse any trajectory samples from: {path}")

    t = np.asarray(times, dtype=float)
    p = np.asarray(pos, dtype=float)
    q = normalize_quat_array(np.asarray(quat, dtype=float))
    return t, p, q


def _load_ov_eval_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    pos: list[list[float]] = []
    quat: list[list[float]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x for x in line.split(",") if x != ""]
            if len(parts) < 8:
                continue
            vals = [float(x) for x in parts[:8]]
            times.append(vals[0] * 1e-9)
            pos.append(vals[1:4])
            quat.append([vals[5], vals[6], vals[7], vals[4]])

    if len(times) == 0:
        raise ValueError(f"Could not parse any CSV samples from: {path}")

    t = np.asarray(times, dtype=float)
    p = np.asarray(pos, dtype=float)
    q = normalize_quat_array(np.asarray(quat, dtype=float))
    return t, p, q


def _load_pose_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".csv":
        return _load_ov_eval_csv(path)
    return _load_ov_eval_txt(path)


def _associate_est_gt(
    t_est: np.ndarray,
    p_est: np.ndarray,
    q_est: np.ndarray,
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    q_gt: np.ndarray,
    max_diff: float,
    offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i_est: list[int] = []
    i_gt: list[int] = []
    gt_ptr = 0

    for i, te in enumerate(np.asarray(t_est, dtype=float).reshape(-1)):
        tgt = te + float(offset)
        best_diff = float(max_diff)
        best_gt = -1

        while gt_ptr < int(t_gt.size) and t_gt[gt_ptr] < tgt and abs(float(t_gt[gt_ptr] - tgt)) > float(max_diff):
            gt_ptr += 1

        while gt_ptr < int(t_gt.size) and abs(float(t_gt[gt_ptr] - tgt)) <= float(max_diff):
            cur = abs(float(t_gt[gt_ptr] - tgt))
            if cur >= best_diff:
                break
            best_diff = cur
            best_gt = int(gt_ptr)
            gt_ptr += 1

        if best_gt != -1:
            i_est.append(int(i))
            i_gt.append(best_gt)

    if len(i_est) < 3:
        raise ValueError(
            "Unable to associate enough timestamps between estimate and ground truth. "
            f"matches={len(i_est)}, max_diff={max_diff}, offset={offset}."
        )

    ie = np.asarray(i_est, dtype=int)
    ig = np.asarray(i_gt, dtype=int)

    t_match = np.asarray(t_gt, dtype=float)[ig]
    return (
        t_match,
        np.asarray(p_est, dtype=float)[ie],
        np.asarray(q_est, dtype=float)[ie],
        t_match,
        np.asarray(p_gt, dtype=float)[ig],
        np.asarray(q_gt, dtype=float)[ig],
        ie,
    )


def _rot_z(theta: float) -> np.ndarray:
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _best_yaw(c_mat: np.ndarray) -> float:
    return float(math.atan2(float(c_mat[0, 1] - c_mat[1, 0]), float(c_mat[0, 0] + c_mat[1, 1])))


def _umeyama(
    data_xyz: np.ndarray,
    model_xyz: np.ndarray,
    known_scale: bool,
    yaw_only: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    data = np.asarray(data_xyz, dtype=float)
    model = np.asarray(model_xyz, dtype=float)
    if data.shape != model.shape or data.shape[0] < 2:
        raise ValueError("Alignment requires paired points with count >= 2.")

    mu_m = np.mean(model, axis=0)
    mu_d = np.mean(data, axis=0)
    m0 = model - mu_m
    d0 = data - mu_d

    n = float(data.shape[0])
    c_mat = (m0.T @ d0) / n
    sigma2 = float(np.mean(np.sum(d0**2, axis=1)))

    u, svals, vt = np.linalg.svd(c_mat)
    s_mat = np.eye(3, dtype=float)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0

    if yaw_only:
        r_fit = _rot_z(_best_yaw(n * c_mat.T))
    else:
        r_fit = u @ s_mat @ vt

    if known_scale:
        scale = 1.0
    else:
        if sigma2 < 1e-12:
            raise ValueError("Degenerate variance for Sim3 alignment.")
        scale = float(np.trace(np.diag(svals) @ s_mat) / sigma2)

    t_fit = mu_m - scale * (r_fit @ mu_d)
    return scale, r_fit, t_fit


def _solve_alignment(
    method: str,
    p_est: np.ndarray,
    q_est: np.ndarray,
    p_gt: np.ndarray,
    q_gt: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    m = str(method).lower()
    if m not in _VALID_ALIGN_MODES:
        raise ValueError(f"Invalid align_mode '{method}'. Expected one of: {sorted(_VALID_ALIGN_MODES)}")

    if m == "none":
        return 1.0, np.eye(3, dtype=float), np.zeros(3, dtype=float)

    if m == "sim3":
        return _umeyama(p_est, p_gt, known_scale=False, yaw_only=False)

    if m == "se3":
        _, r_fit, t_fit = _umeyama(p_est, p_gt, known_scale=True, yaw_only=False)
        return 1.0, r_fit, t_fit

    if m == "posyaw":
        _, r_fit, t_fit = _umeyama(p_est, p_gt, known_scale=True, yaw_only=True)
        return 1.0, r_fit, t_fit

    p0_est = np.asarray(p_est[0], dtype=float)
    p0_gt = np.asarray(p_gt[0], dtype=float)
    q0_est = np.asarray(q_est[0], dtype=float)
    q0_gt = np.asarray(q_gt[0], dtype=float)

    if m == "se3single":
        r_fit = R.from_quat(q0_gt).as_matrix() @ R.from_quat(q0_est).as_matrix().T
        t_fit = p0_gt - r_fit @ p0_est
        return 1.0, r_fit, t_fit

    if m == "posyawsingle":
        yaw_gt = float(R.from_quat(q0_gt).as_euler("zyx", degrees=False)[0])
        yaw_est = float(R.from_quat(q0_est).as_euler("zyx", degrees=False)[0])
        r_fit = _rot_z(yaw_gt - yaw_est)
        t_fit = p0_gt - r_fit @ p0_est
        return 1.0, r_fit, t_fit

    raise ValueError(f"Unsupported alignment mode: {method}")


def _apply_similarity(
    p_est: np.ndarray,
    q_est: np.ndarray,
    scale: float,
    r_fit: np.ndarray,
    t_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_new = float(scale) * (np.asarray(r_fit, dtype=float) @ np.asarray(p_est, dtype=float).T).T + np.asarray(t_fit, dtype=float)
    q_new = normalize_quat_array((R.from_matrix(np.asarray(r_fit, dtype=float)) * R.from_quat(np.asarray(q_est, dtype=float))).as_quat())
    return p_new, q_new


def _evaluate_pair(
    file_gt: Path,
    file_est: Path,
    align_mode: str,
    max_diff: float,
) -> dict:
    t_gt, p_gt, q_gt = _load_pose_file(file_gt)
    t_est, p_est, q_est = _load_pose_file(file_est)

    len_gt = float(np.sum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))) if p_gt.shape[0] > 1 else 0.0
    len_est = float(np.sum(np.linalg.norm(np.diff(p_est, axis=0), axis=1))) if p_est.shape[0] > 1 else 0.0
    ratio = len_est / (len_gt + 1e-12)

    t_est_m, p_est_m, q_est_m, t_gt_m, p_gt_m, q_gt_m, _ = _associate_est_gt(
        t_est=t_est,
        p_est=p_est,
        q_est=q_est,
        t_gt=t_gt,
        p_gt=p_gt,
        q_gt=q_gt,
        max_diff=float(max_diff),
        offset=0.0,
    )

    scale, r_fit, t_fit = _solve_alignment(
        method=align_mode,
        p_est=p_est_m,
        q_est=q_est_m,
        p_gt=p_gt_m,
        q_gt=q_gt_m,
    )
    p_est_aligned, q_est_aligned = _apply_similarity(
        p_est=p_est_m,
        q_est=q_est_m,
        scale=scale,
        r_fit=r_fit,
        t_fit=t_fit,
    )

    ape3 = compute_ape_evo_style(
        pos_ref=p_gt_m,
        quat_ref=q_gt_m,
        pos_est=p_est_aligned,
        quat_est=q_est_aligned,
        include_raw=True,
    )

    p_gt_xy, q_gt_xy = project_to_plane(p_gt_m, q_gt_m, "xy")
    p_est_xy, q_est_xy = project_to_plane(p_est_aligned, q_est_aligned, "xy")
    ape2 = compute_ape_evo_style(
        pos_ref=p_gt_xy,
        quat_ref=q_gt_xy,
        pos_est=p_est_xy,
        quat_est=q_est_xy,
        include_raw=True,
    )

    return {
        "matched": int(t_gt_m.size),
        "length_ratio": float(ratio),
        "length_gt": float(len_gt),
        "length_est": float(len_est),
        "gt_t": t_gt_m,
        "gt_pos": p_gt_m,
        "gt_quat": q_gt_m,
        "est_pos": p_est_aligned,
        "est_quat": q_est_aligned,
        "ate3_ori": dict(ape3["rotation_angle_deg"]),
        "ate3_pos": dict(ape3["translation_part"]),
        "ate2_ori": dict(ape2["rotation_angle_deg"]),
        "ate2_pos": dict(ape2["translation_part"]),
    }


def _compute_rpe_segments(
    gt_pos: np.ndarray,
    gt_quat: np.ndarray,
    est_pos: np.ndarray,
    est_quat: np.ndarray,
    segments_m: list[float],
) -> dict[float, dict[str, np.ndarray | dict[str, float] | int]]:
    out: dict[float, dict[str, np.ndarray | dict[str, float] | int]] = {}
    for seg in segments_m:
        tol_rel = min(1.0, 0.5 / float(seg)) if float(seg) > 0 else 0.1
        blk = compute_rpe_evo_style(
            pos_ref=gt_pos,
            quat_ref=gt_quat,
            pos_est=est_pos,
            quat_est=est_quat,
            delta=float(seg),
            delta_unit="m",
            rel_delta_tol=float(tol_rel),
            all_pairs=True,
            include_raw=True,
        )
        ori_vals = np.asarray(blk["_error_arrays"]["rotation_angle_deg"], dtype=float)
        pos_vals = np.asarray(blk["_error_arrays"]["translation_part"], dtype=float)
        out[float(seg)] = {
            "ori_values": ori_vals,
            "pos_values": pos_vals,
            "ori_stats": compute_error_statistics(ori_vals),
            "pos_stats": compute_error_statistics(pos_vals),
            "pair_count": int(blk.get("pair_count", int(pos_vals.size))),
        }
    return out


def run_format_converter(args: argparse.Namespace) -> int:
    src = Path(args.path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Input path not found: {src}")

    csv_files: list[Path] = []
    if src.is_file():
        if src.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got: {src}")
        csv_files = [src]
    else:
        csv_files = sorted(p for p in src.rglob("*.csv") if p.is_file())

    if len(csv_files) == 0:
        print("No .csv files found.")
        return 0

    for fpath in csv_files:
        t, pos, quat = _load_ov_eval_csv(fpath)
        out_path = fpath.with_suffix(".txt")
        if out_path.exists():
            print(f"[skip] output already exists: {out_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("# timestamp(s) tx ty tz qx qy qz qw\n")
            for i in range(t.size):
                f.write(
                    f"{t[i]:.5f} "
                    f"{pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f} "
                    f"{quat[i, 0]:.6f} {quat[i, 1]:.6f} {quat[i, 2]:.6f} {quat[i, 3]:.6f}\n"
                )
        print(f"[ok] {fpath} -> {out_path}")

    return 0


def run_error_singlerun(args: argparse.Namespace) -> int:
    eval_res = _evaluate_pair(
        file_gt=Path(args.file_gt).expanduser(),
        file_est=Path(args.file_est).expanduser(),
        align_mode=str(args.align_mode),
        max_diff=float(args.max_diff),
    )

    if eval_res["length_ratio"] > 1.1 or eval_res["length_ratio"] < 0.9:
        print(
            f"[WARN] Trajectory length ratio est/gt={eval_res['length_ratio']:.2f} "
            f"(est={eval_res['length_est']:.2f}m, gt={eval_res['length_gt']:.2f}m)"
        )

    ate3_ori = eval_res["ate3_ori"]
    ate3_pos = eval_res["ate3_pos"]

    print("======================================")
    print("Absolute Trajectory Error")
    print("======================================")
    print(f"rmse_ori = {_fmt(ate3_ori['rmse'])} | rmse_pos = {_fmt(ate3_pos['rmse'])}")
    print(f"mean_ori = {_fmt(ate3_ori['mean'])} | mean_pos = {_fmt(ate3_pos['mean'])}")
    print(f"min_ori  = {_fmt(ate3_ori['min'])} | min_pos  = {_fmt(ate3_pos['min'])}")
    print(f"max_ori  = {_fmt(ate3_ori['max'])} | max_pos  = {_fmt(ate3_pos['max'])}")
    print(f"std_ori  = {_fmt(ate3_ori['std'])} | std_pos  = {_fmt(ate3_pos['std'])}")

    segments = [8.0, 16.0, 24.0, 32.0, 40.0]
    rpe = _compute_rpe_segments(
        gt_pos=np.asarray(eval_res["gt_pos"], dtype=float),
        gt_quat=np.asarray(eval_res["gt_quat"], dtype=float),
        est_pos=np.asarray(eval_res["est_pos"], dtype=float),
        est_quat=np.asarray(eval_res["est_quat"], dtype=float),
        segments_m=segments,
    )

    print("======================================")
    print("Relative Pose Error")
    print("======================================")
    for seg in segments:
        blk = rpe[seg]
        ori_stats = blk["ori_stats"]
        pos_stats = blk["pos_stats"]
        n = int(blk["pair_count"])
        print(
            f"seg {int(seg)} - median_ori = {_fmt(ori_stats['median'])} "
            f"| median_pos = {_fmt(pos_stats['median'])} ({n} samples)"
        )

    print("======================================")
    print(f"Aligned pairs: {int(eval_res['matched'])}")
    if args.plot:
        print("[info] --plot is reserved in EPA compatibility mode. Use `epa_ape`/`epa_rpe` for full plotting.")
    return 0


def run_error_dataset(args: argparse.Namespace) -> int:
    file_gt = Path(args.file_gt).expanduser()
    alg_root = Path(args.folder_algorithms).expanduser()
    dataset_name = file_gt.stem

    algo_dirs = sorted([p for p in alg_root.iterdir() if p.is_dir()])
    if len(algo_dirs) == 0:
        raise ValueError(f"No algorithm folders found in: {alg_root}")

    t_gt, p_gt, _ = _load_pose_file(file_gt)
    gt_length = float(np.sum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))) if p_gt.shape[0] > 1 else 0.0
    print(f"[COMP]: {int(t_gt.size)} poses in {dataset_name} => length of {gt_length:.2f} meters")

    segments = [7.0, 14.0, 21.0, 28.0, 35.0]
    print("======================================")

    for algo_dir in algo_dirs:
        run_dir = algo_dir / dataset_name
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"[COMP]: {algo_dir.name} has no runs for dataset {dataset_name}")
            continue

        run_files = sorted([p for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".csv"}])
        if len(run_files) == 0:
            print(f"[COMP]: {algo_dir.name} has empty run folder for {dataset_name}")
            continue

        ate_ori_rmse: list[float] = []
        ate_pos_rmse: list[float] = []
        ate2_ori_rmse: list[float] = []
        ate2_pos_rmse: list[float] = []
        rpe_ori_vals: dict[float, list[float]] = {s: [] for s in segments}
        rpe_pos_vals: dict[float, list[float]] = {s: [] for s in segments}

        for run_file in run_files:
            ev = _evaluate_pair(
                file_gt=file_gt,
                file_est=run_file,
                align_mode=str(args.align_mode),
                max_diff=float(args.max_diff),
            )
            ate_ori_rmse.append(float(ev["ate3_ori"]["rmse"]))
            ate_pos_rmse.append(float(ev["ate3_pos"]["rmse"]))
            ate2_ori_rmse.append(float(ev["ate2_ori"]["rmse"]))
            ate2_pos_rmse.append(float(ev["ate2_pos"]["rmse"]))

            rpe = _compute_rpe_segments(
                gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                est_pos=np.asarray(ev["est_pos"], dtype=float),
                est_quat=np.asarray(ev["est_quat"], dtype=float),
                segments_m=segments,
            )
            for seg in segments:
                rpe_ori_vals[seg].extend(np.asarray(rpe[seg]["ori_values"], dtype=float).tolist())
                rpe_pos_vals[seg].extend(np.asarray(rpe[seg]["pos_values"], dtype=float).tolist())

        ate_ori = compute_error_statistics(np.asarray(ate_ori_rmse, dtype=float))
        ate_pos = compute_error_statistics(np.asarray(ate_pos_rmse, dtype=float))
        ate2_ori = compute_error_statistics(np.asarray(ate2_ori_rmse, dtype=float))
        ate2_pos = compute_error_statistics(np.asarray(ate2_pos_rmse, dtype=float))

        print(f"[COMP]: processing {algo_dir.name} algorithm")
        print(
            f"\tATE: mean_ori = {_fmt(ate_ori['mean'])} | mean_pos = {_fmt(ate_pos['mean'])} ({len(run_files)} runs)"
        )
        print(f"\tATE: std_ori  = {_fmt(ate_ori['std'], 5)} | std_pos  = {_fmt(ate_pos['std'], 5)}")
        print(
            f"\tATE 2D: mean_ori = {_fmt(ate2_ori['mean'])} | mean_pos = {_fmt(ate2_pos['mean'])} ({len(run_files)} runs)"
        )
        print(f"\tATE 2D: std_ori  = {_fmt(ate2_ori['std'], 5)} | std_pos  = {_fmt(ate2_pos['std'], 5)}")

        for seg in segments:
            o_stats = compute_error_statistics(np.asarray(rpe_ori_vals[seg], dtype=float))
            p_stats = compute_error_statistics(np.asarray(rpe_pos_vals[seg], dtype=float))
            n = len(rpe_pos_vals[seg])
            print(
                f"\tRPE: seg {int(seg)} - mean_ori = {_fmt(o_stats['mean'])} "
                f"| mean_pos = {_fmt(p_stats['mean'])} ({n} samples)"
            )

        print("\tNEES: n/a in EPA compatibility mode")
        print("======================================")

    if args.plot:
        print("[info] --plot is reserved in EPA compatibility mode.")
    return 0


def run_error_comparison(args: argparse.Namespace) -> int:
    gt_root = Path(args.folder_groundtruth).expanduser()
    alg_root = Path(args.folder_algorithms).expanduser()

    gt_files = sorted([p for p in gt_root.rglob("*.txt") if p.is_file()])
    algo_dirs = sorted([p for p in alg_root.iterdir() if p.is_dir()])
    if len(gt_files) == 0:
        raise ValueError(f"No groundtruth .txt files found in: {gt_root}")
    if len(algo_dirs) == 0:
        raise ValueError(f"No algorithm folders found in: {alg_root}")

    for gt in gt_files:
        t, p, _ = _load_pose_file(gt)
        length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if p.shape[0] > 1 else 0.0
        print(f"[COMP]: {int(t.size)} poses in {gt.name} => length of {length:.2f} meters")

    segments = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    ate_table: dict[str, dict[str, tuple[float, float]]] = {a.name: {} for a in algo_dirs}
    rpe_all: dict[str, dict[float, tuple[list[float], list[float]]]] = {
        a.name: {s: ([], []) for s in segments} for a in algo_dirs
    }

    print("======================================")
    for algo_dir in algo_dirs:
        print(f"[COMP]: processing {algo_dir.name} algorithm")
        dataset_dirs = {p.name: p for p in algo_dir.iterdir() if p.is_dir()}

        for gt in gt_files:
            ds = gt.stem
            if ds not in dataset_dirs:
                print(f"[COMP]: {algo_dir.name} has no runs for {ds}")
                continue

            run_files = sorted(
                [p for p in dataset_dirs[ds].iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".csv"}]
            )
            if len(run_files) == 0:
                continue

            print(f"[COMP]: processing {algo_dir.name} algorithm => {ds} dataset")
            ate_ori_rmse: list[float] = []
            ate_pos_rmse: list[float] = []

            ds_rpe_ori: dict[float, list[float]] = {s: [] for s in segments}
            ds_rpe_pos: dict[float, list[float]] = {s: [] for s in segments}

            for run_file in run_files:
                ev = _evaluate_pair(
                    file_gt=gt,
                    file_est=run_file,
                    align_mode=str(args.align_mode),
                    max_diff=float(args.max_diff),
                )
                ate_ori_rmse.append(float(ev["ate3_ori"]["rmse"]))
                ate_pos_rmse.append(float(ev["ate3_pos"]["rmse"]))

                rpe = _compute_rpe_segments(
                    gt_pos=np.asarray(ev["gt_pos"], dtype=float),
                    gt_quat=np.asarray(ev["gt_quat"], dtype=float),
                    est_pos=np.asarray(ev["est_pos"], dtype=float),
                    est_quat=np.asarray(ev["est_quat"], dtype=float),
                    segments_m=segments,
                )
                for seg in segments:
                    ori_vals = np.asarray(rpe[seg]["ori_values"], dtype=float).tolist()
                    pos_vals = np.asarray(rpe[seg]["pos_values"], dtype=float).tolist()
                    ds_rpe_ori[seg].extend(ori_vals)
                    ds_rpe_pos[seg].extend(pos_vals)
                    rpe_all[algo_dir.name][seg][0].extend(ori_vals)
                    rpe_all[algo_dir.name][seg][1].extend(pos_vals)

            ate_ori_stats = compute_error_statistics(np.asarray(ate_ori_rmse, dtype=float))
            ate_pos_stats = compute_error_statistics(np.asarray(ate_pos_rmse, dtype=float))
            ate_table[algo_dir.name][ds] = (float(ate_ori_stats["mean"]), float(ate_pos_stats["mean"]))

            print(
                f"\tATE: mean_ori = {_fmt(ate_ori_stats['mean'])} "
                f"| mean_pos = {_fmt(ate_pos_stats['mean'])} ({len(run_files)} runs)"
            )
            for seg in segments:
                o_stats = compute_error_statistics(np.asarray(ds_rpe_ori[seg], dtype=float))
                p_stats = compute_error_statistics(np.asarray(ds_rpe_pos[seg], dtype=float))
                print(
                    f"\tRPE: seg {int(seg)} - median_ori = {_fmt(o_stats['median'], 4)} "
                    f"| median_pos = {_fmt(p_stats['median'], 4)} ({len(ds_rpe_pos[seg])} samples)"
                )

    print("============================================")
    print("ATE LATEX TABLE")
    print("============================================")
    for gt in gt_files:
        name = gt.stem.replace("_", "\\_")
        print(f" & \\textbf{{{name}}}", end="")
    print(" & \\textbf{Average} \\\\hline")

    for algo in algo_dirs:
        name = algo.name.replace("_", "\\_")
        print(name, end="")
        sum_ori = 0.0
        sum_pos = 0.0
        cnt = 0
        for gt in gt_files:
            ds = gt.stem
            if ds not in ate_table[algo.name]:
                print(" & - / -", end="")
                continue
            o, p = ate_table[algo.name][ds]
            print(f" & {_fmt(o)} / {_fmt(p)}", end="")
            if np.isfinite(o) and np.isfinite(p):
                sum_ori += o
                sum_pos += p
                cnt += 1
        avg_ori = sum_ori / cnt if cnt > 0 else float("nan")
        avg_pos = sum_pos / cnt if cnt > 0 else float("nan")
        print(f" & {_fmt(avg_ori)} / {_fmt(avg_pos)} \\\\")
    print("============================================")

    print("============================================")
    print("RPE LATEX TABLE")
    print("============================================")
    for seg in segments:
        print(f" & \\textbf{{{int(seg)}m}}", end="")
    print(" \\\\hline")

    for algo in algo_dirs:
        name = algo.name.replace("_", "\\_")
        print(name, end="")
        for seg in segments:
            ori_vals = np.asarray(rpe_all[algo.name][seg][0], dtype=float)
            pos_vals = np.asarray(rpe_all[algo.name][seg][1], dtype=float)
            o_stats = compute_error_statistics(ori_vals)
            p_stats = compute_error_statistics(pos_vals)
            print(f" & {_fmt(o_stats['mean'])} / {_fmt(p_stats['mean'])}", end="")
        print(" \\\\")
    print("============================================")
    return 0


def run_plot_trajectories(args: argparse.Namespace) -> int:
    align_mode = str(args.align_mode).lower()
    if align_mode not in _VALID_ALIGN_MODES:
        raise ValueError(f"Invalid align_mode '{args.align_mode}'")

    traj_argv: list[str] = [
        "--format",
        "tum",
        "--sync",
        "--sync-max-diff",
        str(float(args.max_diff)),
        "--ref",
        "1",
        "--plot",
    ]

    if align_mode in {"se3", "se3single", "posyaw", "posyawsingle", "sim3"}:
        traj_argv.append("--align")
    if align_mode == "sim3":
        traj_argv.append("--correct-scale")
    if align_mode in {"posyaw", "posyawsingle"}:
        print("[info] align_mode posyaw* is approximated with SE3 alignment in EPA compatibility mode.")

    traj_argv.append(str(args.file_gt))
    traj_argv.extend([str(x) for x in args.est_files])

    parser = build_traj_parser()
    parsed = parser.parse_args(traj_argv)
    return int(run_traj(parsed))


def _build_format_converter_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible format_converter in EPA.")
    p.add_argument("path", help="CSV file or folder")
    return p


def _build_error_singlerun_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible error_singlerun in EPA.")
    p.add_argument("align_mode", help="posyaw|posyawsingle|se3|se3single|sim3|none")
    p.add_argument("file_gt", help="groundtruth trajectory")
    p.add_argument("file_est", help="estimated trajectory")
    p.add_argument("--max-diff", type=float, default=_DEFAULT_ASSOC_MAX_DIFF, help="timestamp association threshold")
    p.add_argument("--plot", action="store_true", help="reserved in compatibility mode")
    return p


def _build_error_dataset_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible error_dataset in EPA.")
    p.add_argument("align_mode", help="posyaw|posyawsingle|se3|se3single|sim3|none")
    p.add_argument("file_gt", help="groundtruth trajectory")
    p.add_argument("folder_algorithms", help="algorithm root folder")
    p.add_argument("--max-diff", type=float, default=_DEFAULT_ASSOC_MAX_DIFF, help="timestamp association threshold")
    p.add_argument("--plot", action="store_true", help="reserved in compatibility mode")
    return p


def _build_error_comparison_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible error_comparison in EPA.")
    p.add_argument("align_mode", help="posyaw|posyawsingle|se3|se3single|sim3|none")
    p.add_argument("folder_groundtruth", help="groundtruth root folder")
    p.add_argument("folder_algorithms", help="algorithm root folder")
    p.add_argument("--max-diff", type=float, default=_DEFAULT_ASSOC_MAX_DIFF, help="timestamp association threshold")
    return p


def _build_plot_trajectories_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OV-Eval compatible plot_trajectories in EPA.")
    p.add_argument("align_mode", help="posyaw|posyawsingle|se3|se3single|sim3|none")
    p.add_argument("file_gt", help="groundtruth trajectory")
    p.add_argument("est_files", nargs="+", help="estimated trajectories")
    p.add_argument("--max-diff", type=float, default=_DEFAULT_ASSOC_MAX_DIFF, help="timestamp association threshold")
    return p


def main_format_converter(argv: list[str] | None = None) -> int:
    return int(run_format_converter(_build_format_converter_parser().parse_args(argv)))


def main_error_singlerun(argv: list[str] | None = None) -> int:
    return int(run_error_singlerun(_build_error_singlerun_parser().parse_args(argv)))


def main_error_dataset(argv: list[str] | None = None) -> int:
    return int(run_error_dataset(_build_error_dataset_parser().parse_args(argv)))


def main_error_comparison(argv: list[str] | None = None) -> int:
    return int(run_error_comparison(_build_error_comparison_parser().parse_args(argv)))


def main_plot_trajectories(argv: list[str] | None = None) -> int:
    return int(run_plot_trajectories(_build_plot_trajectories_parser().parse_args(argv)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenVINS ov_eval compatibility layer for EPA.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("format_converter", help="Convert EuRoC CSV to ov_eval txt format")
    sub.add_parser("error_singlerun", help="Single-run ATE/RPE summary")
    sub.add_parser("error_dataset", help="Single-dataset multi-algorithm summary")
    sub.add_parser("error_comparison", help="Multi-dataset multi-algorithm summary")
    sub.add_parser("plot_trajectories", help="Trajectory overlay plotting")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)

    if args.command == "format_converter":
        return int(main_format_converter(rest))
    if args.command == "error_singlerun":
        return int(main_error_singlerun(rest))
    if args.command == "error_dataset":
        return int(main_error_dataset(rest))
    if args.command == "error_comparison":
        return int(main_error_comparison(rest))
    if args.command == "plot_trajectories":
        return int(main_plot_trajectories(rest))

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
