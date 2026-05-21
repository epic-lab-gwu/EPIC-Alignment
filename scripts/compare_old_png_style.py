from __future__ import annotations

import argparse
import copy
import csv
import importlib
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R

from epa.benchmark.benchmark_harness import load_pose_table
from epa.core.calibration import (
    solve_extrinsic_rotation,
    solve_extrinsic_translation,
    solve_world_alignment,
)
from epa.core.pipeline_modular import (
    _search_direct_offset_from_matched_pairs,
)
from epa.core.time_alignment import (
    get_angular_velocity_norm,
    interpolate_quat_linear,
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    gt_path: Path
    est_path: Path
    evo_t_offset_s: float


def _load_case_offsets(path: Path) -> dict[str, float]:
    offsets: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            offsets[str(row["case"]).strip()] = float(row["t_offset"])
    return offsets


def _build_cases(offsets_csv: Path) -> list[CaseSpec]:
    offsets = _load_case_offsets(offsets_csv)
    base_gt = Path("/home/yifu/epa_data/AlignAnything/AlignAnything/GT/grand_tour/group1")
    base_est = Path("/home/yifu/epa_data/AlignAnything/AlignAnything/benchmark/grand_tour/pose/rovio")
    return [
        CaseSpec(
            case_id="grand_tour_2024-10-01-11-29-55_rovio",
            gt_path=base_gt / "2024-10-01-11-29-55.txt",
            est_path=base_est / "2024-10-01-11-29-55" / "rovio_poses.txt",
            evo_t_offset_s=float(offsets["grand_tour_2024-10-01-11-29-55_rovio"]),
        ),
        CaseSpec(
            case_id="grand_tour_2024-10-01-12-00-49_rovio",
            gt_path=base_gt / "2024-10-01-12-00-49.txt",
            est_path=base_est / "2024-10-01-12-00-49" / "rovio_poses.txt",
            evo_t_offset_s=float(offsets["grand_tour_2024-10-01-12-00-49_rovio"]),
        ),
    ]


def _build_evo_traj(data: np.ndarray):
    trajectory = importlib.import_module("evo.core.trajectory")
    PoseTrajectory3D = trajectory.PoseTrajectory3D

    xyz = data[:, 1:4]
    q_xyzw = data[:, 4:8]
    q_wxyz = np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])
    stamps = data[:, 0]
    return PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=q_wxyz, timestamps=stamps)


def _compute_evo_alignment(gt: np.ndarray, est: np.ndarray, t_offset_s: float, t_max_diff_s: float) -> dict[str, object]:
    sync = importlib.import_module("evo.core.sync")

    traj_ref = _build_evo_traj(gt)
    traj_est = _build_evo_traj(est)
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        copy.deepcopy(traj_ref),
        copy.deepcopy(traj_est),
        max_diff=float(t_max_diff_s),
        offset_2=float(t_offset_s),
        first_name="ref",
        snd_name="est",
    )
    traj_est_aligned = copy.deepcopy(traj_est_sync)
    r_align, t_align, _ = traj_est_aligned.align(
        traj_ref_sync,
        correct_scale=False,
        correct_only_scale=False,
        n=-1,
    )
    pred_sync = traj_est_aligned.positions_xyz
    rmse_sync = float(np.sqrt(np.mean(np.sum((pred_sync - traj_ref_sync.positions_xyz) ** 2, axis=1))))
    est_full_aligned = (r_align @ est[:, 1:4].T).T + t_align
    return {
        "pos_gt_full": gt[:, 1:4],
        "pos_est_aligned": est_full_aligned,
        "rmse_sync_m": rmse_sync,
        "matches": int(traj_ref_sync.num_poses),
        "t_offset_s": float(t_offset_s),
    }


def _compute_epa_global_paths(gt: np.ndarray, est: np.ndarray, t_max_diff_s: float) -> dict[str, object]:
    t_vicon = np.asarray(gt[:, 0], dtype=float)
    pos_vicon = np.asarray(gt[:, 1:4], dtype=float)
    quat_vicon = np.asarray(gt[:, 4:8], dtype=float)
    t_robot = np.asarray(est[:, 0], dtype=float)
    pos_robot = np.asarray(est[:, 1:4], dtype=float)
    quat_robot = np.asarray(est[:, 4:8], dtype=float)

    t_v_mid, om_v = get_angular_velocity_norm(t_vicon, quat_vicon)
    t_r_mid, om_r = get_angular_velocity_norm(t_robot, quat_robot)

    dt_resample = 0.001
    t_min = max(float(t_v_mid[0]), float(t_r_mid[0]))
    t_max = min(float(t_v_mid[-1]), float(t_r_mid[-1]))
    t_uniform = np.arange(t_min, t_max, dt_resample)
    sig_v = interp1d(t_v_mid, om_v, kind="linear")(t_uniform)
    sig_r = interp1d(t_r_mid, om_r, kind="linear")(t_uniform)
    sig_v -= np.mean(sig_v)
    sig_r -= np.mean(sig_r)
    corr = correlate(sig_r, sig_v, mode="full")
    lags = np.arange(-len(sig_v) + 1, len(sig_r))
    offsets_s = lags.astype(float) * dt_resample
    peak_idx = int(np.argmax(corr))
    step1_offset_s = float(offsets_s[peak_idx])

    t_robot_sync = t_robot - step1_offset_s
    pr_sync = interp1d(t_robot_sync, pos_robot, axis=0, fill_value="extrapolate")(t_vicon)
    qr_sync = interpolate_quat_linear(t_robot_sync, quat_robot, t_vicon)

    pos_vicon_solve = np.asarray(pos_vicon, dtype=float)
    quat_vicon_solve = np.asarray(quat_vicon, dtype=float)
    pr_solve = np.asarray(pr_sync, dtype=float)
    qr_solve = np.asarray(qr_sync, dtype=float)

    r_ext = solve_extrinsic_rotation(quat_vicon_solve, qr_solve)
    t_ext = solve_extrinsic_translation(pos_vicon_solve, quat_vicon_solve, pr_solve, qr_solve, r_ext)

    rr_mats = R.from_quat(qr_sync).as_matrix()
    pr_corrected = np.zeros_like(pr_sync)
    for i in range(len(pr_sync)):
        pr_corrected[i] = pr_sync[i] - (rr_mats[i] @ r_ext.T) @ t_ext

    rr_mats_solve = R.from_quat(qr_solve).as_matrix()
    pr_corrected_solve = np.zeros_like(pr_solve)
    for i in range(len(pr_solve)):
        pr_corrected_solve[i] = pr_solve[i] - (rr_mats_solve[i] @ r_ext.T) @ t_ext

    rw_step2, tw_step2 = solve_world_alignment(pr_corrected_solve, pos_vicon_solve)
    pred_step2_sync = (rw_step2 @ pr_corrected_solve.T).T + tw_step2
    rmse_step2 = float(np.sqrt(np.mean(np.sum((pred_step2_sync - pos_vicon_solve) ** 2, axis=1))))
    pred_step2_full = (rw_step2 @ pr_corrected.T).T + tw_step2

    direct_search = _search_direct_offset_from_matched_pairs(
        t_ref=t_vicon,
        pos_ref=pos_vicon,
        t_est=t_robot,
        pos_est=pos_robot,
        initial_offset_s=float(step1_offset_s),
        max_diff_s=float(t_max_diff_s),
        min_match_ratio=0.3,
    )
    if direct_search is None:
        raise RuntimeError("direct offset search failed")

    direct_offset_s = float(direct_search["offset_s"])
    ids_ref = np.asarray(direct_search["ids_ref"], dtype=int)
    ids_est = np.asarray(direct_search["ids_est"], dtype=int)
    rw_direct, tw_direct = solve_world_alignment(pos_robot[ids_est], pos_vicon[ids_ref])
    pred_direct_sync = (rw_direct @ pos_robot[ids_est].T).T + tw_direct
    rmse_direct = float(np.sqrt(np.mean(np.sum((pred_direct_sync - pos_vicon[ids_ref]) ** 2, axis=1))))
    pred_direct_full = (rw_direct @ pr_sync.T).T + tw_direct

    selected_mode = "direct_global_se3" if rmse_direct < rmse_step2 else "step2_then_global_se3"
    pred_selected_full = pred_direct_full if selected_mode == "direct_global_se3" else pred_step2_full
    rmse_selected = rmse_direct if selected_mode == "direct_global_se3" else rmse_step2
    return {
        "pos_gt_full": pos_vicon,
        "step1_offset_s": step1_offset_s,
        "direct_offset_s": direct_offset_s,
        "calibrated": {
            "pos_est_aligned": pred_step2_full,
            "rmse_m": rmse_step2,
        },
        "direct": {
            "pos_est_aligned": pred_direct_full,
            "rmse_m": rmse_direct,
            "matches": int(ids_ref.size),
        },
        "selected": {
            "mode": selected_mode,
            "pos_est_aligned": pred_selected_full,
            "rmse_m": rmse_selected,
        },
    }


def _plot_old_style(ax, pos_ref: np.ndarray, pos_est: np.ndarray, title: str) -> None:
    ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], color="green", linewidth=2.0, label="GT (ref)")
    ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2], color="red", linestyle="--", linewidth=2.0, label="EST aligned")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right")
    ax.view_init(elev=28, azim=-62)


def _add_old_png(ax, image_path: Path, title: str) -> None:
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Old-PNG Style Comparison",
        "",
        "| case | evo_se3_rmse_m | evo_se3_toffset_rmse_m | epa_calibrated_rmse_m | epa_direct_rmse_m | epa_selected_mode | epa_selected_rmse_m |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {evo_se3_rmse_m:.6f} | {evo_se3_toffset_rmse_m:.6f} | {epa_calibrated_rmse_m:.6f} | {epa_direct_rmse_m:.6f} | {epa_selected_mode} | {epa_selected_rmse_m:.6f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recreate old PNG style and compare evo vs epa on case2/case3.")
    parser.add_argument(
        "--old-root",
        default="/tmp/evo_failed_cases_20260421",
        help="Directory containing old PNGs and case offset CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/old_png_style_compare",
        help="Output directory for recreated comparison figures.",
    )
    parser.add_argument(
        "--t-max-diff",
        type=float,
        default=0.02,
        help="Max timestamp difference used in evo/EPA matching.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    old_root = Path(args.old_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _build_cases(old_root / "cases_offsets.csv")
    summary_rows: list[dict[str, object]] = []
    for case in cases:
        gt = load_pose_table(case.gt_path)
        est = load_pose_table(case.est_path)

        evo_se3 = _compute_evo_alignment(gt, est, t_offset_s=0.0, t_max_diff_s=float(args.t_max_diff))
        evo_se3_toffset = _compute_evo_alignment(
            gt,
            est,
            t_offset_s=float(case.evo_t_offset_s),
            t_max_diff_s=float(args.t_max_diff),
        )
        epa_paths = _compute_epa_global_paths(gt, est, t_max_diff_s=float(args.t_max_diff))

        fig = plt.figure(figsize=(22, 13))
        ax_old_se3 = fig.add_subplot(2, 3, 1)
        _add_old_png(ax_old_se3, old_root / "plots" / f"{case.case_id}__se3.png", "Old PNG: se3")

        ax_evo_se3 = fig.add_subplot(2, 3, 2, projection="3d")
        _plot_old_style(
            ax_evo_se3,
            np.asarray(evo_se3["pos_gt_full"], dtype=float),
            np.asarray(evo_se3["pos_est_aligned"], dtype=float),
            f"Recreated evo se3\nrmse={float(evo_se3['rmse_sync_m']):.3f}m, t_offset=0.000s",
        )

        ax_epa_cal = fig.add_subplot(2, 3, 3, projection="3d")
        _plot_old_style(
            ax_epa_cal,
            np.asarray(epa_paths["pos_gt_full"], dtype=float),
            np.asarray(epa_paths["calibrated"]["pos_est_aligned"], dtype=float),
            (
                "EPA calibrated global\n"
                f"rmse={float(epa_paths['calibrated']['rmse_m']):.3f}m, "
                f"step1_offset={float(epa_paths['step1_offset_s']):.3f}s"
            ),
        )

        ax_old_toff = fig.add_subplot(2, 3, 4)
        _add_old_png(ax_old_toff, old_root / "plots" / f"{case.case_id}__se3_toffset.png", "Old PNG: se3+t_offset")

        ax_evo_toff = fig.add_subplot(2, 3, 5, projection="3d")
        _plot_old_style(
            ax_evo_toff,
            np.asarray(evo_se3_toffset["pos_gt_full"], dtype=float),
            np.asarray(evo_se3_toffset["pos_est_aligned"], dtype=float),
            (
                "Recreated evo se3+t_offset\n"
                f"rmse={float(evo_se3_toffset['rmse_sync_m']):.3f}m, "
                f"t_offset={float(case.evo_t_offset_s):.3f}s"
            ),
        )

        ax_epa_direct = fig.add_subplot(2, 3, 6, projection="3d")
        _plot_old_style(
            ax_epa_direct,
            np.asarray(epa_paths["pos_gt_full"], dtype=float),
            np.asarray(epa_paths["direct"]["pos_est_aligned"], dtype=float),
            (
                "EPA direct global / auto\n"
                f"direct_rmse={float(epa_paths['direct']['rmse_m']):.3f}m, "
                f"selected={epa_paths['selected']['mode']}, "
                f"selected_rmse={float(epa_paths['selected']['rmse_m']):.3f}m"
            ),
        )

        fig.suptitle(case.case_id, fontsize=18)
        fig.tight_layout()
        out_path = out_dir / f"{case.case_id}__compare.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        summary_rows.append(
            {
                "case_id": case.case_id,
                "evo_se3_rmse_m": float(evo_se3["rmse_sync_m"]),
                "evo_se3_toffset_rmse_m": float(evo_se3_toffset["rmse_sync_m"]),
                "epa_calibrated_rmse_m": float(epa_paths["calibrated"]["rmse_m"]),
                "epa_direct_rmse_m": float(epa_paths["direct"]["rmse_m"]),
                "epa_selected_mode": str(epa_paths["selected"]["mode"]),
                "epa_selected_rmse_m": float(epa_paths["selected"]["rmse_m"]),
                "epa_step1_offset_s": float(epa_paths["step1_offset_s"]),
                "epa_direct_offset_s": float(epa_paths["direct_offset_s"]),
                "evo_t_offset_s": float(case.evo_t_offset_s),
            }
        )

    _write_summary_csv(out_dir / "summary.csv", summary_rows)
    _write_summary_md(out_dir / "summary.md", summary_rows)
    print(f"Saved comparison figures to: {out_dir}")
    for row in summary_rows:
        print(
            f"{row['case_id']}: evo_se3={row['evo_se3_rmse_m']:.6f}, "
            f"evo_se3_toffset={row['evo_se3_toffset_rmse_m']:.6f}, "
            f"epa_calibrated={row['epa_calibrated_rmse_m']:.6f}, "
            f"epa_direct={row['epa_direct_rmse_m']:.6f}, "
            f"epa_selected={row['epa_selected_mode']}:{row['epa_selected_rmse_m']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
