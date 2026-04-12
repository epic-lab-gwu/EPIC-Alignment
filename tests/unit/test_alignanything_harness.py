from pathlib import Path

import numpy as np

from vicon_ws.benchmark.alignanything_harness import (
    _write_summary_md,
    discover_cases,
    load_pose_table,
)


def test_load_pose_table_handles_csv_and_dedup(tmp_path: Path) -> None:
    path = tmp_path / "traj.txt"
    path.write_text(
        "\n".join(
            [
                "#timestamp, tx, ty, tz, qx, qy, qz, qw",
                "2.0, 1, 2, 3, 0, 0, 0, 2",
                "1.0, 0, 0, 0, 0, 0, 0, 1",
                "1.0, 9, 9, 9, 0, 0, 0, 1",
            ]
        ),
        encoding="utf-8",
    )

    data = load_pose_table(path)
    assert data.shape == (2, 8)
    assert np.all(np.diff(data[:, 0]) > 0.0)
    assert np.allclose(np.linalg.norm(data[:, 4:8], axis=1), 1.0)
    assert np.allclose(data[0, 1:4], [0.0, 0.0, 0.0])


def test_discover_cases_maps_euroc_gt(tmp_path: Path) -> None:
    align_root = tmp_path / "AlignAnything"
    gt_file = align_root / "GT" / "euroc_mav" / "MH_01_easy.txt"
    est_file = (
        align_root
        / "benchmark"
        / "euroc_mav"
        / "pose"
        / "rovio"
        / "MH_01_easy"
        / "rovio_poses.txt"
    )
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    est_file.parent.mkdir(parents=True, exist_ok=True)
    gt_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")
    est_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")

    cases, unresolved = discover_cases(align_root)
    assert unresolved == []
    assert len(cases) == 1
    case = cases[0]
    assert case.dataset == "euroc_mav"
    assert case.method == "rovio"
    assert case.sequence == "MH_01_easy"
    assert case.gt_path == gt_file
    assert case.est_path == est_file


def test_summary_markdown_contains_direction_arrows(tmp_path: Path) -> None:
    rows = [
        {
            "case": "demo_case",
            "status": "ok",
            "vicon_status": "ok",
            "evo_status": "ok",
            "vicon_ate_rmse_raw_m": 2.0,
            "vicon_ate_rmse_step3_m": 1.0,
            "evo_ape_raw_rmse_m": 2.5,
            "evo_ape_se3_rmse_m": 1.5,
            "vicon_improve_pct": 50.0,
            "evo_improve_pct": 40.0,
            "vicon_offset_est_s": 0.3,
            "evo_offset_s": 0.1,
            "vicon_matches_equivalent": 100.0,
            "evo_matches": 80.0,
            "vicon_xcorr_peak": 0.9,
            "vicon_xcorr_psr": 12.0,
            "vicon_omega_improve_pct": 70.0,
            "evo_sweep_evals": 88,
        }
    ]
    out = tmp_path / "summary.md"
    _write_summary_md(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "raw_rmse_m (v/e, ↓)" in text
    assert "improve_pct (v/e, ↑)" in text
    assert "vicon_xcorr_peak (↑)" in text
