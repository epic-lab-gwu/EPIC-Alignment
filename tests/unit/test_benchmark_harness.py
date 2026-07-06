from pathlib import Path

import numpy as np
import pytest

from epa.benchmark.benchmark_harness import (
    BenchmarkCase,
    _run_benchmark_case,
    _resolve_jobs,
    _write_summary_html,
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


def test_discover_cases_without_pose_directory(tmp_path: Path) -> None:
    align_root = tmp_path / "cases"
    gt_file = align_root / "GT" / "custom" / "seq_01.txt"
    est_file = align_root / "benchmark" / "custom" / "orbslam3" / "seq_01" / "trajectory.txt"
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    est_file.parent.mkdir(parents=True, exist_ok=True)
    gt_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")
    est_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")

    cases, unresolved = discover_cases(align_root)

    assert unresolved == []
    assert len(cases) == 1
    assert cases[0].dataset == "custom"
    assert cases[0].method == "orbslam3"
    assert cases[0].sequence == "seq_01"


def test_discover_cases_system_with_sequence_files(tmp_path: Path) -> None:
    align_root = tmp_path / "cases"
    gt_file = align_root / "GT" / "custom" / "seq_02.tum"
    est_file = align_root / "benchmark" / "custom" / "vins" / "seq_02_poses.txt"
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    est_file.parent.mkdir(parents=True, exist_ok=True)
    gt_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")
    est_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")

    cases, unresolved = discover_cases(align_root)

    assert unresolved == []
    assert len(cases) == 1
    assert cases[0].method == "vins"
    assert cases[0].sequence == "seq_02"
    assert cases[0].gt_path == gt_file


def test_discover_cases_multiple_sequences_under_one_system(tmp_path: Path) -> None:
    align_root = tmp_path / "cases"
    gt_root = align_root / "GT" / "custom"
    system_root = align_root / "benchmark" / "custom" / "vins"
    gt_root.mkdir(parents=True, exist_ok=True)
    system_root.mkdir(parents=True, exist_ok=True)

    for seq in ["seq_01", "seq_02"]:
        (gt_root / f"{seq}.txt").write_text(
            "1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n",
            encoding="utf-8",
        )
        (system_root / f"{seq}_poses.txt").write_text(
            "1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n",
            encoding="utf-8",
        )

    cases, unresolved = discover_cases(align_root)

    assert unresolved == []
    assert len(cases) == 2
    assert {case.method for case in cases} == {"vins"}
    assert {case.sequence for case in cases} == {"seq_01", "seq_02"}


def test_discover_cases_multiple_sequence_dirs_under_one_system(tmp_path: Path) -> None:
    align_root = tmp_path / "cases"
    gt_root = align_root / "GT" / "custom"
    system_root = align_root / "benchmark" / "custom" / "orbslam3"
    gt_root.mkdir(parents=True, exist_ok=True)

    for seq in ["room_a", "room_b"]:
        (gt_root / f"{seq}.txt").write_text(
            "1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n",
            encoding="utf-8",
        )
        est_file = system_root / seq / "trajectory.txt"
        est_file.parent.mkdir(parents=True, exist_ok=True)
        est_file.write_text(
            "1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n",
            encoding="utf-8",
        )

    cases, unresolved = discover_cases(align_root)

    assert unresolved == []
    assert len(cases) == 2
    assert {case.method for case in cases} == {"orbslam3"}
    assert {case.sequence for case in cases} == {"room_a", "room_b"}


def test_discover_cases_disambiguates_duplicate_case_ids(tmp_path: Path) -> None:
    align_root = tmp_path / "cases"
    for subset in ["add", "add1"]:
        gt_file = align_root / "GT" / "lamaria" / "add1" / "sequence_1_19.txt"
        est_file = (
            align_root
            / "benchmark"
            / "lamaria"
            / subset
            / "pose"
            / "svo_mono"
            / "sequence_1_19"
            / "svo_poses.txt"
        )
        gt_file.parent.mkdir(parents=True, exist_ok=True)
        est_file.parent.mkdir(parents=True, exist_ok=True)
        gt_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")
        est_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")

    cases, unresolved = discover_cases(align_root)

    assert unresolved == []
    assert len(cases) == 2
    assert {case.case_id for case in cases} == {
        "lamaria_add_sequence_1_19_svo_mono",
        "lamaria_add1_sequence_1_19_svo_mono",
    }


def test_discover_cases_ignores_non_trajectory_csv_and_tum_files(tmp_path: Path) -> None:
    align_root = tmp_path / "cases"
    gt_file = align_root / "GT" / "custom" / "seq_01.txt"
    est_file = align_root / "benchmark" / "custom" / "vins" / "seq_01.tum"
    summary_file = align_root / "benchmark" / "custom" / "vins" / "summary.csv"
    bad_tum_file = align_root / "benchmark" / "custom" / "vins" / "notes.tum"
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    est_file.parent.mkdir(parents=True, exist_ok=True)
    gt_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")
    est_file.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")
    summary_file.write_text("case,status\nseq_01,ok\n", encoding="utf-8")
    bad_tum_file.write_text("not a trajectory\n", encoding="utf-8")

    cases, unresolved = discover_cases(align_root)

    assert unresolved == []
    assert len(cases) == 1
    assert cases[0].est_path == est_file


def test_summary_markdown_contains_direction_arrows(tmp_path: Path) -> None:
    rows = [
        {
            "case": "demo_case",
            "status": "ok",
            "epa_status": "ok",
            "evo_status": "ok",
            "epa_ate_rmse_raw_m": 2.0,
            "epa_ate_rmse_step3_m": 1.0,
            "evo_ape_raw_rmse_m": 2.5,
            "evo_ape_se3_rmse_m": 1.5,
            "epa_improve_pct": 50.0,
            "evo_improve_pct": 40.0,
            "epa_offset_est_s": 0.3,
            "evo_offset_s": 0.1,
            "epa_matches_equivalent": 100.0,
            "evo_matches": 80.0,
            "epa_xcorr_peak": 0.9,
            "epa_xcorr_psr": 12.0,
            "epa_omega_improve_pct": 70.0,
            "evo_sweep_evals": 88,
            "epa_sr_distance": 0.5,
            "epa_sr_time": 0.6,
            "epa_case_status": "globally_unstable",
            "epa_global_gate_failed": "True",
            "epa_global_gate_value_m": 45.0,
            "epa_valid_distance_m": 12.0,
            "epa_step3_alignment_mode": "robust_trimmed",
            "epa_step3_rejection_ratio": 0.25,
        }
    ]
    out = tmp_path / "summary.md"
    _write_summary_md(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "raw_rmse_m (↓)" in text
    assert "improve_pct (↑)" in text
    assert "Valid Segment Status" in text
    assert "globally_unstable" in text
    assert "robust_trimmed" in text
    assert "epa_xcorr_peak (↑)" in text


def test_summary_html_links_interactive_report_and_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run" / "epa_runs" / "case_a"
    plot = run_dir / "plots" / "step3_alignment_map.png"
    plot.parent.mkdir(parents=True)
    plot.write_text("png", encoding="utf-8")
    (run_dir / "interactive_report.html").write_text("<html></html>", encoding="utf-8")
    rows = [
        {
            "case": "case_a",
            "dataset": "demo",
            "method": "epa_sim3",
            "status": "ok",
            "epa_case_status": "globally_unstable",
            "epa_sr_distance_raw": 0.9,
            "epa_sr_distance": 0.42,
            "epa_sr_time_raw": 0.91,
            "epa_sr_time": 0.5,
            "epa_sr_reliability_status": "failed",
            "epa_sr_warning_explanation": "Sim3 may be masking a real trajectory failure.",
            "epa_sim3_may_mask_failure": "True",
            "epa_global_gate_failed": "True",
            "epa_global_gate_value_m": 50.0,
            "epa_valid_distance_m": 4.2,
            "epa_step3_alignment_mode": "robust_trimmed",
            "epa_step3_rejection_ratio": 0.33,
            "epa_step3_stable_solve_ratio": 0.42,
            "epa_sim3_scale": 0.01,
            "epa_sim3_reliable": "False",
            "epa_sim3_warning": "Sim3 scale warning",
            "epa_orientation_unstable": "True",
            "epa_orientation_ape_rmse_deg": 80.0,
            "epa_orientation_rpe_rmse_deg": 35.0,
            "epa_orientation_rpe_time_1s_rmse_deg": 32.0,
            "epa_rpe_time_1s_trans_rmse_m": 1.2,
            "epa_rpe_time_1s_rot_rmse_deg": 31.0,
            "epa_orientation_warning": "rotation unstable",
            "epa_ate_rmse_step3_m": 3.0,
            "epa_case_diagnosis_primary": "trajectory_jump",
            "epa_case_diagnosis_summary": "trajectory_jump; global_gate_too_large",
            "epa_case_diagnosis_tags": "trajectory_jump,global_gate_too_large",
            "epa_run_dir": str(run_dir),
        }
    ]

    out = tmp_path / "run" / "summary.html"
    _write_summary_html(rows, out)
    text = out.read_text(encoding="utf-8")

    assert "globally_unstable" in text
    assert "robust_trimmed" in text
    assert "reports/case_a__epa_sim3.html" in text
    assert "caseMarker" in text
    assert "caseLink" in text
    assert "showMarked" in text
    assert "showAll" in text
    assert "localStorage" in text
    assert "Sim3 audit" in text
    assert "SR reliability" in text
    assert "1s RPE trans" in text
    assert "valid distance" in text
    assert "global gate" in text
    assert "42.0%" in text
    assert "raw / local / gated" not in text
    assert "90.0% / 42.0% / 0.0%" not in text
    assert "Sim3 may be masking a real trajectory failure" in text
    assert "1.200 m" in text
    assert 'data-case="case_a"' in text
    assert "Trajectory has jump or divergence" in text
    assert "Sim3 is not reliable" in text
    assert "openImageViewer" not in text


def test_summary_html_hides_sim3_audit_for_non_sim3_mode(tmp_path: Path) -> None:
    out = tmp_path / "summary.html"
    _write_summary_html(
        [
            {
                "case": "case_a",
                "dataset": "demo",
                "method": "epa_se3",
                "status": "ok",
                "epa_case_status": "valid_segment",
                "epa_sr_distance": 1.0,
                "epa_sr_time": 1.0,
                "epa_ate_rmse_step3_m": 0.1,
            }
        ],
        out,
    )
    text = out.read_text(encoding="utf-8")
    assert "Sim3 audit" not in text
    assert "(epa_se3)" in text
    assert "raw / local / gated" not in text


def test_benchmark_case_does_not_run_evo_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    align_root = tmp_path / "cases"
    gt = align_root / "GT" / "demo" / "seq.txt"
    est = align_root / "benchmark" / "demo" / "sys" / "seq_poses.txt"
    gt.parent.mkdir(parents=True, exist_ok=True)
    est.parent.mkdir(parents=True, exist_ok=True)
    gt.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")
    est.write_text("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n", encoding="utf-8")

    def fake_run_epa_case(*args, **kwargs):
        assert kwargs["output_root"] == tmp_path / "run" / "epa_runs"
        assert args[0].case_id == "demo_seq_sys"
        return {
            "status": "ok",
            "run_dir": str(tmp_path / "run" / "epa_runs" / "run_demo"),
            "offset_est_s": 0.0,
            "ate_rmse_raw_m": 2.0,
            "ate_rmse_step3_m": 1.0,
            "improve_raw_to_step3_pct": 50.0,
            "sr_distance": 0.75,
            "sr_time": 0.8,
            "case_status": "valid_segment",
            "global_gate_failed": False,
            "global_gate_value_m": 0.2,
            "global_gate_m": 30.0,
            "valid_distance_m": 3.0,
            "total_distance_m": 4.0,
            "step3_alignment_mode": "standard",
            "step3_inlier_count": 20.0,
            "step3_rejected_count": 0.0,
            "step3_rejection_ratio": 0.0,
            "step3_stable_segment_used": 1.0,
            "step3_stable_solve_ratio": 0.5,
            "orientation_unstable": "True",
            "orientation_ape_rmse_deg": 80.0,
            "orientation_rpe_rmse_deg": 35.0,
            "orientation_rpe_time_1s_rmse_deg": 32.0,
            "orientation_warning": "rotation unstable",
        }

    def fake_run_evo_case(*args, **kwargs):
        raise AssertionError("evo should not run unless --with-evo is set")

    monkeypatch.setattr("epa.benchmark.benchmark_harness._run_epa_case", fake_run_epa_case)
    monkeypatch.setattr("epa.benchmark.benchmark_harness._run_evo_case", fake_run_evo_case)

    row = _run_benchmark_case(
        BenchmarkCase("demo_seq_sys", "demo", "sys", "seq", gt, est),
        align_root=align_root,
        prepared_dir=tmp_path / "run" / "prepared_tum",
        logs_dir=tmp_path / "run" / "logs",
        case_json_dir=tmp_path / "run" / "cases",
        repo_root=tmp_path,
        python_bin=Path("/usr/bin/python"),
        epa_src=tmp_path / "src",
        dt_resample=0.001,
        quat_interp="linear",
        downsample_hz=100.0,
        no_downsample=False,
        mplconfig_root=tmp_path / "run" / ".mplconfig",
        output_root=tmp_path / "run" / "epa_runs",
        evo_repo=tmp_path / "evo",
        with_evo=False,
        t_max_diff=0.02,
        offset_min=-1.0,
        offset_max=1.0,
        offset_coarse_step=0.5,
        offset_refine_window=0.5,
        offset_refine_step=0.1,
        min_match_ratio=0.05,
    )

    assert row["status"] == "ok"
    assert row["epa_status"] == "ok"
    assert row["evo_status"] == "not_run"
    assert row["epa_case_status"] == "valid_segment"
    assert row["epa_sr_distance"] == 0.75
    assert row["epa_step3_alignment_mode"] == "standard"
    assert row["epa_orientation_unstable"] == "True"
    assert row["epa_orientation_ape_rmse_deg"] == 80.0
    assert row["epa_step3_stable_segment_used"] == 1.0
    assert row["epa_step3_stable_solve_ratio"] == 0.5


def test_discover_cases_missing_root_error_has_examples(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing_alignanything"

    with pytest.raises(FileNotFoundError) as exc_info:
        discover_cases(missing_root)

    msg = str(exc_info.value)
    assert "Invalid --cases-root" in msg
    assert "EPA_DATA_ROOT" in msg
    assert "EPA_CASES_ROOT" in msg
    assert "epa_bench " in msg
    assert "benchmark_cases" in msg


def test_benchmark_parser_accepts_positional_cases_root_and_jobs() -> None:
    parser = __import__("epa.benchmark.benchmark_harness", fromlist=["build_parser"]).build_parser()
    args = parser.parse_args(["/tmp/cases_root", "--jobs", "4"])
    assert args.cases_root_pos == "/tmp/cases_root"
    assert args.jobs == "4"


def test_resolve_jobs_auto_is_bounded_by_case_count() -> None:
    assert _resolve_jobs("auto", 0) == 1
    assert _resolve_jobs("auto", 2) <= 2
    assert _resolve_jobs("1", 100) == 1
