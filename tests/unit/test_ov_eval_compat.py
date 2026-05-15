from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import epa
from epa.ov_eval_compat import _format_source_counts
from epa.ov_eval_compat import (
    _build_error_comparison_parser,
    _compute_time_rpe_1s,
    _drift_rate_percent,
    _evaluate_pair,
    _evaluate_pair_epa_step3,
    _evaluate_pair_ov_style,
    _format_source_details,
    run_error_comparison,
)


def test_package_version_matches_release() -> None:
    assert epa.__version__ == "0.1.8"


def test_format_source_counts_is_deterministic() -> None:
    assert _format_source_counts({"ov_eval_style": 1, "epa_step3": 2}) == "epa=2, ov_eval=1"
    assert _format_source_counts({}) == "epa=0, ov_eval=0"
    assert _format_source_counts({"unknown": 1}) == "epa=0, ov_eval=0, unknown=1"
    assert _format_source_details(["run1.txt:ov_eval_style", "run2.txt:unknown"]) == (
        "run1.txt:ov_eval_style, run2.txt:unknown"
    )
    assert _format_source_details([]) == "none"


def test_compute_time_rpe_1s_reports_translation_drift() -> None:
    t = np.arange(0.0, 4.1, 0.5)
    gt_pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    est_pos = np.column_stack([1.1 * t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    out = _compute_time_rpe_1s(
        gt_t=t,
        gt_pos=gt_pos,
        gt_quat=quat,
        est_pos=est_pos,
        est_quat=quat,
    )

    assert int(out["pair_count"]) == 7
    np.testing.assert_allclose(np.asarray(out["pos_values"], dtype=float), np.full(7, 0.1))
    np.testing.assert_allclose(float(out["pos_stats"]["rmse"]), 0.1)


def test_drift_rate_percent_normalizes_translation_by_segment_length() -> None:
    np.testing.assert_allclose(_drift_rate_percent([0.4, 0.8], 8.0), 7.5)
    assert np.isnan(_drift_rate_percent([], 8.0))


def _write_tum(path: Path, t: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> None:
    rows = [
        f"{float(ts):.9f} {float(p[0]):.9f} {float(p[1]):.9f} {float(p[2]):.9f} "
        f"{float(q[0]):.9f} {float(q[1]):.9f} {float(q[2]):.9f} {float(q[3]):.9f}"
        for ts, p, q in zip(t, pos, quat)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_evaluate_pair_epa_step3_reduces_rotation_error_for_body_frame_mismatch(tmp_path: Path) -> None:
    n = 400
    t = np.arange(n, dtype=float) * 0.05
    yaw = np.linspace(0.0, np.deg2rad(100.0), n)
    pos_gt = np.column_stack(
        [
            2.0 * np.cos(0.2 * t) + 0.4 * t,
            1.5 * np.sin(0.2 * t),
            0.2 * np.sin(0.05 * t),
        ]
    )
    quat_gt = R.from_euler("z", yaw).as_quat()

    r_ext = R.from_euler("xyz", [15.0, -20.0, 35.0], degrees=True).as_matrix()
    t_ext = np.array([0.4, -0.2, 0.1], dtype=float)

    r_gt = R.from_quat(quat_gt).as_matrix()
    pos_est = np.empty_like(pos_gt)
    for i in range(n):
        pos_est[i] = pos_gt[i] - r_gt[i] @ t_ext
    quat_est = (R.from_matrix(r_gt @ r_ext.T)).as_quat()

    gt_path = tmp_path / "gt.tum"
    est_path = tmp_path / "est.tum"
    _write_tum(gt_path, t, pos_gt, quat_gt)
    _write_tum(est_path, t, pos_est, quat_est)

    ov = _evaluate_pair_ov_style(gt_path, est_path, "se3", 0.02)
    epa = _evaluate_pair_epa_step3(gt_path, est_path, 0.02)

    assert float(ov["ate3_ori"]["rmse"]) > 10.0
    assert np.isfinite(float(epa["ate3_ori"]["rmse"]))
    assert np.isfinite(float(epa["ate3_pos"]["rmse"]))
    assert epa["eval_source"] == "epa_step3"


def test_error_comparison_parser_accepts_epa_advanced_args() -> None:
    parser = _build_error_comparison_parser()
    args = parser.parse_args(
        [
            "se3",
            "/gt",
            "/algorithms",
            "--epa-dt-resample",
            "0.01",
            "--epa-offset-min-match-ratio",
            "0.1",
            "--epa-downsample-hz",
            "20",
            "--epa-quat-interp",
            "slerp",
            "--epa-no-fallback",
            "--epa-verbose-fallback",
            "--epa-success-threshold-m",
            "12.5",
            "--epa-success-threshold-mode",
            "adaptive_knee",
            "--epa-success-threshold-min-m",
            "2",
            "--epa-success-threshold-max-m",
            "40",
            "--epa-success-threshold-trim-percentile",
            "90",
            "--epa-success-global-gate-m",
            "35",
            "--epa-success-global-gate-percentile",
            "10",
            "--epa-success-drift-rpe-1s-m",
            "3",
            "--epa-success-drift-ape-slope-mps",
            "1.5",
            "--epa-success-drift-ape-jump-m",
            "7",
        ]
    )

    assert args.epa_dt_resample == 0.01
    assert args.epa_offset_min_match_ratio == 0.1
    assert args.epa_downsample_hz == 20
    assert args.epa_quat_interp == "slerp"
    assert bool(args.epa_no_fallback)
    assert bool(args.epa_verbose_fallback)
    assert args.epa_success_threshold_m == 12.5
    assert args.epa_success_threshold_mode == "adaptive_knee"
    assert args.epa_success_threshold_min_m == 2
    assert args.epa_success_threshold_max_m == 40
    assert args.epa_success_threshold_trim_percentile == 90
    assert args.epa_success_global_gate_m == 35
    assert args.epa_success_global_gate_percentile == 10
    assert args.epa_success_drift_rpe_1s_m == 3
    assert args.epa_success_drift_ape_slope_mps == 1.5
    assert args.epa_success_drift_ape_jump_m == 7


def test_evaluate_pair_no_fallback_raises_for_impossible_epa_step3(tmp_path: Path) -> None:
    t = np.arange(20, dtype=float) * 0.05
    pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t.size, 1))

    gt_path = tmp_path / "gt.tum"
    est_path = tmp_path / "est.tum"
    _write_tum(gt_path, t, pos, quat)
    _write_tum(est_path, t, pos, quat)

    with pytest.raises(RuntimeError, match="EPA Step3 evaluation failed"):
        _evaluate_pair(
            file_gt=gt_path,
            file_est=est_path,
            align_mode="se3",
            max_diff=0.02,
            epa_no_fallback=True,
        )


def test_evaluate_pair_fallback_is_quiet_by_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    t = np.arange(20, dtype=float) * 0.05
    pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t.size, 1))

    gt_path = tmp_path / "gt.tum"
    est_path = tmp_path / "est.tum"
    _write_tum(gt_path, t, pos, quat)
    _write_tum(est_path, t, pos, quat)

    result = _evaluate_pair(
        file_gt=gt_path,
        file_est=est_path,
        align_mode="se3",
        max_diff=0.02,
    )

    captured = capsys.readouterr()
    assert "[warn] EPA Step3 evaluation failed" not in captured.out
    assert result["eval_source"] == "ov_eval_style"


def test_evaluate_pair_epa_step3_step1_fallback_is_quiet_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    n = 400
    t = np.arange(n, dtype=float) * 0.05
    yaw = np.linspace(0.0, np.deg2rad(100.0), n)
    pos_gt = np.column_stack(
        [
            2.0 * np.cos(0.2 * t) + 0.4 * t,
            1.5 * np.sin(0.2 * t),
            0.2 * np.sin(0.05 * t),
        ]
    )
    quat_gt = R.from_euler("z", yaw).as_quat()
    pos_est = pos_gt.copy()
    quat_est = quat_gt.copy()

    gt_path = tmp_path / "gt.tum"
    est_path = tmp_path / "est.tum"
    _write_tum(gt_path, t, pos_gt, quat_gt)
    _write_tum(est_path, t, pos_est, quat_est)

    result = _evaluate_pair(
        file_gt=gt_path,
        file_est=est_path,
        align_mode="se3",
        max_diff=0.02,
    )

    captured = capsys.readouterr()
    assert "--- STEP 1 FALLBACK ---" not in captured.out
    assert result["eval_source"] == "epa_step3"


def test_error_comparison_aggregates_mixed_sources_and_valid_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t = np.arange(5, dtype=float)
    gt_pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))
    gt_root = tmp_path / "gt"
    alg_root = tmp_path / "algorithms"
    run_dir = alg_root / "algo" / "seq"
    gt_root.mkdir()
    run_dir.mkdir(parents=True)
    _write_tum(gt_root / "seq.txt", t, gt_pos, quat)
    (run_dir / "run_epa.txt").write_text("", encoding="utf-8")
    (run_dir / "run_ov.txt").write_text("", encoding="utf-8")

    def fake_evaluate_pair(**kwargs):
        run_name = Path(kwargs["file_est"]).name
        if run_name == "run_epa.txt":
            est_pos = gt_pos.copy()
            source = "epa_step3"
            pos_rmse = 0.0
        else:
            est_pos = gt_pos.copy()
            est_pos[:, 1] = 50.0
            source = "ov_eval_style"
            pos_rmse = 50.0
        return {
            "ate3_ori": {"rmse": 0.0},
            "ate3_pos": {"rmse": pos_rmse},
            "eval_source": source,
            "gt_t": t,
            "gt_pos": gt_pos,
            "gt_quat": quat,
            "est_pos": est_pos,
            "est_quat": quat,
        }

    monkeypatch.setattr("epa.ov_eval_compat._evaluate_pair", fake_evaluate_pair)
    args = _build_error_comparison_parser().parse_args(
        [
            "se3",
            str(gt_root),
            str(alg_root),
            "--epa-success-threshold-mode",
            "fixed",
            "--epa-success-threshold-m",
            "10",
            "--epa-success-global-gate-m",
            "30",
        ]
    )

    assert run_error_comparison(args) == 0
    out = capsys.readouterr().out

    assert "eval_source: epa=1, ov_eval=1" in out
    assert "TOOL SOURCE: epa=1, ov_eval=1" in out
    assert "EVAL SOURCE NON-EPA RUNS: algo/seq/run_ov.txt:ov_eval_style" in out
    assert "DRIFT-VALID SUCCESS RATE LATEX TABLE (% PATH LENGTH)" in out
