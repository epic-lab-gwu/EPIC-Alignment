from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import epa
from epa.ov_eval_compat import _format_source_counts
from epa.ov_eval_compat import (
    _build_error_dataset_parser,
    _build_error_comparison_parser,
    _compute_rpe_segments,
    _compute_time_rpe_1s,
    _compute_valid_rpe_segments,
    _drift_rate_percent,
    _evaluate_pair,
    _evaluate_pair_epa_step3,
    _evaluate_pair_ov_style,
    _format_source_details,
    main,
    run_error_comparison,
    run_error_dataset,
)


def test_package_version_matches_release() -> None:
    assert epa.__version__ == "0.1.15"


def test_format_source_counts_is_deterministic() -> None:
    assert _format_source_counts({"epa_eval_align": 1, "epa_step3": 2, "failed": 3}) == (
        "epa_step3=2, epa_eval=1, failed=3"
    )
    assert _format_source_counts({}) == "epa_step3=0, epa_eval=0, failed=0"
    assert _format_source_counts({"unknown": 1}) == "epa_step3=0, epa_eval=0, failed=0, unknown=1"
    assert _format_source_details(["run1.txt:epa_eval_align", "run2.txt:unknown"]) == (
        "run1.txt:epa_eval_align, run2.txt:unknown"
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


def test_ov_eval_rpe_segments_use_fixed_half_meter_distance_window() -> None:
    t = np.arange(11, dtype=float)
    gt_pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    est_pos = gt_pos.copy()
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))

    out = _compute_rpe_segments(
        gt_pos=gt_pos,
        gt_quat=quat,
        est_pos=est_pos,
        est_quat=quat,
        segments_m=[8.0, 40.0],
    )

    assert int(out[8.0]["pair_count"]) == 3
    np.testing.assert_array_equal(
        np.asarray(out[8.0]["pair_ids"], dtype=int), np.array([[0, 8], [1, 9], [2, 10]])
    )
    assert float(out[8.0]["ori_stats"]["median"]) == 0.0
    assert float(out[8.0]["pos_stats"]["median"]) == 0.0
    assert int(out[40.0]["pair_count"]) == 0
    assert float(out[40.0]["ori_stats"]["median"]) == 0.0
    assert float(out[40.0]["pos_stats"]["median"]) == 0.0


def test_valid_rpe_segments_filter_by_valid_edge_mask() -> None:
    t = np.arange(11, dtype=float)
    gt_pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    est_pos = gt_pos.copy()
    est_pos[6:, 1] = 100.0
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))
    valid_segment_mask = np.array([True, True, True, True, True, False, False, False, False, False])

    full = _compute_rpe_segments(
        gt_pos=gt_pos,
        gt_quat=quat,
        est_pos=est_pos,
        est_quat=quat,
        segments_m=[4.0, 8.0],
    )
    valid = _compute_valid_rpe_segments(
        gt_pos=gt_pos,
        gt_quat=quat,
        est_pos=est_pos,
        est_quat=quat,
        valid_segment_mask=valid_segment_mask,
        segments_m=[4.0, 8.0],
    )

    assert int(full[4.0]["pair_count"]) == 7
    assert int(valid[4.0]["pair_count"]) == 2
    np.testing.assert_array_equal(
        np.asarray(valid[4.0]["pair_ids"], dtype=int), np.array([[0, 4], [1, 5]])
    )
    assert int(valid[8.0]["pair_count"]) == 0
    np.testing.assert_allclose(float(valid[4.0]["pos_stats"]["rmse"]), 0.0)


def _write_tum(path: Path, t: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> None:
    rows = [
        f"{float(ts):.9f} {float(p[0]):.9f} {float(p[1]):.9f} {float(p[2]):.9f} "
        f"{float(q[0]):.9f} {float(q[1]):.9f} {float(q[2]):.9f} {float(q[3]):.9f}"
        for ts, p, q in zip(t, pos, quat)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_evaluate_pair_epa_step3_reduces_rotation_error_for_body_frame_mismatch(
    tmp_path: Path,
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


def test_evaluate_pair_epa_step3_uses_sparse_timeline_for_low_rate_estimates(
    tmp_path: Path,
) -> None:
    t_gt = np.arange(0.0, 10.001, 0.01)
    pos_gt = np.column_stack(
        [
            0.4 * t_gt,
            np.sin(0.2 * t_gt),
            0.1 * np.cos(0.15 * t_gt),
        ]
    )
    quat_gt = R.from_euler("zyx", np.column_stack([0.5 * t_gt, 0.2 * t_gt, 0.3 * t_gt])).as_quat()

    sparse_ids = np.arange(0, t_gt.size, 100, dtype=int)
    t_est = t_gt[sparse_ids]
    pos_est = pos_gt[sparse_ids]
    quat_est = quat_gt[sparse_ids]

    gt_path = tmp_path / "gt_dense.tum"
    est_path = tmp_path / "est_sparse.tum"
    _write_tum(gt_path, t_gt, pos_gt, quat_gt)
    _write_tum(est_path, t_est, pos_est, quat_est)

    result = _evaluate_pair_epa_step3(gt_path, est_path, 0.02)

    assert int(result["matched"]) == int(t_est.size)
    assert result["eval_source"] == "epa_step3"
    assert result["eval_alignment"]["timeline_policy"] == "sparse_est_association"
    assert int(result["eval_alignment"]["sparse_matched"]) == int(t_est.size)
    assert result["eval_alignment"]["auto_sparse_low_rate_estimate_used"] is True


def test_evaluate_pair_epa_step3_uses_dense_timeline_for_moderate_rate_estimates(
    tmp_path: Path,
) -> None:
    t_gt = np.arange(0.0, 10.001, 0.01)
    pos_gt = np.column_stack(
        [
            0.4 * t_gt,
            np.sin(0.2 * t_gt),
            0.1 * np.cos(0.15 * t_gt),
        ]
    )
    quat_gt = R.from_euler("zyx", np.column_stack([0.5 * t_gt, 0.2 * t_gt, 0.3 * t_gt])).as_quat()

    sparse_ids = np.arange(0, t_gt.size, 10, dtype=int)
    t_est = t_gt[sparse_ids]
    pos_est = pos_gt[sparse_ids]
    quat_est = quat_gt[sparse_ids]

    gt_path = tmp_path / "gt_dense.tum"
    est_path = tmp_path / "est_10hz.tum"
    _write_tum(gt_path, t_gt, pos_gt, quat_gt)
    _write_tum(est_path, t_est, pos_est, quat_est)

    result = _evaluate_pair_epa_step3(gt_path, est_path, 0.02)

    assert int(result["matched"]) > int(t_est.size)
    assert result["eval_alignment"]["timeline_policy"] == "dense_overlap"
    assert result["eval_alignment"]["auto_sparse_low_rate_estimate_used"] is False


def test_evaluate_pair_epa_step3_no_fallback_keeps_sparse_estimate_association(
    tmp_path: Path,
) -> None:
    t_gt = np.arange(0.0, 10.001, 0.01)
    pos_gt = np.column_stack(
        [
            0.4 * t_gt,
            np.sin(0.2 * t_gt),
            0.1 * np.cos(0.15 * t_gt),
        ]
    )
    quat_gt = R.from_euler("zyx", np.column_stack([0.5 * t_gt, 0.2 * t_gt, 0.3 * t_gt])).as_quat()

    sparse_ids = np.arange(0, t_gt.size, 100, dtype=int)
    t_est = t_gt[sparse_ids]
    pos_est = pos_gt[sparse_ids]
    quat_est = quat_gt[sparse_ids]

    gt_path = tmp_path / "gt_dense.tum"
    est_path = tmp_path / "est_sparse.tum"
    _write_tum(gt_path, t_gt, pos_gt, quat_gt)
    _write_tum(est_path, t_est, pos_est, quat_est)

    result = _evaluate_pair_epa_step3(
        gt_path,
        est_path,
        0.02,
        allow_resampled_fallback=False,
    )

    assert int(result["matched"]) == int(t_est.size)
    assert result["eval_source"] == "epa_step3"
    assert result["eval_alignment"]["timeline_policy"] == "sparse_est_association"


def test_evaluate_pair_epa_step3_uses_dense_timeline_for_high_coverage_estimates(
    tmp_path: Path,
) -> None:
    t_gt = np.arange(0.0, 10.001, 0.01)
    pos_gt = np.column_stack(
        [
            0.4 * t_gt,
            np.sin(0.2 * t_gt),
            0.1 * np.cos(0.15 * t_gt),
        ]
    )
    quat_gt = R.from_euler("zyx", np.column_stack([0.5 * t_gt, 0.2 * t_gt, 0.3 * t_gt])).as_quat()

    sparse_ids = np.arange(0, t_gt.size, 2, dtype=int)
    t_est = t_gt[sparse_ids]
    pos_est = pos_gt[sparse_ids]
    quat_est = quat_gt[sparse_ids]

    gt_path = tmp_path / "gt_dense.tum"
    est_path = tmp_path / "est_high_coverage.tum"
    _write_tum(gt_path, t_gt, pos_gt, quat_gt)
    _write_tum(est_path, t_est, pos_est, quat_est)

    result = _evaluate_pair_epa_step3(gt_path, est_path, 0.02)

    assert int(result["matched"]) > int(t_est.size)
    assert result["eval_alignment"]["timeline_policy"] == "dense_overlap"
    assert result["eval_source"] == "epa_step3"


def test_evaluate_pair_epa_step3_resamples_when_low_rate_gt_misses_timestamp_gate(
    tmp_path: Path,
) -> None:
    t_gt = np.arange(0.0, 30.001, 1.0)
    t_est = np.arange(0.025, 30.0, 0.05)
    pos_gt = np.column_stack(
        [
            0.2 * t_gt,
            np.sin(0.2 * t_gt),
            0.1 * np.cos(0.1 * t_gt),
        ]
    )
    pos_est = np.column_stack(
        [
            0.2 * t_est,
            np.sin(0.2 * t_est),
            0.1 * np.cos(0.1 * t_est),
        ]
    )
    quat_gt = R.from_euler(
        "zyx", np.column_stack([0.05 * t_gt, 0.02 * t_gt, 0.01 * t_gt])
    ).as_quat()
    quat_est = R.from_euler(
        "zyx", np.column_stack([0.05 * t_est, 0.02 * t_est, 0.01 * t_est])
    ).as_quat()

    gt_path = tmp_path / "gt_1hz.tum"
    est_path = tmp_path / "est_20hz_shifted.tum"
    _write_tum(gt_path, t_gt, pos_gt, quat_gt)
    _write_tum(est_path, t_est, pos_est, quat_est)

    result = _evaluate_pair_epa_step3(gt_path, est_path, 0.02)

    assert int(result["matched"]) >= 20
    assert result["eval_source"] == "epa_step3"
    assert result["eval_alignment"]["timeline_policy"] == "gt_resampled_association_fallback"
    assert result["eval_alignment"]["sparse_association_failed"] is True
    assert result["eval_alignment"]["resampled_fallback_used"] is True
    assert float(result["ate3_pos"]["rmse"]) < 1e-2


def test_evaluate_pair_epa_step3_no_fallback_rejects_low_rate_gt(
    tmp_path: Path,
) -> None:
    t_gt = np.arange(0.0, 30.001, 1.0)
    t_est = np.arange(0.025, 30.0, 0.05)
    pos_gt = np.column_stack([0.2 * t_gt, np.sin(0.2 * t_gt), np.zeros_like(t_gt)])
    pos_est = np.column_stack([0.2 * t_est, np.sin(0.2 * t_est), np.zeros_like(t_est)])
    quat_gt = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t_gt.size, 1))
    quat_est = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t_est.size, 1))

    gt_path = tmp_path / "gt_1hz.tum"
    est_path = tmp_path / "est_20hz_shifted.tum"
    _write_tum(gt_path, t_gt, pos_gt, quat_gt)
    _write_tum(est_path, t_est, pos_est, quat_est)

    with pytest.raises(ValueError, match="resampled fallback disabled"):
        _evaluate_pair_epa_step3(
            gt_path,
            est_path,
            0.02,
            allow_resampled_fallback=False,
        )


def test_evaluate_pair_ov_style_sim3_recovers_scaled_similarity(tmp_path: Path) -> None:
    n = 120
    t = np.arange(n, dtype=float) * 0.1
    pos_gt = np.column_stack(
        [
            2.0 * np.cos(0.15 * t),
            1.5 * np.sin(0.2 * t),
            0.2 * t,
        ]
    )
    quat_gt = R.from_euler("zyx", np.column_stack([0.1 * t, 0.05 * np.sin(t), 0.03 * t])).as_quat()

    scale_true = 2.5
    r_align = R.from_euler("zyx", [35.0, -12.0, 8.0], degrees=True)
    t_align = np.array([0.6, -1.2, 0.4], dtype=float)
    pos_est = (r_align.inv().as_matrix() @ ((pos_gt - t_align) / scale_true).T).T
    quat_est = (r_align.inv() * R.from_quat(quat_gt)).as_quat()

    gt_path = tmp_path / "gt.tum"
    est_path = tmp_path / "est.tum"
    _write_tum(gt_path, t, pos_gt, quat_gt)
    _write_tum(est_path, t, pos_est, quat_est)

    result = _evaluate_pair_ov_style(gt_path, est_path, "sim3", 0.02)

    assert result["eval_alignment"]["timeline_policy"] == "dense_overlap"
    assert int(result["eval_alignment"]["sparse_matched"]) == int(t.size)
    np.testing.assert_allclose(
        float(result["eval_alignment"]["align_scale"]), scale_true, rtol=1e-6
    )
    assert float(result["ate3_pos"]["rmse"]) < 1e-6
    assert float(result["ate3_ori"]["rmse"]) < 1e-6
    assert result["eval_alignment"]["sim3_reliable"] is True


def test_evaluate_pair_ov_style_sim3_resamples_when_low_rate_gt_misses_timestamp_gate(
    tmp_path: Path,
) -> None:
    t_gt = np.arange(0.0, 30.001, 1.0)
    t_est = np.arange(0.025, 30.0, 0.05)
    pos_gt = np.column_stack(
        [
            0.2 * t_gt,
            np.sin(0.2 * t_gt),
            0.1 * np.cos(0.1 * t_gt),
        ]
    )
    pos_est_base = np.column_stack(
        [
            0.2 * t_est,
            np.sin(0.2 * t_est),
            0.1 * np.cos(0.1 * t_est),
        ]
    )
    pos_est = 1.4 * pos_est_base + np.array([2.0, -0.5, 0.2], dtype=float)
    quat_gt = R.from_euler(
        "zyx", np.column_stack([0.05 * t_gt, 0.02 * t_gt, 0.01 * t_gt])
    ).as_quat()
    quat_est = R.from_euler(
        "zyx", np.column_stack([0.05 * t_est, 0.02 * t_est, 0.01 * t_est])
    ).as_quat()

    gt_path = tmp_path / "gt_1hz.tum"
    est_path = tmp_path / "est_20hz_shifted.tum"
    _write_tum(gt_path, t_gt, pos_gt, quat_gt)
    _write_tum(est_path, t_est, pos_est, quat_est)

    result = _evaluate_pair_ov_style(gt_path, est_path, "sim3", 0.02)

    assert int(result["matched"]) >= 20
    assert result["eval_source"] == "epa_eval_align"
    assert result["eval_alignment"]["align_mode"] == "sim3"
    assert result["eval_alignment"]["timeline_policy"] == "gt_resampled_association_fallback"
    assert result["eval_alignment"]["sparse_association_failed"] is True
    assert result["eval_alignment"]["resampled_fallback_used"] is True
    assert float(result["ate3_pos"]["rmse"]) < 0.5


def test_evaluate_pair_ov_style_no_fallback_rejects_low_rate_gt(
    tmp_path: Path,
) -> None:
    t_gt = np.arange(0.0, 30.001, 1.0)
    t_est = np.arange(0.025, 30.0, 0.05)
    pos_gt = np.column_stack([0.2 * t_gt, np.sin(0.2 * t_gt), np.zeros_like(t_gt)])
    pos_est = np.column_stack([0.2 * t_est, np.sin(0.2 * t_est), np.zeros_like(t_est)])
    quat_gt = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t_gt.size, 1))
    quat_est = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t_est.size, 1))

    gt_path = tmp_path / "gt_1hz.tum"
    est_path = tmp_path / "est_20hz_shifted.tum"
    _write_tum(gt_path, t_gt, pos_gt, quat_gt)
    _write_tum(est_path, t_est, pos_est, quat_est)

    with pytest.raises(ValueError, match="Unable to associate enough timestamps"):
        _evaluate_pair_ov_style(
            gt_path,
            est_path,
            "sim3",
            0.02,
            allow_resampled_fallback=False,
        )


def test_error_comparison_sim3_resamples_when_low_rate_gt_misses_timestamp_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t_gt = np.arange(0.0, 30.001, 1.0)
    t_est = np.arange(0.025, 30.0, 0.05)
    pos_gt = np.column_stack(
        [
            0.2 * t_gt,
            np.sin(0.2 * t_gt),
            0.1 * np.cos(0.1 * t_gt),
        ]
    )
    pos_est_base = np.column_stack(
        [
            0.2 * t_est,
            np.sin(0.2 * t_est),
            0.1 * np.cos(0.1 * t_est),
        ]
    )
    pos_est = 1.4 * pos_est_base + np.array([2.0, -0.5, 0.2], dtype=float)
    quat_gt = R.from_euler(
        "zyx", np.column_stack([0.05 * t_gt, 0.02 * t_gt, 0.01 * t_gt])
    ).as_quat()
    quat_est = R.from_euler(
        "zyx", np.column_stack([0.05 * t_est, 0.02 * t_est, 0.01 * t_est])
    ).as_quat()
    gt_root = tmp_path / "gt"
    alg_root = tmp_path / "algorithms"
    run_dir = alg_root / "sqrtVINs_Mono" / "archaeo_sequence_2"
    gt_root.mkdir()
    run_dir.mkdir(parents=True)
    _write_tum(gt_root / "archaeo_sequence_2.txt", t_gt, pos_gt, quat_gt)
    _write_tum(run_dir / "traj_estimate.txt", t_est, pos_est, quat_est)

    args = _build_error_comparison_parser().parse_args(
        [
            "sim3",
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
    assert "skipping traj_estimate.txt" not in out
    assert "eval_source: epa_step3=0, epa_eval=1, failed=0" in out
    assert "TOOL SOURCE: epa_step3=0, epa_eval=1, failed=0" in out


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
            "--epa-success-global-gate-mode",
            "scale_aware",
            "--epa-success-global-gate-m",
            "35",
            "--epa-success-global-gate-path-ratio",
            "0.07",
            "--epa-success-global-gate-min-m",
            "3",
            "--epa-success-global-gate-max-m",
            "120",
            "--epa-success-global-gate-percentile",
            "10",
            "--epa-success-drift-rpe-1s-m",
            "3",
            "--epa-success-drift-ape-slope-mps",
            "1.5",
            "--epa-success-drift-ape-jump-m",
            "7",
            "--fail-on-skipped",
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
    assert args.epa_success_global_gate_mode == "scale_aware"
    assert args.epa_success_global_gate_m == 35
    assert args.epa_success_global_gate_path_ratio == 0.07
    assert args.epa_success_global_gate_min_m == 3
    assert args.epa_success_global_gate_max_m == 120
    assert args.epa_success_global_gate_percentile == 10
    assert args.epa_success_drift_rpe_1s_m == 3
    assert args.epa_success_drift_ape_slope_mps == 1.5
    assert args.epa_success_drift_ape_jump_m == 7
    assert bool(args.fail_on_skipped)


def test_module_subcommand_help_uses_real_command_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["error_comparison", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "align_mode folder_groundtruth folder_algorithms" in out
    assert "--epa-no-fallback" in out


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


def test_evaluate_pair_fails_without_ov_style_fallback_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
        )

    captured = capsys.readouterr()
    assert "[warn] EPA Step3 evaluation failed" not in captured.out


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
            source = "epa_eval_align"
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

    assert "eval_source: epa_step3=1, epa_eval=1, failed=0" in out
    assert "TOOL SOURCE: epa_step3=1, epa_eval=1, failed=0" in out
    assert "EVAL SOURCE NON-EPA RUNS" not in out
    assert "DRIFT-VALID SUCCESS RATE LATEX TABLE (% PATH LENGTH)" in out
    assert "FULL TRAJECTORY DISTANCE RPE LATEX TABLE" not in out
    assert "DRIFT-VALID ONLY DISTANCE RPE LATEX TABLE" not in out
    assert "DRIFT-VALID ONLY 10m SEGMENT RPE LATEX TABLE (ROT DEG / TRANS M)" in out
    assert "DRIFT-VALID ONLY 20m SEGMENT RPE LATEX TABLE (ROT DEG / TRANS M)" in out
    assert "DRIFT-VALID ONLY 50m SEGMENT RPE LATEX TABLE (ROT DEG / TRANS M)" in out
    assert "DRIFT-VALID ONLY 100m SEGMENT RPE LATEX TABLE (ROT DEG / TRANS M)" in out
    assert "& \\textbf{Average} \\\\hline" in out
    assert "(2/2 valid runs)" in out


def test_error_comparison_skips_failed_small_trajectory_runs(
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
    (run_dir / "bad_small.txt").write_text("", encoding="utf-8")

    def fake_evaluate_pair(**_kwargs):
        raise ValueError("Unable to associate enough timestamps")

    monkeypatch.setattr("epa.ov_eval_compat._evaluate_pair", fake_evaluate_pair)
    args = _build_error_comparison_parser().parse_args(["se3", str(gt_root), str(alg_root)])

    assert run_error_comparison(args) == 0
    out = capsys.readouterr().out
    assert "[warn] skipping bad_small.txt" in out
    assert "no valid runs for algo/seq" in out
    assert "FAILED RUNS: algo/seq/bad_small.txt:failed" in out
    assert "EVAL SOURCE NON-EPA RUNS" not in out


def test_error_comparison_fail_on_skipped_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    (run_dir / "bad_small.txt").write_text("", encoding="utf-8")

    def fake_evaluate_pair(**_kwargs):
        raise ValueError("Unable to associate enough timestamps")

    monkeypatch.setattr("epa.ov_eval_compat._evaluate_pair", fake_evaluate_pair)
    args = _build_error_comparison_parser().parse_args(
        ["se3", str(gt_root), str(alg_root), "--fail-on-skipped"]
    )

    assert run_error_comparison(args) == 1


def test_error_dataset_skips_failed_small_trajectory_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t = np.arange(5, dtype=float)
    gt_pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))
    gt_path = tmp_path / "seq.txt"
    alg_root = tmp_path / "algorithms"
    run_dir = alg_root / "algo" / "seq"
    run_dir.mkdir(parents=True)
    _write_tum(gt_path, t, gt_pos, quat)
    (run_dir / "bad_small.txt").write_text("", encoding="utf-8")

    def fake_evaluate_pair(**_kwargs):
        raise ValueError("Unable to associate enough timestamps")

    monkeypatch.setattr("epa.ov_eval_compat._evaluate_pair", fake_evaluate_pair)
    args = _build_error_dataset_parser().parse_args(["se3", str(gt_path), str(alg_root)])

    assert run_error_dataset(args) == 0
    out = capsys.readouterr().out
    assert "[warn] skipping bad_small.txt" in out
    assert "no valid runs for algo/seq" in out
    assert "failed_runs: bad_small.txt:failed" in out


def test_error_dataset_fail_on_skipped_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = np.arange(5, dtype=float)
    gt_pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))
    gt_path = tmp_path / "seq.txt"
    alg_root = tmp_path / "algorithms"
    run_dir = alg_root / "algo" / "seq"
    run_dir.mkdir(parents=True)
    _write_tum(gt_path, t, gt_pos, quat)
    (run_dir / "bad_small.txt").write_text("", encoding="utf-8")

    def fake_evaluate_pair(**_kwargs):
        raise ValueError("Unable to associate enough timestamps")

    monkeypatch.setattr("epa.ov_eval_compat._evaluate_pair", fake_evaluate_pair)
    args = _build_error_dataset_parser().parse_args(
        ["se3", str(gt_path), str(alg_root), "--fail-on-skipped"]
    )

    assert run_error_dataset(args) == 1


def test_error_dataset_reports_unreliable_rotation_with_high_translation_sr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t = np.arange(5, dtype=float)
    gt_pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat_gt = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))
    quat_bad = np.tile(R.from_euler("z", 120.0, degrees=True).as_quat(), (t.size, 1))
    gt_path = tmp_path / "seq.txt"
    alg_root = tmp_path / "algorithms"
    run_dir = alg_root / "algo" / "seq"
    run_dir.mkdir(parents=True)
    _write_tum(gt_path, t, gt_pos, quat_gt)
    (run_dir / "bad_rotation.txt").write_text("", encoding="utf-8")

    def fake_evaluate_pair(**_kwargs):
        return {
            "matched": int(t.size),
            "length_ratio": 1.0,
            "ate3_ori": {"rmse": 120.0},
            "ate3_pos": {"rmse": 0.0},
            "ate2_ori": {"rmse": 120.0},
            "ate2_pos": {"rmse": 0.0},
            "eval_source": "ov_eval_style",
            "gt_t": t,
            "gt_pos": gt_pos,
            "gt_quat": quat_gt,
            "est_pos": gt_pos,
            "est_quat": quat_bad,
        }

    monkeypatch.setattr("epa.ov_eval_compat._evaluate_pair", fake_evaluate_pair)
    args = _build_error_dataset_parser().parse_args(
        [
            "sim3",
            str(gt_path),
            str(alg_root),
            "--epa-success-threshold-mode",
            "fixed",
            "--epa-success-threshold-m",
            "10",
            "--epa-success-global-gate-m",
            "30",
        ]
    )

    assert run_error_dataset(args) == 0
    out = capsys.readouterr().out
    assert "unreliable_runs: bad_rotation.txt:" in out
    assert "Translation SR is high but rotation error is unstable" in out


def test_error_comparison_gates_unreliable_sim3_valid_only_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    t = np.arange(5, dtype=float)
    gt_pos = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    quat_gt = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (t.size, 1))
    quat_bad = np.tile(R.from_euler("z", 179.0, degrees=True).as_quat(), (t.size, 1))
    gt_root = tmp_path / "gt"
    alg_root = tmp_path / "algorithms"
    run_dir = alg_root / "algo" / "seq"
    gt_root.mkdir()
    run_dir.mkdir(parents=True)
    _write_tum(gt_root / "seq.txt", t, gt_pos, quat_gt)
    (run_dir / "unreliable_sim3.txt").write_text("", encoding="utf-8")

    def fake_evaluate_pair(**_kwargs):
        return {
            "matched": int(t.size),
            "length_ratio": 1.0,
            "ate3_ori": {"rmse": 179.0},
            "ate3_pos": {"rmse": 0.0},
            "ate2_ori": {"rmse": 179.0},
            "ate2_pos": {"rmse": 0.0},
            "eval_source": "epa_eval_align",
            "eval_alignment": {
                "align_mode": "sim3",
                "sim3_reliable": False,
                "sim3_failure_reason": "no_dominant_global_support",
            },
            "gt_t": t,
            "gt_pos": gt_pos,
            "gt_quat": quat_gt,
            "est_pos": gt_pos,
            "est_quat": quat_bad,
        }

    monkeypatch.setattr("epa.ov_eval_compat._evaluate_pair", fake_evaluate_pair)
    args = _build_error_comparison_parser().parse_args(
        [
            "sim3",
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
    assert "unreliable_runs: unreliable_sim3.txt:no_dominant_global_support" in out
    assert "algo & 0.00 & 0.00" in out
    assert "algo & nan / nan & nan / nan" in out
