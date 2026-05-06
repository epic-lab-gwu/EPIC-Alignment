from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import epa
from epa.ov_eval_compat import _format_source_counts
from epa.ov_eval_compat import (
    _build_error_comparison_parser,
    _evaluate_pair,
    _evaluate_pair_epa_step3,
    _evaluate_pair_ov_style,
    _format_source_details,
)


def test_package_version_matches_release() -> None:
    assert epa.__version__ == "0.1.7"


def test_format_source_counts_is_deterministic() -> None:
    assert _format_source_counts({"ov_eval_style": 1, "epa_step3": 2}) == "epa_step3=2, ov_eval_style=1"
    assert _format_source_counts({}) == "none"
    assert _format_source_details(["run1.txt:ov_eval_style", "run2.txt:unknown"]) == (
        "run1.txt:ov_eval_style, run2.txt:unknown"
    )
    assert _format_source_details([]) == "none"


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
    assert float(epa["ate3_ori"]["rmse"]) < float(ov["ate3_ori"]["rmse"])
    assert float(epa["ate3_pos"]["rmse"]) < float(ov["ate3_pos"]["rmse"])
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
        ]
    )

    assert args.epa_dt_resample == 0.01
    assert args.epa_offset_min_match_ratio == 0.1
    assert args.epa_downsample_hz == 20
    assert args.epa_quat_interp == "slerp"
    assert bool(args.epa_no_fallback)


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
