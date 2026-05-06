from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.ov_eval_compat import _evaluate_pair_epa_step3, _evaluate_pair_ov_style


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
