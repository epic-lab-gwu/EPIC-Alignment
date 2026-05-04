import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.pipeline_modular import (
    _associate_gt_est,
    _compute_piecewise_alignment,
    _downsample_by_max_hz,
    _match_nearest_timestamps,
    _offset_match_diagnostics,
    _select_gt_overlap_window,
    _search_direct_offset_from_matched_pairs,
)
from epa.core.calibration import solve_world_alignment


def _make_ref(n: int = 200) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([5.0 * t, np.sin(6.0 * t), 0.2 * np.cos(3.0 * t)])


def test_search_direct_offset_from_matched_pairs_recovers_small_offset() -> None:
    n = 600
    dt = 0.02
    true_offset = 0.18
    t_ref = np.arange(n, dtype=float) * dt
    t_est = t_ref + true_offset
    pos_ref = np.column_stack(
        [
            0.4 * t_ref + 0.05 * np.sin(0.7 * t_ref),
            np.sin(0.3 * t_ref),
            np.cos(0.2 * t_ref),
        ]
    )
    pos_est = pos_ref + 0.001 * np.column_stack([np.sin(t_ref), np.cos(t_ref), np.sin(0.5 * t_ref)])

    out = _search_direct_offset_from_matched_pairs(
        t_ref=t_ref,
        pos_ref=pos_ref,
        t_est=t_est,
        pos_est=pos_est,
        initial_offset_s=0.0,
        max_diff_s=0.03,
        min_match_ratio=0.3,
    )

    assert out is not None
    assert abs(float(out["offset_s"]) - float(true_offset)) < 0.03
    assert int(out["matches"]) > 100


def test_downsample_by_max_hz_caps_dense_eval_grid_and_keeps_endpoints() -> None:
    t = np.arange(0.0, 1.001, 0.001)
    pos = np.column_stack([t, t * 0.0, t * 0.0])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t.size, 1))

    t_ds, pos_ds, quat_ds, info = _downsample_by_max_hz(t, pos, quat, 100.0)

    assert bool(info["enabled"])
    assert int(info["input_samples"]) == t.size
    assert int(info["output_samples"]) == t_ds.size
    assert t_ds.size < t.size
    np.testing.assert_allclose(t_ds[0], t[0])
    np.testing.assert_allclose(t_ds[-1], t[-1])
    assert np.min(np.diff(t_ds[:-1])) >= 0.01 * (1.0 - 1e-6)
    assert pos_ds.shape[0] == t_ds.size
    assert quat_ds.shape[0] == t_ds.size


def test_select_gt_overlap_window_trims_to_estimation_support() -> None:
    t = np.arange(0.0, 10.0, 1.0)
    pos = np.column_stack([t, t * 0.0, t * 0.0])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t.size, 1))
    t_est_sync = np.arange(3.0, 8.0, 1.0)

    t_sel, pos_sel, quat_sel, info = _select_gt_overlap_window(
        t, pos, quat, t_est_sync, min_samples=3
    )

    assert not bool(info["fallback_used"])
    assert str(info["mode"]) == "overlap"
    np.testing.assert_allclose(t_sel, np.array([3.0, 4.0, 5.0, 6.0, 7.0]))
    assert pos_sel.shape[0] == t_sel.size
    assert quat_sel.shape[0] == t_sel.size
    assert float(info["extrapolated_sample_ratio"]) == 0.0


def test_select_gt_overlap_window_falls_back_when_overlap_too_small() -> None:
    t = np.arange(0.0, 10.0, 1.0)
    pos = np.column_stack([t, t * 0.0, t * 0.0])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (t.size, 1))
    t_est_sync = np.array([3.0, 4.0])

    t_sel, pos_sel, quat_sel, info = _select_gt_overlap_window(
        t, pos, quat, t_est_sync, min_samples=3
    )

    assert bool(info["fallback_used"])
    assert str(info["mode"]) == "full_gt_extrapolate_fallback"
    np.testing.assert_allclose(t_sel, t)
    assert pos_sel.shape[0] == t.size
    assert quat_sel.shape[0] == t.size
    assert float(info["extrapolated_sample_ratio"]) > 0.0


def test_match_nearest_timestamps_allows_repeated_est_indices() -> None:
    t_ref = np.array([0.00, 0.01, 0.02, 0.03], dtype=float)
    t_est = np.array([0.00, 0.02], dtype=float)
    ids_ref, ids_est = _match_nearest_timestamps(
        t_ref,
        t_est,
        max_diff_s=0.02,
        offset_est_s=0.0,
    )
    assert ids_ref.size == 4
    # Middle two points map to nearest 0.02 sample (repeated index allowed).
    assert np.array_equal(ids_est, np.array([0, 0, 1, 1], dtype=int))


def test_associate_gt_est_uses_shorter_side_cardinality() -> None:
    t_ref = np.array([0.00, 0.01, 0.02, 0.03, 0.04], dtype=float)
    t_est = np.array([0.00, 0.02], dtype=float)

    ids_ref, ids_est = _associate_gt_est(
        t_ref,
        t_est,
        max_diff_s=0.02,
        offset_est_s=0.0,
    )

    # Evo-style association matches from shorter(est) to longer(ref).
    assert ids_ref.size == t_est.size
    assert ids_est.size == t_est.size
    assert np.array_equal(ids_est, np.array([0, 1], dtype=int))


def test_compute_piecewise_alignment_handles_segmented_drift() -> None:
    n = 480
    t_ref = np.linspace(0.0, 48.0, n)
    pos_ref = np.column_stack(
        [
            0.35 * t_ref + 0.4 * np.sin(0.2 * t_ref),
            np.sin(0.35 * t_ref) + 0.1 * np.cos(0.12 * t_ref),
            0.3 * np.cos(0.18 * t_ref),
        ]
    )

    seg_edges = [0, 160, 320, n]
    scales = [1.00, 1.12, 0.88]
    yaw_deg = [0.0, 10.0, -8.0]
    trans = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.8, -0.5, 0.2]),
        np.array([-0.6, 0.4, -0.1]),
    ]

    pos_base = np.zeros_like(pos_ref)
    for (s, e), scale, yaw, tvec in zip(zip(seg_edges[:-1], seg_edges[1:]), scales, yaw_deg, trans):
        R_seg = R.from_euler("z", yaw, degrees=True).as_matrix()
        pos_base[s:e] = ((R_seg.T @ (pos_ref[s:e] - tvec).T).T) / scale

    quat_base = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))

    R_global, t_global = solve_world_alignment(pos_base, pos_ref)
    pred_global = (R_global @ pos_base.T).T + t_global
    rmse_global = float(np.sqrt(np.mean(np.sum((pred_global - pos_ref) ** 2, axis=1))))

    piecewise = _compute_piecewise_alignment(
        t_ref=t_ref,
        pos_ref=pos_ref,
        pos_base=pos_base,
        quat_base=quat_base,
        fallback_pos=pred_global,
        fallback_quat=quat_base,
        segment_duration_s=18.0,
        overlap_ratio=0.5,
        min_samples=60,
        with_scale=True,
    )

    assert piecewise is not None
    assert float(piecewise["coverage_ratio"]) > 0.99
    assert float(piecewise["rmse_m"]) < 0.35 * rmse_global


def test_offset_match_diagnostics_uses_overlap_ratio_when_overlap_is_large() -> None:
    t_ref = np.arange(0.0, 1000.0, 1.0)
    t_est = np.arange(750.0, 1750.0, 1.0)

    out = _offset_match_diagnostics(
        t_ref,
        t_est,
        max_diff_s=1e-6,
        offset_est_s=0.0,
    )

    assert int(out["match_count"]) == 250
    assert float(out["ratio_global"]) < 0.3
    assert float(out["ratio_overlap"]) > 0.99
    assert float(out["ratio_gate"]) > 0.99
    assert int(out["pair_cap_overlap"]) >= int(out["overlap_gate_min_pairs"])


def test_offset_match_diagnostics_keeps_global_ratio_when_overlap_is_too_small() -> None:
    t_ref = np.arange(0.0, 1000.0, 1.0)
    t_est = np.arange(970.0, 1970.0, 1.0)

    out = _offset_match_diagnostics(
        t_ref,
        t_est,
        max_diff_s=1e-6,
        offset_est_s=0.0,
    )

    assert int(out["match_count"]) == 30
    assert float(out["ratio_global"]) < 0.05
    assert float(out["ratio_overlap"]) > 0.99
    assert int(out["pair_cap_overlap"]) < int(out["overlap_gate_min_pairs"])
    np.testing.assert_allclose(float(out["ratio_gate"]), float(out["ratio_global"]))
