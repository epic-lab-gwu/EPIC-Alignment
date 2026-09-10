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
from epa.core.pipeline_views import _default_interactive_view, _interactive_eval_view_specs
from epa.core.calibration import solve_world_alignment
from epa.core.trajectory_alignment import (
    _select_world_alignment_variant,
    _solve_extrinsic_world_candidate,
    _solve_world_alignment_posyaw_robust_trimmed,
)
from epa.core.solve_eval import _map_est_to_ref_nearest


def _legacy_match_nearest_timestamps(
    stamps_ref,
    stamps_est,
    *,
    max_diff_s: float,
    offset_est_s: float,
):
    ref = np.asarray(stamps_ref, dtype=float).reshape(-1)
    est = np.asarray(stamps_est, dtype=float).reshape(-1) + float(offset_est_s)
    ids_ref = []
    ids_est = []
    for index, timestamp in enumerate(ref):
        differences = np.abs(est - timestamp)
        nearest = int(np.argmin(differences))
        if float(differences[nearest]) <= float(max_diff_s):
            ids_ref.append(index)
            ids_est.append(nearest)
    return np.asarray(ids_ref, dtype=int), np.asarray(ids_est, dtype=int)


def _legacy_map_est_to_ref_nearest(t_ref, t_est, pos_est, quat_est, offset_est_s):
    ref = np.asarray(t_ref, dtype=float).reshape(-1)
    est_shifted = np.asarray(t_est, dtype=float).reshape(-1) - float(offset_est_s)
    indices = np.empty(ref.size, dtype=int)
    insertions = np.searchsorted(est_shifted, ref, side="left")
    for index, insertion in enumerate(insertions):
        if insertion <= 0:
            indices[index] = 0
        elif insertion >= est_shifted.size:
            indices[index] = est_shifted.size - 1
        else:
            left_error = abs(ref[index] - est_shifted[insertion - 1])
            right_error = abs(ref[index] - est_shifted[insertion])
            indices[index] = insertion - 1 if left_error <= right_error else insertion
    return np.asarray(pos_est)[indices], np.asarray(quat_est)[indices]


def test_interactive_eval_views_only_include_requested_alignment() -> None:
    assert _default_interactive_view("none") == "step3"
    assert _interactive_eval_view_specs("none") == ()
    assert _interactive_eval_view_specs("epa_step3") == ()
    assert _interactive_eval_view_specs("se3-original") == ()

    assert _default_interactive_view("sim3") == "sim3"
    assert _interactive_eval_view_specs("sim3") == (("sim3", "sim3", "sim3"),)
    assert _interactive_eval_view_specs("epa_sim3_v2") == (("sim3", "sim3", "sim3"),)

    assert _default_interactive_view("epica_sim3_stable") == "sim3"
    assert _interactive_eval_view_specs("epica_sim3_stable") == (("sim3", "sim3", "sim3"),)

    assert _default_interactive_view("posyaw") == "epa_posyaw"
    assert _interactive_eval_view_specs("posyaw") == (("epa_posyaw", "posyaw", "posyaw"),)


def _make_ref(n: int = 200) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([5.0 * t, np.sin(6.0 * t), 0.2 * np.cos(3.0 * t)])


def test_posyaw_world_alignment_does_not_absorb_roll_pitch() -> None:
    pos_est = _make_ref(160)
    r_tilted = R.from_euler("zyx", [35.0, -12.0, 8.0], degrees=True).as_matrix()
    trans = np.array([0.8, -1.1, 0.4], dtype=float)
    pos_gt = (r_tilted @ pos_est.T).T + trans

    fit = _solve_world_alignment_posyaw_robust_trimmed(pos_est, pos_gt)
    r_fit = np.asarray(fit["R"], dtype=float)
    pred = (r_fit @ pos_est.T).T + np.asarray(fit["t"], dtype=float)

    np.testing.assert_allclose(r_fit[2], np.array([0.0, 0.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(r_fit[:, 2], np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert float(fit["rmse_all_m"]) > 0.05
    assert float(np.sqrt(np.mean(np.sum((pred - pos_gt) ** 2, axis=1)))) > 0.05


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


def test_match_nearest_timestamps_vectorization_matches_legacy_scalar_oracle() -> None:
    rng = np.random.default_rng(20260905)
    ref = np.sort(rng.uniform(-1.0, 12.0, size=900))
    est = np.sort(rng.uniform(0.0, 10.0, size=350))
    est[40:45] = est[40]

    for offset in [-0.17, 0.0, 0.23]:
        for max_diff in [0.0, 0.01, 0.05, 0.5]:
            expected = _legacy_match_nearest_timestamps(
                ref, est, max_diff_s=max_diff, offset_est_s=offset
            )
            actual = _match_nearest_timestamps(
                ref, est, max_diff_s=max_diff, offset_est_s=offset
            )
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])


def test_match_nearest_timestamps_preserves_first_index_on_ties() -> None:
    ref = np.array([0.5, 1.0], dtype=float)
    est = np.array([0.0, 1.0, 1.0], dtype=float)

    ids_ref, ids_est = _match_nearest_timestamps(
        ref, est, max_diff_s=0.5, offset_est_s=0.0
    )

    np.testing.assert_array_equal(ids_ref, np.array([0, 1]))
    np.testing.assert_array_equal(ids_est, np.array([0, 1]))


def test_map_est_to_ref_nearest_vectorization_matches_legacy_scalar_oracle() -> None:
    rng = np.random.default_rng(20260906)
    ref = np.sort(rng.uniform(-2.0, 12.0, size=700))
    est = np.sort(rng.uniform(0.0, 10.0, size=260))
    positions = rng.normal(size=(est.size, 3))
    quaternions = R.random(est.size, random_state=rng).as_quat()

    for offset in [-0.2, 0.0, 0.35]:
        expected = _legacy_map_est_to_ref_nearest(
            ref, est, positions, quaternions, offset
        )
        actual = _map_est_to_ref_nearest(
            t_ref=ref,
            t_est=est,
            pos_est=positions,
            quat_est=quaternions,
            offset_est_s=offset,
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


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


def test_step3_robust_trimmed_alignment_ignores_drift_for_transform_only() -> None:
    n = 120
    t = np.linspace(0.0, 1.0, n)
    pos_gt = np.column_stack([8.0 * t, np.sin(5.0 * t), 0.4 * np.cos(4.0 * t)])
    R_world = R.from_euler("zyx", [25.0, -5.0, 8.0], degrees=True).as_matrix()
    t_world = np.array([2.0, -1.2, 0.6])
    pr = (R_world.T @ (pos_gt - t_world).T).T
    pr[80:] += np.column_stack(
        [
            np.linspace(0.0, 25.0, n - 80),
            np.linspace(0.0, -18.0, n - 80),
            np.zeros(n - 80),
        ]
    )
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))

    out = _solve_extrinsic_world_candidate(
        name="base",
        R_calc=np.eye(3),
        pr_sync=pr,
        qr_sync=quat,
        pos_gt_solve=pos_gt,
        quat_gt_solve=quat,
        pr_solve=pr,
        qr_solve=quat,
        global_align_mode="se3-original",
    )

    assert out["step3_alignment_mode"] == "robust_trimmed"
    assert int(out["step3_rejected_count"]) > 0
    early_rmse = float(np.sqrt(np.mean(np.sum((out["pr_final"][:80] - pos_gt[:80]) ** 2, axis=1))))
    late_rmse = float(np.sqrt(np.mean(np.sum((out["pr_final"][80:] - pos_gt[80:]) ** 2, axis=1))))
    assert early_rmse < 0.2
    assert late_rmse > 5.0


def test_step3_uses_longest_stable_window_when_trajectory_jumps() -> None:
    n = 90
    t = np.linspace(0.0, 1.0, n)
    pos_gt = np.column_stack([6.0 * t, np.sin(4.0 * t), 0.2 * np.cos(3.0 * t)])
    R_world = R.from_euler("zyx", [18.0, 4.0, -6.0], degrees=True).as_matrix()
    t_world = np.array([1.0, -0.7, 0.3])
    pr = (R_world.T @ (pos_gt - t_world).T).T
    pr[35:] += np.array([5000.0, -2000.0, 1000.0])
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))

    out = _solve_extrinsic_world_candidate(
        name="base",
        R_calc=np.eye(3),
        pr_sync=pr,
        qr_sync=quat,
        pos_gt_solve=pos_gt,
        quat_gt_solve=quat,
        pr_solve=pr,
        qr_solve=quat,
        global_align_mode="se3-original",
    )

    assert np.isfinite(out["step3_candidate_stable_sr_proxy"])
    # The robust full fit can win when it agrees with the longest-window fit.
    assert out["step3_candidate_stable_anchor_rmse_m"] < 0.2
    assert not np.any(out["step3_selected_mask"][:35])
    assert np.all(out["step3_selected_mask"][35:])
    early_rmse = float(np.sqrt(np.mean(np.sum((out["pr_final"][:35] - pos_gt[:35]) ** 2, axis=1))))
    late_rmse = float(np.sqrt(np.mean(np.sum((out["pr_final"][35:] - pos_gt[35:]) ** 2, axis=1))))
    assert early_rmse > 1000.0
    assert late_rmse < 0.2


def test_stable_candidate_can_use_early_anchor_preference():
    full = {"step3_solve_variant": "full", "step3_sr_proxy": .7,
            "step3_gate_proxy_m": 1., "step3_stable_anchor_rmse_m": 40.}
    stable = {"step3_solve_variant": "stable", "step3_sr_proxy": .4,
              "step3_gate_proxy_m": .1, "step3_stable_anchor_rmse_m": 1.}
    assert _select_world_alignment_variant([full, stable]) is stable


def test_step3_motion_prefix_only_overrides_global_failures() -> None:
    full_ok = {
        "step3_solve_variant": "full",
        "step3_sr_proxy": 0.13,
        "step3_gate_proxy_m": 4.4,
        "step3_stable_anchor_rmse_m": 64.0,
    }
    motion = {
        "step3_solve_variant": "motion_stable_prefix",
        "step3_sr_proxy": 0.38,
        "step3_gate_proxy_m": 1.0,
        "step3_stable_anchor_rmse_m": 59.0,
        "step3_stable_solve_ratio": 0.38,
    }

    assert _select_world_alignment_variant([full_ok, motion]) is full_ok

    full_failed = {
        "step3_solve_variant": "full",
        "step3_sr_proxy": 0.0,
        "step3_gate_proxy_m": 639.0,
        "step3_stable_anchor_rmse_m": 1093.0,
    }
    motion_recovery = {
        "step3_solve_variant": "motion_stable_prefix",
        "step3_sr_proxy": 0.25,
        "step3_gate_proxy_m": 2.0,
        "step3_stable_anchor_rmse_m": 5.5,
        "step3_stable_solve_ratio": 0.16,
    }

    assert _select_world_alignment_variant([full_failed, motion_recovery]) is motion_recovery
