import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from epa.core.calibration import (
    _constrain_weak_rotation, _rotation_observability, solve_extrinsic_rotation_multibaseline,
)
from epa.core import trajectory_alignment as alignment


def _pose_case(single_axis=False, translation=True):
    rng = np.random.default_rng(2409)
    increments = rng.normal(size=(159, 3)) * 0.08
    if single_axis:
        increments[:, :2] = 0.0
    poses = [R.identity()]
    for delta in R.from_rotvec(increments):
        poses.append(poses[-1] * delta)
    reference = R.concatenate(poses)
    extrinsic = R.from_euler("xyz", [12, -8, 23], degrees=True)
    world = R.from_euler("xyz", [17, 11, -32], degrees=True)
    world_t = np.array([3.0, -2.0, 1.0])
    extrinsic_t = np.array([0.2, -0.1, 0.15]) if translation else np.zeros(3)
    estimate = world.inv() * reference * extrinsic
    u = np.linspace(0, 10, len(reference))
    pg = np.column_stack((3 * np.sin(u), 2 * np.cos(u * 0.7), u))
    pe = world.inv().apply(pg - world_t) + (world.inv() * reference).apply(extrinsic_t)
    kwargs = dict(
        pr_sync=pe, qr_sync=estimate.as_quat(), pos_gt_solve=pg,
        quat_gt_solve=reference.as_quat(), pr_solve=pe, qr_solve=estimate.as_quat(),
        calibration_timestamps_s=np.arange(len(reference)) * 0.1,
    )
    return kwargs, extrinsic, extrinsic_t, world, world_t


def test_observability_accepts_two_nonparallel_axes_not_just_rank_three():
    vectors = np.tile([[0.2, 0, 0], [0, 0.2, 0]], (30, 1))
    x = R.from_euler("xyz", [12, 8, 35], degrees=True).as_matrix()
    info = _rotation_observability(vectors @ x.T, vectors, x, np.ones(len(vectors)))
    assert info["observable"]
    assert info["information_ratio"] == pytest.approx(0.5)


@pytest.mark.parametrize("transverse", [0.0, 0.01])
def test_observability_rejects_single_or_nearly_single_axis(transverse):
    vectors = np.tile([[transverse, 0, 1], [0, transverse, 1]], (30, 1))
    info = _rotation_observability(vectors, vectors, np.eye(3), np.ones(len(vectors)))
    assert not info["observable"]
    assert info["information_ratio"] < 0.1


def test_observability_checks_matched_agreement_not_only_excitation():
    a = np.vstack((np.eye(3), -np.eye(3)))
    b = np.vstack((np.eye(3), np.eye(3)))
    info = _rotation_observability(a, b, np.eye(3), np.ones(len(a)))
    assert info["reference_information_ratio"] == pytest.approx(1)
    assert info["estimate_information_ratio"] == pytest.approx(1)
    assert info["matched_information_ratio"] == 0
    assert not info["observable"]


def test_observability_ratio_is_invariant_to_motion_scale():
    rng = np.random.default_rng(21)
    a = rng.normal(size=(100, 3))
    first = _rotation_observability(a, a, np.eye(3), np.ones(len(a)))
    second = _rotation_observability(a * 0.01, a * 0.01, np.eye(3), np.ones(len(a)))
    assert first["information_ratio"] == pytest.approx(second["information_ratio"])


def test_well_observed_nonidentity_extrinsic_still_recovers_full_transform():
    # Isolate the rotation safeguard from the existing lever-arm convention.
    kwargs, x, t, world, world_t = _pose_case(translation=False)
    fitted = alignment._solve_extrinsic_and_world_alignment(**kwargs)
    np.testing.assert_allclose(fitted["R_calc"], x.as_matrix(), atol=1e-10)
    np.testing.assert_allclose(fitted["t_calc"], t, atol=1e-10)
    np.testing.assert_allclose(fitted["Rw_calc"], world.as_matrix(), atol=1e-10)
    np.testing.assert_allclose(fitted["tw_calc"], world_t, atol=1e-10)
    choice = fitted["step3_choice"]
    assert choice["extrinsic_rotation_observable"] == 1
    assert choice["extrinsic_identity_selected"] == 0
    assert choice["extrinsic_selection_reason"] == "calibrated_candidate_accepted"
    assert choice["extrinsic_selected_position_rmse_m"] < 1e-10
    assert choice["orientation_candidate_count"] == 5


def test_unobservable_fit_uses_constrained_candidate_without_axis_flips(monkeypatch):
    kwargs, *_ = _pose_case(single_axis=True)
    def unexpected_flips(*args, **kwargs):
        pytest.fail("Constrained rotation must not produce axis-flipped candidates")
    monkeypatch.setattr(alignment, "_extrinsic_rotation_candidates", unexpected_flips)
    actual = alignment._solve_extrinsic_and_world_alignment(**kwargs, compare_identity_candidate=False)
    choice = actual["step3_choice"]
    assert choice["extrinsic_selection_reason"] == "constrained_candidate_accepted"
    assert choice["extrinsic_rotation_observable"] == 0
    assert choice["extrinsic_rotation_constraint_success"] == 1
    assert choice["extrinsic_rotation_constraint_dimension"] == 1
    assert choice["orientation_candidate_count"] == 2
    assert choice["extrinsic_identity_selected"] == 0
    assert choice["extrinsic_identity_comparison_enabled"] == 0


def test_constrained_solver_recovers_supported_rotation_and_zero_weak_component():
    a = np.tile(np.vstack((np.diag([.003, .003, .1]), -np.diag([.003, .003, .1]))), (20, 1))
    truth = R.from_rotvec([.08, -.04, 0])
    b = truth.inv().apply(a)
    info = _rotation_observability(a, b, truth.as_matrix(), np.ones(len(a)))
    assert not info["observable"]
    x, constrained = _constrain_weak_rotation(a, b, truth.as_matrix(), np.ones(len(a)), info)
    assert constrained["constraint_success"]
    np.testing.assert_allclose(x, truth.as_matrix(), atol=1e-8)
    axes = np.asarray(constrained["constraint_weak_axes"])
    np.testing.assert_allclose(axes @ R.from_matrix(x).as_rotvec(), 0, atol=1e-10)


def test_constraints_preserve_original_rotation_pairs_and_mask():
    kwargs, *_ = _pose_case(single_axis=True)
    args = (kwargs["quat_gt_solve"], kwargs["qr_solve"])
    extra = dict(timestamps_s=kwargs["calibration_timestamps_s"])
    raw, pairs, mask, _ = solve_extrinsic_rotation_multibaseline(*args, **extra, constrain_unobservable=False)
    x, constrained_pairs, constrained_mask, info = solve_extrinsic_rotation_multibaseline(*args, **extra)
    assert info["constraint_success"]
    np.testing.assert_array_equal(pairs, constrained_pairs)
    np.testing.assert_array_equal(mask, constrained_mask)
    assert info["unconstrained_angle_deg"] == pytest.approx(np.degrees(R.from_matrix(raw).magnitude()))


def test_failed_constrained_fit_still_has_safe_identity_fallback(monkeypatch):
    from epa.core import calibration
    kwargs, *_ = _pose_case(single_axis=True)
    monkeypatch.setattr(calibration, "_constrain_weak_rotation", lambda a,b,x,w,info: (x, {"constraint_success": False}))
    actual = alignment._solve_extrinsic_and_world_alignment(**kwargs, compare_identity_candidate=False)
    assert actual["step3_choice"]["extrinsic_selection_reason"] == "unobservable_rotation"
    np.testing.assert_array_equal(actual["R_calc"], np.eye(3))
    np.testing.assert_array_equal(actual["t_calc"], np.zeros(3))


def test_identity_comparison_is_enabled_by_default_for_constrained_fit(monkeypatch):
    kwargs, *_ = _pose_case(single_axis=True)
    seen = []
    def select(candidates, identity):
        seen.extend(candidates)
        return identity
    monkeypatch.setattr(alignment, "_select_extrinsic_with_identity", select)
    actual = alignment._solve_extrinsic_and_world_alignment(**kwargs)
    assert len(seen) == 1 and seen[0]["candidate_name"] == "constrained"
    assert actual["step3_choice"]["extrinsic_selection_reason"] == "identity_preferred"


def test_no_supported_information_does_not_claim_success():
    a = np.zeros((10, 3))
    info = _rotation_observability(a, a, np.eye(3), np.ones(10))
    _, constrained = _constrain_weak_rotation(a, a, np.eye(3), np.ones(10), info)
    assert not constrained["constraint_success"]


@pytest.mark.parametrize("disable_safeguard", [False, True])
@pytest.mark.parametrize("disable_extrinsics", [False, True])
def test_compat_identity_option_reaches_solver_and_preserves_disable_extrinsics(
    tmp_path, monkeypatch, disable_safeguard, disable_extrinsics,
):
    from epa.compat.ov_eval.evaluate import _evaluate_pair
    kwargs, *_ = _pose_case(translation=False)
    times = kwargs["calibration_timestamps_s"]
    gt, est = tmp_path / "gt.txt", tmp_path / "est.txt"
    np.savetxt(gt, np.column_stack((times, kwargs["pos_gt_solve"], kwargs["quat_gt_solve"])))
    np.savetxt(est, np.column_stack((times, kwargs["pr_solve"], kwargs["qr_solve"])))
    calls = []
    def prefer_identity(candidates, identity):
        calls.append(len(candidates))
        return identity
    monkeypatch.setattr(alignment, "_select_extrinsic_with_identity", prefer_identity)
    result = _evaluate_pair(
        gt, est, "se3", .02, epa_disable_time_offset_calibration=True,
        epa_disable_identity_safeguard=disable_safeguard,
        epa_disable_extrinsic_calibration=disable_extrinsics,
    )
    info = result["eval_alignment"]
    assert info["extrinsic_identity_comparison_enabled"] == float(not disable_safeguard)
    if disable_extrinsics:
        assert info["extrinsic_selection_reason"] == "disabled"
        assert info["extrinsic_identity_selected"] == 1
    elif disable_safeguard:
        assert not calls
        assert info["extrinsic_identity_selected"] == 0
    else:
        assert calls
        assert info["extrinsic_identity_selected"] == 1


def test_insufficient_rotation_falls_back_but_malformed_input_does_not():
    kwargs, *_ = _pose_case()
    kwargs["quat_gt_solve"] = np.tile([0, 0, 0, 1.0], (len(kwargs["pr_solve"]), 1))
    kwargs["qr_solve"] = kwargs["quat_gt_solve"].copy()
    kwargs["qr_sync"] = kwargs["qr_solve"]
    actual = alignment._solve_extrinsic_and_world_alignment(**kwargs)
    assert actual["step3_choice"]["extrinsic_selection_reason"] == "insufficient_rotation_excitation"
    np.testing.assert_array_equal(actual["R_calc"], np.eye(3))
    np.testing.assert_array_equal(actual["t_calc"], np.zeros(3))
    kwargs["qr_solve"] = kwargs["qr_solve"][:-1]
    with pytest.raises(ValueError, match="paired Nx4"):
        alignment._solve_extrinsic_and_world_alignment(**kwargs)


def _candidate(name, full_rmse, trimmed_rmse=0.01, orientation_rmse=0.1):
    return dict(candidate_name=name, identity_comparison_rmse_m=full_rmse,
                step3_rmse_selected_m=trimmed_rmse, rotation_ape_rmse_deg=orientation_rmse,
                rot_res_median_deg=0.1)


def test_identity_comparison_does_not_hide_regression_behind_trimmed_score():
    identity = _candidate("identity", 1.0, 1.0, 2.0)
    calibrated = _candidate("base", 1.2, 0.01, 0.1)
    assert alignment._select_extrinsic_with_identity([calibrated], identity) is identity


def test_identity_gate_filters_bad_candidates_before_selecting_a_good_one():
    identity = _candidate("identity", 1.0, 1.0, 2.0)
    bad = _candidate("base", 1.2, 0.01, 0.1)
    good = _candidate("other", 0.8, 0.8, 0.2)
    assert alignment._select_extrinsic_with_identity([bad, good], identity) is good


def test_position_ties_prefer_identity_unless_orientation_improves():
    identity = _candidate("identity", 1.0, 1.0, 2.0)
    tied = _candidate("base", 1.0, 0.01, 2.0)
    improved = _candidate("base", 1.0, 0.01, 1.0)
    assert alignment._select_extrinsic_with_identity([tied], identity) is identity
    assert alignment._select_extrinsic_with_identity([improved], identity) is improved


@pytest.mark.parametrize("error", [np.nan, np.inf])
def test_nonfinite_candidate_cannot_override_identity(error):
    identity = _candidate("identity", 1.0)
    assert alignment._select_extrinsic_with_identity([_candidate("base", error)], identity) is identity


def test_observability_metadata_is_reported_without_discarding_pair_mask():
    kwargs, *_ = _pose_case(single_axis=True)
    _, pairs, mask, info = solve_extrinsic_rotation_multibaseline(
        kwargs["quat_gt_solve"], kwargs["qr_solve"], timestamps_s=kwargs["calibration_timestamps_s"]
    )
    assert not info["observable"]
    assert len(pairs) == len(mask)
    assert mask.sum() == info["accepted_count"] > 0


def test_compat_reports_identity_reason_for_zero_rotation_even_with_no_resampling_fallback(tmp_path):
    from epa.compat.ov_eval.evaluate import _evaluate_pair

    t = np.arange(20) * 0.05
    data = np.column_stack((t, t, np.zeros((len(t), 5)), np.ones(len(t))))
    trajectory = tmp_path / "stationary_orientation.tum"
    np.savetxt(trajectory, data)
    result = _evaluate_pair(
        trajectory, trajectory, "se3", 0.02, epa_no_fallback=True,
        epa_disable_time_offset_calibration=True,
    )
    assert result["ate3_pos"]["rmse"] < 1e-12
    assert result["eval_source"] == "epa_se3"
    assert result["eval_alignment"]["extrinsic_selection_reason"] == "insufficient_rotation_excitation"
    assert result["eval_alignment"]["extrinsic_identity_selected"] == 1
