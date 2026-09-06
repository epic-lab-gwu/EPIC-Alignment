import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.calibration import _rotation_median_inliers, solve_rotation_first_alignment


def test_rotation_jumps_do_not_bias_world_alignment():
    rng = np.random.default_rng(42)
    positions = rng.normal(size=(100, 3))
    orientations = R.random(100, random_state=rng)
    world = R.from_euler("xyz", [20, -15, 40], degrees=True)
    translation = np.array([1.0, -2.0, 0.5])
    reference_positions = world.apply(positions) + translation
    reference_orientations = world * orientations
    estimate_quats = orientations.as_quat()
    bad = np.arange(0, 100, 5)
    estimate_quats[bad] = (
        R.from_euler("z", 120, degrees=True) * orientations[bad]
    ).as_quat()

    relative = reference_orientations * R.from_quat(estimate_quats).inv()
    baseline_error = (world.inv() * relative.mean()).magnitude()
    assert np.degrees(baseline_error) > 10
    mask = _rotation_median_inliers(relative)
    assert np.count_nonzero(~mask) == len(bad)
    assert not mask[bad].any()
    rotation, offset = solve_rotation_first_alignment(
        positions, reference_positions, estimate_quats, reference_orientations.as_quat()
    )
    np.testing.assert_allclose(rotation, world.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(offset, translation, atol=1e-12)
    # All positions, including those with bad orientations, remain in the output fit.
    np.testing.assert_allclose(positions @ rotation.T + offset, reference_positions, atol=1e-12)


def test_shared_fast_motion_and_quaternion_signs_are_not_outliers():
    rng = np.random.default_rng(7)
    orientations = R.random(120, random_state=rng)
    world = R.from_euler("xyz", [179, 30, -150], degrees=True)
    reference = (world * orientations).as_quat()
    reference[::2] *= -1
    relative = R.from_quat(reference) * orientations.inv()
    assert _rotation_median_inliers(relative).all()


def test_noise_floor_preserves_small_orientation_noise_and_weighted_fit():
    rng = np.random.default_rng(8)
    positions = rng.normal(size=(50, 3))
    noise = R.from_rotvec(rng.uniform(-0.001, 0.001, size=(50, 3)))
    weights = np.arange(1, 51, dtype=float)
    assert _rotation_median_inliers(noise).all()
    rotation, offset = solve_rotation_first_alignment(
        positions, positions, R.identity(50).as_quat(), noise.as_quat(), weights
    )
    expected = noise.mean(weights=weights).as_matrix()
    np.testing.assert_allclose(rotation, expected, atol=1e-12)
    np.testing.assert_allclose(
        offset, np.average(positions - positions @ expected.T, axis=0, weights=weights),
        atol=1e-12,
    )


def test_two_pairs_keep_existing_mean_behavior():
    relative = R.from_euler("z", [0, 90], degrees=True)
    assert _rotation_median_inliers(relative).all()
    rotation, _ = solve_rotation_first_alignment(
        np.zeros((2, 3)), np.zeros((2, 3)), R.identity(2).as_quat(), relative.as_quat()
    )
    np.testing.assert_allclose(rotation, relative.mean().as_matrix(), atol=1e-12)
