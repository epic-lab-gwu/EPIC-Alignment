import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.calibration import (
    cauchy_weights,
    huber_weights,
    solve_world_alignment,
    solve_world_alignment_robust,
)


def test_huber_weights_downweight_only_large_residuals() -> None:
    weights = huber_weights(np.array([0.01, 0.1, 1.0, np.inf]), delta=0.1)

    np.testing.assert_allclose(weights[:2], np.array([1.0, 1.0]))
    np.testing.assert_allclose(weights[2], 0.1)
    assert weights[3] == 0.0


def test_cauchy_weights_are_smooth_and_bounded() -> None:
    weights = cauchy_weights(np.array([0.0, 0.1, 1.0, np.inf]), delta=0.1)

    np.testing.assert_allclose(weights[0], 1.0)
    np.testing.assert_allclose(weights[1], 0.5)
    assert 0.0 < weights[2] < weights[1]
    assert weights[3] == 0.0


def test_robust_world_alignment_recovers_transform_with_sparse_outliers() -> None:
    rng = np.random.default_rng(2)
    source = rng.normal(size=(100, 3))
    true_rotation = R.from_euler("zyx", [20.0, -8.0, 12.0], degrees=True).as_matrix()
    true_translation = np.array([1.2, -0.4, 0.7])
    target = (true_rotation @ source.T).T + true_translation
    target[:10] += rng.normal(scale=4.0, size=(10, 3))

    standard_rotation, standard_translation = solve_world_alignment(source, target)
    robust_rotation, robust_translation, info = solve_world_alignment_robust(
        source,
        target,
        robust_kernel="huber",
        return_info=True,
    )

    standard_rotation_error = np.degrees(
        R.from_matrix(standard_rotation @ true_rotation.T).magnitude()
    )
    robust_rotation_error = np.degrees(
        R.from_matrix(robust_rotation @ true_rotation.T).magnitude()
    )
    standard_translation_error = np.linalg.norm(standard_translation - true_translation)
    robust_translation_error = np.linalg.norm(robust_translation - true_translation)

    assert robust_rotation_error < standard_rotation_error * 0.25
    assert robust_translation_error < standard_translation_error * 0.25
    assert info["kernel"] == "huber"
    assert int(info["downweighted_count"]) >= 10
    assert int(info["iterations"]) >= 1
