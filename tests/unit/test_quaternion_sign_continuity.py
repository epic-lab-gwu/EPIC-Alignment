from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from epa.core.time_alignment import interpolate_quat_linear


def _assert_same_rotations(actual: np.ndarray, expected: np.ndarray) -> None:
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-14)
    residual = R.from_quat(expected).inv() * R.from_quat(actual)
    np.testing.assert_allclose(residual.magnitude(), 0.0, atol=1e-13)


@pytest.mark.parametrize("first_sign", [-1.0, 1.0])
def test_linear_quaternion_interpolation_is_invariant_to_arbitrary_sign_flips(
    first_sign: float,
) -> None:
    rng = np.random.default_rng(20260906)
    t_src = np.linspace(0.0, 10.0, 41)
    angles = np.column_stack(
        (0.4 * np.sin(t_src), 0.3 * np.cos(0.7 * t_src), 0.9 * t_src)
    )
    q_src = R.from_euler("xyz", angles).as_quat()
    signs = rng.choice([-1.0, 1.0], size=t_src.size)
    signs[0] = first_sign
    t_query = np.linspace(t_src[0], t_src[-1], 401)

    expected = interpolate_quat_linear(t_src, q_src, t_query)
    actual = interpolate_quat_linear(t_src, q_src * signs[:, None], t_query)

    _assert_same_rotations(actual, expected)


def test_linear_quaternion_interpolation_handles_exact_antipodal_midpoint() -> None:
    q = R.from_euler("xyz", [30.0, -20.0, 70.0], degrees=True).as_quat()
    t_query = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    actual = interpolate_quat_linear([0.0, 1.0], np.array([q, -q]), t_query)

    _assert_same_rotations(actual, np.repeat(q[None, :], t_query.size, axis=0))


def test_linear_quaternion_interpolation_crosses_multiple_scalar_sign_boundaries() -> None:
    yaw = np.array([150.0, 170.0, 190.0, 210.0, 330.0, 350.0, 370.0, 390.0,
                    510.0, 530.0, 550.0, 570.0])
    q_src = R.from_euler("z", yaw, degrees=True).as_quat()
    # A canonical nonnegative scalar component jumps at 180 and 540 degrees.
    q_src[q_src[:, 3] < 0.0] *= -1.0
    t_src = np.arange(yaw.size, dtype=float)
    t_query = t_src[:-1] + 0.5
    expected = R.from_euler("z", 0.5 * (yaw[:-1] + yaw[1:]), degrees=True).as_quat()

    actual = interpolate_quat_linear(t_src, q_src, t_query)

    _assert_same_rotations(actual, expected)


def test_linear_quaternion_interpolation_normalizes_unequal_source_magnitudes() -> None:
    q_src = R.from_euler("z", [20.0, 100.0], degrees=True).as_quat()
    q_src *= np.array([0.25, -7.0])[:, None]

    actual = interpolate_quat_linear([0.0, 1.0], q_src, [0.5])

    _assert_same_rotations(actual, R.from_euler("z", [60.0], degrees=True).as_quat())


def test_linear_quaternion_interpolation_does_not_mutate_inputs() -> None:
    t_src = np.array([2.0, 0.0, 1.0])
    q_src = R.from_euler("z", [240.0, 0.0, 120.0], degrees=True).as_quat()
    q_src *= np.array([-3.0, 0.25, 2.0])[:, None]
    t_query = np.array([0.25, 0.75, 1.5])
    originals = [array.copy() for array in (t_src, q_src, t_query)]

    interpolate_quat_linear(t_src, q_src, t_query)

    for actual, expected in zip((t_src, q_src, t_query), originals):
        np.testing.assert_array_equal(actual, expected)


def test_linear_quaternion_extrapolation_preserves_geometry_under_sign_flips() -> None:
    t_src = np.array([0.0, 1.0, 2.0])
    q_src = R.from_euler("z", [20.0, 60.0, 95.0], degrees=True).as_quat()
    t_query = np.array([-0.25, 2.25])
    # Preserve the existing endpoint linear extrapolation, including its motion
    # beyond the source interval instead of clamping to endpoint rotations.
    expected = np.array([1.25 * q_src[0] - 0.25 * q_src[1],
                         -0.25 * q_src[1] + 1.25 * q_src[2]])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)

    for signs in ([1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]):
        actual = interpolate_quat_linear(t_src, q_src * np.array(signs)[:, None], t_query)
        _assert_same_rotations(actual, expected)


def test_linear_quaternion_sign_continuity_follows_sorted_timestamps() -> None:
    t_src = np.arange(4, dtype=float)
    q_src = R.from_euler("z", [0.0, 120.0, 240.0, 360.0], degrees=True).as_quat()
    order = np.array([0, 2, 1, 3])

    actual = interpolate_quat_linear(t_src[order], q_src[order], [0.5, 1.5, 2.5])

    expected = R.from_euler("z", [60.0, 180.0, 300.0], degrees=True).as_quat()
    _assert_same_rotations(actual, expected)
