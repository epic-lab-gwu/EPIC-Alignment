import numpy as np
from scipy.spatial.transform import Rotation as R

from epa.core.math_utils import (
    compute_error_statistics,
    normalize_time_to_seconds,
    quat_angle_error_rad,
    quat_multiply_xyzw,
    relative_se3,
    rmse,
)


def test_rmse_basic() -> None:
    values = np.array([1.0, -1.0])
    assert rmse(values) == 1.0


def test_compute_error_statistics_empty() -> None:
    stats = compute_error_statistics([])
    assert np.isnan(stats["rmse"])
    assert np.isnan(stats["mean"])


def test_normalize_time_to_seconds_from_nanoseconds() -> None:
    t_ns = np.array([1_000_000_000, 2_000_000_000, 3_500_000_000], dtype=float)
    t_sec = normalize_time_to_seconds(t_ns)
    np.testing.assert_allclose(t_sec, np.array([0.0, 1.0, 2.5]))


def test_relative_se3_identity_when_inputs_match() -> None:
    pose = np.eye(4)
    rel = relative_se3(pose, pose)
    np.testing.assert_allclose(rel, np.eye(4))


def test_vectorized_quaternion_product_matches_scipy() -> None:
    rng = np.random.default_rng(20260902)
    left = R.random(512, random_state=rng).as_quat()
    right = R.random(512, random_state=rng).as_quat()

    expected = (R.from_quat(left) * R.from_quat(right)).as_quat()
    actual = quat_multiply_xyzw(left, right)

    signs = np.where(np.sum(expected * actual, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    np.testing.assert_allclose(actual * signs, expected, rtol=1e-12, atol=1e-12)


def test_vectorized_quaternion_product_broadcasts_single_left_rotation() -> None:
    rng = np.random.default_rng(20260903)
    left = R.random(random_state=rng).as_quat()
    right = R.random(128, random_state=rng).as_quat()

    expected = (R.from_quat(left) * R.from_quat(right)).as_quat()
    actual = quat_multiply_xyzw(left, right)

    signs = np.where(np.sum(expected * actual, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
    np.testing.assert_allclose(actual * signs, expected, rtol=1e-12, atol=1e-12)


def test_quaternion_angle_error_matches_scipy() -> None:
    rng = np.random.default_rng(20260904)
    q_ref = R.random(512, random_state=rng).as_quat()
    q_est = R.random(512, random_state=rng).as_quat()

    expected = (R.from_quat(q_est).inv() * R.from_quat(q_ref)).magnitude()
    actual = quat_angle_error_rad(q_ref, q_est)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
