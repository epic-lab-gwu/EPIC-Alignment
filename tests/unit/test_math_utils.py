import numpy as np

from vicon_ws.core.math_utils import (
    compute_error_statistics,
    normalize_time_to_seconds,
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
