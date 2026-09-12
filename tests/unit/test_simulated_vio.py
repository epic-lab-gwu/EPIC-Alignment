from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments" / "calibration_sensitivity"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_pilot as pilot  # noqa: E402


def _trajectory(count: int = 200) -> tuple[np.ndarray, R, pilot.Profile]:
    t = np.arange(count, dtype=float) * 0.1
    position = np.column_stack((0.2 * t, np.sin(0.3 * t), 0.1 * np.cos(0.2 * t)))
    rotation = R.from_euler(
        "zyx", np.column_stack((0.4 * np.sin(0.2 * t), 0.2 * np.cos(0.1 * t), 0.1 * t))
    )
    profile = pilot.Profile("test", "test", "sequence", Path("/tmp/gt.tum"), "tum")
    return position, rotation, profile


def test_oracle_simulation_is_identity() -> None:
    position, rotation, profile = _trajectory()
    simulated_position, simulated_rotation, translation_rms, rotation_rms = pilot.simulated_vio(
        position, rotation, profile, pilot.Accuracy("oracle", 0.0, 0.0), 1101, 10.0
    )

    np.testing.assert_allclose(simulated_position, position)
    np.testing.assert_allclose((simulated_rotation * rotation.inv()).as_rotvec(), 0.0, atol=1e-12)
    assert translation_rms == 0.0
    assert rotation_rms == 0.0


def test_rotation_error_is_left_multiplied_and_matches_reported_rms() -> None:
    position, rotation, profile = _trajectory()
    _, simulated_rotation, _, reported_rms = pilot.simulated_vio(
        position, rotation, profile, pilot.Accuracy("medium", 0.0, 0.5), 2202, 1.0
    )

    # The injected error is Delta_R_world = R_vio R_gt^{-1}; this is the
    # global-frame perturbation specified by the simulation model.
    global_error = simulated_rotation * rotation.inv()
    measured_rms = float(np.sqrt(np.mean(np.square(global_error.magnitude()))) * 180.0 / np.pi)
    assert np.isclose(reported_rms, measured_rms, atol=1e-12)
    assert np.isclose(measured_rms, 0.5, atol=1e-10)


def test_yaw_drift_parameter_changes_the_seeded_rotation_trace() -> None:
    position, rotation, profile = _trajectory()
    _, no_drift, _, _ = pilot.simulated_vio(
        position, rotation, profile, pilot.Accuracy("medium", 0.0, 0.5), 3303, 1.0
    )
    _, drift, _, _ = pilot.simulated_vio(
        position,
        rotation,
        profile,
        pilot.Accuracy("medium", 0.0, 0.5),
        3303,
        1.0,
        yaw_drift_deg_per_s=1.0,
    )
    assert np.max((drift * no_drift.inv()).magnitude()) > 1e-6
