from pathlib import Path

import numpy as np
import pytest

from vicon_ws.core.io_utils import (
    load_estimation_kitti,
    load_estimation_trajectory,
    load_estimation_tum,
    load_reference_trajectory,
    load_vicon_csv,
)


def test_load_estimation_tum(tmp_path: Path) -> None:
    traj_path = tmp_path / "traj.tum"
    traj_path.write_text(
        "# t tx ty tz qx qy qz qw\n"
        "0 0 0 0 0 0 0 1\n"
        "1 1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    t, pos, quat = load_estimation_tum(traj_path)

    np.testing.assert_allclose(t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(pos[:, 0], np.array([0.0, 1.0]))
    np.testing.assert_allclose(quat[:, 3], np.array([1.0, 1.0]))


def test_load_vicon_csv_with_euroc_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "gt.csv"
    csv_path.write_text(
        "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
        "q_RS_x [],q_RS_y [],q_RS_z [],q_RS_w []\n"
        "1000000000,0,0,0,0,0,0,1\n"
        "2000000000,1,0,0,0,0,0,1\n",
        encoding="utf-8",
    )

    t, pos, quat = load_vicon_csv(csv_path)

    np.testing.assert_allclose(t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(pos[:, 0], np.array([0.0, 1.0]))
    np.testing.assert_allclose(quat[:, 3], np.array([1.0, 1.0]))


def test_load_estimation_trajectory_auto_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "est.csv"
    csv_path.write_text(
        "timestamp,x,y,z,qx,qy,qz,qw\n"
        "0,0,0,0,0,0,0,1\n"
        "1,1,0,0,0,0,0,1\n",
        encoding="utf-8",
    )

    t, pos, quat = load_estimation_trajectory(csv_path, est_format="auto")

    np.testing.assert_allclose(t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(pos[:, 0], np.array([0.0, 1.0]))
    np.testing.assert_allclose(quat[:, 3], np.array([1.0, 1.0]))


def test_load_estimation_kitti(tmp_path: Path) -> None:
    kitti_path = tmp_path / "traj_kitti.txt"
    kitti_path.write_text(
        "1 0 0 1 0 1 0 2 0 0 1 3\n"
        "1 0 0 2 0 1 0 3 0 0 1 4\n",
        encoding="utf-8",
    )

    t, pos, quat = load_estimation_kitti(kitti_path)

    np.testing.assert_allclose(t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(pos, np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    np.testing.assert_allclose(quat[:, 3], np.array([1.0, 1.0]))


def test_load_estimation_trajectory_auto_kitti(tmp_path: Path) -> None:
    kitti_path = tmp_path / "est.txt"
    kitti_path.write_text(
        "1 0 0 1 0 1 0 2 0 0 1 3\n"
        "1 0 0 2 0 1 0 3 0 0 1 4\n",
        encoding="utf-8",
    )

    t, pos, _ = load_estimation_trajectory(kitti_path, est_format="auto")

    np.testing.assert_allclose(t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(pos[:, 0], np.array([1.0, 2.0]))


def test_load_reference_trajectory_tum(tmp_path: Path) -> None:
    traj_path = tmp_path / "gt.tum"
    traj_path.write_text(
        "0 0 0 0 0 0 0 1\n"
        "1 1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    t, pos, quat = load_reference_trajectory(traj_path, gt_format="tum")

    np.testing.assert_allclose(t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(pos[:, 0], np.array([0.0, 1.0]))
    np.testing.assert_allclose(quat[:, 3], np.array([1.0, 1.0]))


def test_bag_format_requires_topic_before_rosbags_import(tmp_path: Path) -> None:
    bag_path = tmp_path / "demo.bag"
    bag_path.write_bytes(b"")

    with pytest.raises(ValueError, match="topic"):
        load_estimation_trajectory(bag_path, est_format="bag", est_topic="")


def test_bag_tf_topic_requires_parent_child_format(tmp_path: Path) -> None:
    bag_path = tmp_path / "demo.bag"
    bag_path.write_bytes(b"")

    with pytest.raises(ValueError, match="parent.child"):
        load_estimation_trajectory(bag_path, est_format="bag", est_topic="/tf:map")
