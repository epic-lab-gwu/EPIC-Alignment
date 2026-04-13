from pathlib import Path
import csv
import json
import sys
import types
import numpy as np

from vicon_ws import traj_tool


def _write_tum(path: Path, x_offset: float = 0.0, t_offset: float = 0.0) -> None:
    lines = [
        f"{0.0 + t_offset} {0.0 + x_offset} 0.0 0.0 0.0 0.0 0.0 1.0",
        f"{1.0 + t_offset} {1.0 + x_offset} 0.0 0.0 0.0 0.0 0.0 1.0",
        f"{2.0 + t_offset} {2.0 + x_offset} 0.0 0.0 0.0 0.0 0.0 1.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_spec_with_topic() -> None:
    path, topic = traj_tool._parse_spec("/tmp/a.bag::/odom", default_topic="")
    assert str(path).endswith("/tmp/a.bag")
    assert topic == "/odom"


def test_traj_tool_run_plot_table_and_export(tmp_path: Path) -> None:
    t1 = tmp_path / "a.tum"
    t2 = tmp_path / "b.tum"
    _write_tum(t1, x_offset=0.0)
    _write_tum(t2, x_offset=10.0)

    out_dir = tmp_path / "out"
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--plot",
            "--plot-mode",
            "xy",
            "--save-as",
            "kitti",
            "--out-dir",
            str(out_dir),
            str(t1),
            str(t2),
        ]
    )
    ret = traj_tool.run(args)
    assert ret == 0
    assert (out_dir / "traj_plot.png").exists()
    assert (out_dir / "traj_summary.csv").exists()
    assert (out_dir / "a.kitti").exists()
    assert (out_dir / "b.kitti").exists()


def test_load_traj_auto_prefers_tum_for_text_with_extra_columns(tmp_path: Path) -> None:
    p = tmp_path / "mixed.txt"
    p.write_text(
        "\n".join(
            [
                "0.0 0 0 0 0 0 0 1 100",
                "1.0 1 0 0 0 0 0 1 200",
                "2.0 2 0 0 0 0 0 1 300",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    t, pos, quat = traj_tool._load_traj(p, fmt="auto", topic="")
    assert t.shape[0] == 3
    assert float(pos[-1, 0]) == 2.0
    assert float(quat[0, 3]) == 1.0


def test_traj_tool_sync_with_reference(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=10.0, t_offset=0.01)

    out_dir = tmp_path / "sync_out"
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--sync",
            "--ref",
            "1",
            "--sync-max-diff",
            "0.05",
            "--out-dir",
            str(out_dir),
            str(ref),
            str(est),
        ]
    )
    ret = traj_tool.run(args)
    assert ret == 0

    table = out_dir / "traj_summary.csv"
    assert table.exists()
    with table.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    est_row = rows[1]
    assert est_row["synced"] == "true"
    assert est_row["ref_label"] == rows[0]["label"]
    assert int(est_row["matched_samples"]) == 3


def test_traj_tool_downsample_and_merge(tmp_path: Path) -> None:
    t1 = tmp_path / "a.tum"
    t2 = tmp_path / "b.tum"
    _write_tum(t1, x_offset=0.0)
    _write_tum(t2, x_offset=10.0)

    out_dir = tmp_path / "out_merge"
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--downsample",
            "2",
            "--merge",
            "--out-dir",
            str(out_dir),
            str(t1),
            str(t2),
        ]
    )
    ret = traj_tool.run(args)
    assert ret == 0
    table = out_dir / "traj_summary.csv"
    with table.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["label"] == "merged_trajectory"


def test_traj_tool_transform_left_applies_translation(tmp_path: Path) -> None:
    t1 = tmp_path / "a.tum"
    _write_tum(t1, x_offset=0.0)
    tf = tmp_path / "tf.json"
    tf.write_text(
        json.dumps(
            {
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_tf"
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--transform-left",
            str(tf),
            "--save-as",
            "tum",
            "--out-dir",
            str(out_dir),
            str(t1),
        ]
    )
    ret = traj_tool.run(args)
    assert ret == 0
    exported = out_dir / "a.tum"
    arr = np.loadtxt(exported)
    if arr.ndim == 1:
        arr = arr[None, :]
    assert float(arr[0, 1]) == 1.0


def test_traj_tool_motion_filter_reduces_samples(tmp_path: Path) -> None:
    t1 = tmp_path / "a.tum"
    _write_tum(t1, x_offset=0.0)
    out_dir = tmp_path / "out_filter"
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--motion-filter",
            "5.0",
            "180.0",
            "--out-dir",
            str(out_dir),
            str(t1),
        ]
    )
    ret = traj_tool.run(args)
    assert ret == 0
    table = out_dir / "traj_summary.csv"
    with table.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert int(rows[0]["samples"]) <= 2


def test_traj_tool_plot_mode_yx_and_ros_map_overlay(tmp_path: Path) -> None:
    t1 = tmp_path / "a.tum"
    _write_tum(t1, x_offset=0.0)
    img = np.zeros((4, 5), dtype=float)
    img_path = tmp_path / "map.png"
    import matplotlib.pyplot as plt

    plt.imsave(img_path, img, cmap="gray")
    yaml_path = tmp_path / "map.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "image: map.png",
                "resolution: 1.0",
                "origin: [0.0, 0.0, 0.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_map"
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--plot",
            "--plot-mode",
            "yx",
            "--out-dir",
            str(out_dir),
            str(t1),
        ]
    )
    assert traj_tool.run(args) == 0
    assert (out_dir / "traj_plot.png").exists()

    args_xy = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--plot",
            "--plot-mode",
            "xy",
            "--ros_map_yaml",
            str(yaml_path),
            "--out-dir",
            str(out_dir),
            str(t1),
        ]
    )
    assert traj_tool.run(args_xy) == 0
    assert (out_dir / "traj_plot.png").exists()


def test_traj_tool_map_tile_provider_overlay(tmp_path: Path, monkeypatch) -> None:
    t1 = tmp_path / "a.tum"
    _write_tum(t1, x_offset=0.0)
    calls: list[dict] = []
    fake_cx = types.SimpleNamespace(
        providers={"OpenStreetMap": {"Mapnik": "osm_mapnik"}},
        add_basemap=lambda _ax, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "contextily", fake_cx)

    out_dir = tmp_path / "out_tile"
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--plot",
            "--plot-mode",
            "xy",
            "--map_tile",
            "OpenStreetMap.Mapnik",
            "--out-dir",
            str(out_dir),
            str(t1),
        ]
    )
    assert traj_tool.run(args) == 0
    assert (out_dir / "traj_plot.png").exists()
    assert len(calls) == 1
    assert calls[0].get("source") == "osm_mapnik"


def test_traj_tool_evo_subcommand_and_save_as_flag(tmp_path: Path) -> None:
    t1 = tmp_path / "a.tum"
    _write_tum(t1, x_offset=0.0)
    out_dir = tmp_path / "out_subcommand"
    args = traj_tool.build_parser().parse_args(
        [
            "tum",
            str(t1),
            "--save_as_tum",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert traj_tool.run(args) == 0
    assert (out_dir / "a.tum").exists()


def test_traj_tool_project_to_plane_and_save_plot_prefix(tmp_path: Path) -> None:
    t1 = tmp_path / "a.tum"
    _write_tum(t1, x_offset=0.0)
    out_dir = tmp_path / "out_plane"
    plot_prefix = tmp_path / "plot_export.png"
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--project_to_plane",
            "yz",
            "--save_as_tum",
            "--plot",
            "--save_plot",
            str(plot_prefix),
            "--out-dir",
            str(out_dir),
            str(t1),
        ]
    )
    assert traj_tool.run(args) == 0
    arr = np.loadtxt(out_dir / "a.tum")
    if arr.ndim == 1:
        arr = arr[None, :]
    assert np.allclose(arr[:, 1], 0.0)
    assert (tmp_path / "plot_export_trajectories.png").exists()
    assert (tmp_path / "plot_export_xyz.png").exists()
    assert (tmp_path / "plot_export_rpy.png").exists()
    assert (tmp_path / "plot_export_speeds.png").exists()


def test_traj_tool_rerun_invokes_logger(tmp_path: Path, monkeypatch) -> None:
    t1 = tmp_path / "a.tum"
    _write_tum(t1, x_offset=0.0)
    called = {}

    def _fake_log(**kwargs):
        called["kwargs"] = kwargs
        return {"enabled": "true", "status": "ok", "message": "mock"}

    monkeypatch.setattr(traj_tool, "log_trajectories_to_rerun", _fake_log)
    args = traj_tool.build_parser().parse_args(
        [
            "--format",
            "tum",
            "--rerun",
            "--rerun-rec-id",
            "session-1",
            str(t1),
        ]
    )
    assert traj_tool.run(args) == 0
    assert called["kwargs"]["app_id"] == "vicon_ws_traj"
    assert called["kwargs"]["recording_id"] == "session-1"
