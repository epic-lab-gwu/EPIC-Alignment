from pathlib import Path
import json
import sys
import types
import numpy as np

from epa import ape_tool, rpe_tool
from epa.metric_cli_common import sim3_scale_guard


def _write_tum(path: Path, x_offset: float = 0.0, t_offset: float = 0.0) -> None:
    lines = [
        f"{0.0 + t_offset:.6f} {0.0 + x_offset:.6f} 0.0 0.0 0.0 0.0 0.0 1.0",
        f"{1.0 + t_offset:.6f} {1.0 + x_offset:.6f} 0.0 0.0 0.0 0.0 0.0 1.0",
        f"{2.0 + t_offset:.6f} {2.0 + x_offset:.6f} 0.0 0.0 0.0 0.0 0.0 1.0",
        f"{3.0 + t_offset:.6f} {3.0 + x_offset:.6f} 0.0 0.0 0.0 0.0 0.0 1.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ape_rpe_default_plot_mode_is_xyz() -> None:
    ape_args = ape_tool._build_parser().parse_args(["tum", "ref.tum", "est.tum"])
    rpe_args = rpe_tool._build_parser().parse_args(["tum", "ref.tum", "est.tum"])
    assert ape_args.plot_mode == "xyz"
    assert rpe_args.plot_mode == "xyz"


def test_ape_tool_tum_eval_and_plots(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)

    out_dir = tmp_path / "out_ape"
    parser = ape_tool._build_parser()
    args = parser.parse_args(
        [
            "--pose_relation",
            "trans_part",
            "--plot",
            "--out_dir",
            str(out_dir),
            "tum",
            str(ref),
            str(est),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    ret = ape_tool.run(args)
    assert ret == 0
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "ape_raw.png").exists()
    assert (out_dir / "ape_map.png").exists()

    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    rmse = float(payload["pose_metrics"]["ape"]["raw"]["translation_part"]["rmse"])
    assert rmse > 0.5


def test_sim3_scale_guard_marks_near_zero_scale_unreliable() -> None:
    info = sim3_scale_guard(0.01)
    assert info["sim3_scale_warning"] is True
    assert info["sim3_scale_severe"] is True
    assert info["sim3_reliable"] is False


def test_ape_tool_sim3_writes_scale_reliability(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    ref.write_text(
        "\n".join(
            [
                "0 0 0 0 0 0 0 1",
                "1 1 0 0 0 0 0 1",
                "2 2 0 0 0 0 0 1",
                "3 3 0 0 0 0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    est.write_text(
        "\n".join(
            [
                "0 0 0 0 0 0 0 1",
                "1 100 0 0 0 0 0 1",
                "2 200 0 0 0 0 0 1",
                "3 300 0 0 0 0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_sim3"
    args = ape_tool._build_parser().parse_args(
        [
            "--align",
            "--correct_scale",
            "--out_dir",
            str(out_dir),
            "tum",
            str(ref),
            str(est),
        ]
    )

    assert ape_tool.run(args) == 0

    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    info = payload["metadata"]["eval_alignment"]
    assert info["align_mode"] == "sim3"
    assert info["sim3_reliable"] is False
    assert info["sim3_scale_severe"] is True
    assert info["sim3_scale"] == 0.01


def test_ape_and_rpe_tools_accept_explicit_eval_align_posyaw(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.0)

    ape_out = tmp_path / "ape_posyaw"
    ape_args = ape_tool._build_parser().parse_args(
        ["--eval-align", "posyaw", "--out_dir", str(ape_out), "tum", str(ref), str(est)]
    )
    assert ape_tool.run(ape_args) == 0
    ape_payload = json.loads((ape_out / "metrics.json").read_text(encoding="utf-8"))
    assert ape_payload["metadata"]["eval_alignment"]["align_mode"] == "posyaw"

    rpe_out = tmp_path / "rpe_posyaw"
    rpe_args = rpe_tool._build_parser().parse_args(
        ["--eval-align", "posyaw", "--out_dir", str(rpe_out), "tum", str(ref), str(est)]
    )
    assert rpe_tool.run(rpe_args) == 0
    rpe_payload = json.loads((rpe_out / "metrics.json").read_text(encoding="utf-8"))
    assert rpe_payload["metadata"]["eval_alignment"]["align_mode"] == "posyaw"


def test_ape_tool_accepts_options_after_subcommand(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)
    out_dir = tmp_path / "out_ape_order"
    parser = ape_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    ret = ape_tool.run(args)
    assert ret == 0
    assert (out_dir / "metrics.json").exists()


def test_ape_tool_change_unit_and_plot_full_ref(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)
    out_dir = tmp_path / "out_ape_unit"
    parser = ape_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--change_unit",
            "cm",
            "--plot",
            "--plot_full_ref",
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    assert ape_tool.run(args) == 0
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    rmse = float(payload["pose_metrics"]["ape"]["raw"]["translation_part"]["rmse"])
    assert rmse > 50.0


def test_rpe_tool_tum_eval(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)

    out_dir = tmp_path / "out_rpe"
    parser = rpe_tool._build_parser()
    args = parser.parse_args(
        [
            "--pose_relation",
            "trans_part",
            "--delta",
            "1",
            "--delta_unit",
            "f",
            "--out_dir",
            str(out_dir),
            "tum",
            str(ref),
            str(est),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    ret = rpe_tool.run(args)
    assert ret == 0
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    pair_count = int(payload["pose_metrics"]["rpe"]["raw"]["pair_count"])
    rmse = float(payload["pose_metrics"]["rpe"]["raw"]["translation_part"]["rmse"])
    assert pair_count > 0
    assert rmse < 1e-9


def test_rpe_tool_tum_eval_seconds_delta(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=0.0, t_offset=0.0)

    out_dir = tmp_path / "out_rpe_seconds"
    parser = rpe_tool._build_parser()
    args = parser.parse_args(
        [
            "--pose_relation",
            "trans_part",
            "--delta",
            "1",
            "--delta_unit",
            "s",
            "--all_pairs",
            "--out_dir",
            str(out_dir),
            "tum",
            str(ref),
            str(est),
            "--t_max_diff",
            "0.05",
        ]
    )

    assert rpe_tool.run(args) == 0
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["pose_metrics"]["rpe_config"]["delta_unit"] == "s"
    assert int(payload["pose_metrics"]["rpe"]["raw"]["pair_count"]) == 3


def test_rpe_tool_accepts_options_after_subcommand(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)
    out_dir = tmp_path / "out_rpe_order"
    parser = rpe_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--delta",
            "1",
            "--delta_unit",
            "f",
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    ret = rpe_tool.run(args)
    assert ret == 0
    assert (out_dir / "metrics.json").exists()


def test_rpe_tool_change_unit_and_downsample(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)
    out_dir = tmp_path / "out_rpe_unit"
    parser = rpe_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "angle_deg",
            "--change_unit",
            "rad",
            "--downsample",
            "3",
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    assert rpe_tool.run(args) == 0
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "raw" in payload["pose_metrics"]["rpe"]


def test_ape_tool_ros_map_overlay_xy(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)

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

    out_dir = tmp_path / "out_ape_map"
    parser = ape_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--plot",
            "--plot_mode",
            "xy",
            "--ros_map_yaml",
            str(yaml_path),
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    assert ape_tool.run(args) == 0
    assert (out_dir / "ape_map.png").exists()


def test_ape_tool_map_tile_provider_overlay(tmp_path: Path, monkeypatch) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)

    calls: list[dict] = []
    fake_cx = types.SimpleNamespace(
        providers={"OpenStreetMap": {"Mapnik": "osm_mapnik"}},
        add_basemap=lambda _ax, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "contextily", fake_cx)

    out_dir = tmp_path / "out_ape_tile"
    parser = ape_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--plot",
            "--plot_mode",
            "xy",
            "--map_tile",
            "OpenStreetMap.Mapnik",
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    assert ape_tool.run(args) == 0
    assert (out_dir / "ape_map.png").exists()
    assert len(calls) == 1
    assert calls[0].get("source") == "osm_mapnik"


def test_rpe_tool_map_tile_epsg_overlay(tmp_path: Path, monkeypatch) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)

    calls: list[dict] = []
    fake_cx = types.SimpleNamespace(
        providers={},
        add_basemap=lambda _ax, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "contextily", fake_cx)

    out_dir = tmp_path / "out_rpe_tile"
    parser = rpe_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--delta",
            "1",
            "--delta_unit",
            "f",
            "--plot",
            "--plot_mode",
            "xy",
            "--map_tile",
            "EPSG:3857",
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    assert rpe_tool.run(args) == 0
    assert (out_dir / "rpe_map.png").exists()
    assert len(calls) == 1
    assert calls[0].get("crs") == "EPSG:3857"


def test_ape_tool_rerun_invokes_logger(tmp_path: Path, monkeypatch) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)
    called = {}

    def _fake_log(**kwargs):
        called["kwargs"] = kwargs
        return {"enabled": "true", "status": "ok", "message": "mock"}

    monkeypatch.setattr(ape_tool, "log_metric_to_rerun", _fake_log)
    out_dir = tmp_path / "out_ape_rerun"
    parser = ape_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--rerun",
            "--rerun-rec-id",
            "ape-session",
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    assert ape_tool.run(args) == 0
    assert called["kwargs"]["app_id"] == "epa_ape"
    assert called["kwargs"]["recording_id"] == "ape-session"


def test_rpe_tool_rerun_invokes_logger(tmp_path: Path, monkeypatch) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)
    called = {}

    def _fake_log(**kwargs):
        called["kwargs"] = kwargs
        return {"enabled": "true", "status": "ok", "message": "mock"}

    monkeypatch.setattr(rpe_tool, "log_metric_to_rerun", _fake_log)
    out_dir = tmp_path / "out_rpe_rerun"
    parser = rpe_tool._build_parser()
    args = parser.parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--delta",
            "1",
            "--delta_unit",
            "f",
            "--rerun",
            "--rerun-rec-id",
            "rpe-session",
            "--out_dir",
            str(out_dir),
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
        ]
    )
    assert rpe_tool.run(args) == 0
    assert called["kwargs"]["app_id"] == "epa_rpe"
    assert called["kwargs"]["recording_id"] == "rpe-session"
