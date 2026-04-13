from pathlib import Path
import json

import pipeline


def test_pipeline_shim_delegates_to_modular_runner(monkeypatch) -> None:
    calls = {}

    def fake_runner(*, args, script_dir):
        calls["args"] = args
        calls["script_dir"] = script_dir

    monkeypatch.setattr(pipeline, "_load_modular_runner", lambda: fake_runner)

    args = pipeline.parse_args(["--synthetic"])
    pipeline.run_pipeline(args)

    assert calls["args"] is args
    assert calls["script_dir"] == Path(pipeline.__file__).resolve().parent


def test_pipeline_parser_keeps_legacy_flags() -> None:
    args = pipeline.parse_args([
        "--rpe-delta",
        "2",
        "--rpe-delta-unit",
        "m",
        "--rpe-all-pairs",
    ])

    assert args.rpe_delta == 2.0
    assert args.rpe_delta_unit == "m"
    assert args.rpe_all_pairs is True


def test_pipeline_parser_accepts_extended_formats() -> None:
    args = pipeline.parse_args([
        "--gt-format",
        "bag2",
        "--gt-topic",
        "/vicon/pose",
        "--est-format",
        "kitti",
    ])

    assert args.gt_format == "bag2"
    assert args.gt_topic == "/vicon/pose"
    assert args.est_format == "kitti"


def test_pipeline_parser_accepts_plot_eval_options() -> None:
    args = pipeline.parse_args([
        "--eval-align",
        "se3",
        "--eval-project-to-plane",
        "xy",
        "--t-max-diff",
        "0.03",
        "--t-offset",
        "0.2",
        "--ape-pose-relation",
        "angle_deg",
        "--rpe-pose-relation",
        "point_distance_error_ratio",
        "--no-plot",
    ])

    assert args.eval_align == "se3"
    assert args.eval_project_to_plane == "xy"
    assert args.t_max_diff == 0.03
    assert args.t_offset == 0.2
    assert args.ape_pose_relation == "angle_deg"
    assert args.rpe_pose_relation == "point_distance_error_ratio"
    assert args.plot is False


def test_pipeline_parser_accepts_plot_options() -> None:
    args = pipeline.parse_args([
        "--plot-x-dimension",
        "index",
        "--plot-ape-relation",
        "rotation_angle_deg",
        "--plot-rpe-relation",
        "rotation_angle_deg",
        "--no-plot",
    ])
    assert args.plot_x_dimension == "index"
    assert args.plot_ape_relation == "rotation_angle_deg"
    assert args.plot_rpe_relation == "rotation_angle_deg"
    assert args.plot is False


def test_pipeline_parser_accepts_save_results() -> None:
    args = pipeline.parse_args(["--save-results", "outputs/results/demo.zip"])
    assert args.save_results == "outputs/results/demo.zip"


def test_pipeline_parser_config_overrides_cli(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "plot": False,
                "rpe_delta": 3.0,
            }
        ),
        encoding="utf-8",
    )
    args = pipeline.parse_args(["--plot", "--rpe-delta", "1", "--config", str(cfg)])
    assert args.plot is False
    assert args.rpe_delta == 3.0
