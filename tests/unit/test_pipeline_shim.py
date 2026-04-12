from pathlib import Path

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
