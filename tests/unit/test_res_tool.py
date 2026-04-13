from pathlib import Path
import json
import zipfile

import pytest

from vicon_ws.benchmark import res


def _payload(ape_val: float, rpe_val: float) -> dict:
    return {
        "pose_metrics": {
            "ape": {
                "step3": {
                    "translation_part": {"rmse": ape_val, "mean": ape_val},
                    "rotation_angle_deg": {"rmse": 1.0},
                }
            },
            "rpe": {
                "step3": {
                    "translation_part": {"rmse": rpe_val, "mean": rpe_val},
                    "rotation_angle_deg": {"rmse": 2.0},
                    "pair_count": 12,
                }
            },
        }
    }


def _payload_with_arrays(ape_vals, rpe_vals, title: str) -> dict:
    ape_vals = [float(x) for x in ape_vals]
    rpe_vals = [float(x) for x in rpe_vals]
    return {
        "info": {"title": title},
        "pose_metrics": {
            "ape": {
                "step3": {
                    "translation_part": {"rmse": float(sum(v * v for v in ape_vals) / len(ape_vals)) ** 0.5, "mean": float(sum(ape_vals) / len(ape_vals))},
                    "_error_arrays": {"translation_part": ape_vals},
                    "_x_axis": {"seconds_from_start": [float(i) for i in range(len(ape_vals))]},
                }
            },
            "rpe": {
                "step3": {
                    "translation_part": {"rmse": float(sum(v * v for v in rpe_vals) / len(rpe_vals)) ** 0.5, "mean": float(sum(rpe_vals) / len(rpe_vals))},
                    "pair_count": len(rpe_vals),
                    "_error_arrays": {"translation_part": rpe_vals},
                    "_x_axis": {"seconds_from_start": [float(i) for i in range(len(rpe_vals))]},
                }
            },
        },
    }


def test_res_tool_compare_metrics_json_and_plot(tmp_path: Path) -> None:
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    a.mkdir()
    b.mkdir()
    (a / "metrics.json").write_text(json.dumps(_payload(0.2, 0.3)), encoding="utf-8")
    (b / "metrics.json").write_text(json.dumps(_payload(0.1, 0.25)), encoding="utf-8")

    out_dir = tmp_path / "out"
    args = res.build_parser().parse_args(
        [
            str(a / "metrics.json"),
            str(b / "metrics.json"),
            "--metric",
            "ape",
            "--stage",
            "step3",
            "--ape-relation",
            "trans_part",
            "--plot",
            "--out-dir",
            str(out_dir),
        ]
    )
    ret = res.run(args)
    assert ret == 0
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "ape_translation_part_step3" / "aggregated_stats.csv").exists()
    assert (out_dir / "ape_translation_part_step3" / "aggregated_raw.png").exists()


def test_res_tool_load_zip_bundle(tmp_path: Path) -> None:
    payload = _payload(0.3, 0.4)
    bundle = tmp_path / "demo.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("metrics.json", json.dumps(payload))
        zf.writestr("manifest.json", "{}")

    args = res.build_parser().parse_args(
        [
            str(bundle),
            "--metric",
            "rpe",
            "--rpe-relation",
            "trans_part",
            "--stage",
            "step3",
        ]
    )
    ret = res.run(args)
    assert ret == 0


def test_res_tool_merge_and_use_rel_time(tmp_path: Path) -> None:
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    a.mkdir()
    b.mkdir()
    (a / "metrics.json").write_text(
        json.dumps(_payload_with_arrays([0.2, 0.3], [0.1, 0.2], title="same-title")),
        encoding="utf-8",
    )
    (b / "metrics.json").write_text(
        json.dumps(_payload_with_arrays([0.4, 0.5], [0.3, 0.4], title="same-title")),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_merge"
    args = res.build_parser().parse_args(
        [
            str(a / "metrics.json"),
            str(b / "metrics.json"),
            "--metric",
            "ape",
            "--stage",
            "step3",
            "--ape-relation",
            "trans_part",
            "--merge",
            "--use-rel-time",
            "--plot",
            "--plot-markers",
            "--out-dir",
            str(out_dir),
        ]
    )
    ret = res.run(args)
    assert ret == 0
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "ape_translation_part_step3" / "aggregated_raw.png").exists()


def test_res_tool_title_mismatch_requires_ignore_title(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(_payload_with_arrays([0.1, 0.2], [0.1, 0.2], title="title-a")),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps(_payload_with_arrays([0.2, 0.3], [0.2, 0.3], title="title-b")),
        encoding="utf-8",
    )
    args = res.build_parser().parse_args(
        [
            str(a),
            str(b),
            "--metric",
            "ape",
            "--stage",
            "step3",
            "--ape-relation",
            "trans_part",
            "--plot",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(ValueError, match="Mismatching titles"):
        res.run(args)

    args_ok = res.build_parser().parse_args(
        [
            str(a),
            str(b),
            "--metric",
            "ape",
            "--stage",
            "step3",
            "--ape-relation",
            "trans_part",
            "--plot",
            "--ignore-title",
            "--out-dir",
            str(tmp_path / "out_ok"),
        ]
    )
    assert res.run(args_ok) == 0


def test_res_tool_save_plot_exports(tmp_path: Path) -> None:
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    a.mkdir()
    b.mkdir()
    (a / "metrics.json").write_text(
        json.dumps(_payload_with_arrays([0.2, 0.3], [0.1, 0.2], title="same-title")),
        encoding="utf-8",
    )
    (b / "metrics.json").write_text(
        json.dumps(_payload_with_arrays([0.4, 0.5], [0.3, 0.4], title="same-title")),
        encoding="utf-8",
    )
    args = res.build_parser().parse_args(
        [
            str(a / "metrics.json"),
            str(b / "metrics.json"),
            "--metric",
            "ape",
            "--stage",
            "step3",
            "--ape-relation",
            "trans_part",
            "--save_plot",
            str(tmp_path / "plots.png"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert res.run(args) == 0
    exported = sorted(tmp_path.glob("plots_*.png"))
    assert len(exported) >= 2


def test_res_tool_title_mismatch_no_warnings_can_continue(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps(_payload_with_arrays([0.1, 0.2], [0.1, 0.2], title="title-a")),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps(_payload_with_arrays([0.2, 0.3], [0.2, 0.3], title="title-b")),
        encoding="utf-8",
    )
    args = res.build_parser().parse_args(
        [
            str(a),
            str(b),
            "--metric",
            "ape",
            "--stage",
            "step3",
            "--ape-relation",
            "trans_part",
            "--plot",
            "--no_warnings",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert res.run(args) == 0
