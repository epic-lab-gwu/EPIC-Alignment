from pathlib import Path

from epa import ape_tool, fig_tool, rpe_tool
from epa.viz.plot_bundle import load_plot_bundle


def _write_tum(path: Path, x_offset: float = 0.0, t_offset: float = 0.0) -> None:
    lines = [
        f"{0.0 + t_offset:.6f} {0.0 + x_offset:.6f} 0.0 0.0 0.0 0.0 0.0 1.0",
        f"{1.0 + t_offset:.6f} {1.0 + x_offset:.6f} 0.0 0.0 0.0 0.0 0.0 1.0",
        f"{2.0 + t_offset:.6f} {2.0 + x_offset:.6f} 0.0 0.0 0.0 0.0 0.0 1.0",
        f"{3.0 + t_offset:.6f} {3.0 + x_offset:.6f} 0.0 0.0 0.0 0.0 0.0 1.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ape_serialize_plot_and_rerender(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)

    out_dir = tmp_path / "out_ape"
    bundle = tmp_path / "ape_figs.json"
    args = ape_tool._build_parser().parse_args(
        [
            "tum",
            str(ref),
            str(est),
            "--pose_relation",
            "trans_part",
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
            "--out_dir",
            str(out_dir),
            "--serialize_plot",
            str(bundle),
        ]
    )
    assert ape_tool.run(args) == 0
    assert bundle.exists()
    saved = load_plot_bundle(bundle)
    assert len(saved["figures"]) == 2
    raw = next(fig for fig in saved["figures"] if fig.get("name") == "raw")
    assert raw.get("line_label") == "APE (m)"
    assert isinstance(raw.get("stats"), dict)
    assert {"rmse", "mean", "median", "std"}.issubset(set(raw["stats"].keys()))

    rerender_dir = tmp_path / "rerender_ape"
    fig_args = fig_tool.build_parser().parse_args([str(bundle), "--out_dir", str(rerender_dir)])
    assert fig_tool.run(fig_args) == 0
    assert (rerender_dir / "raw.png").exists()
    assert (rerender_dir / "map.png").exists()


def test_rpe_serialize_plot_and_rerender(tmp_path: Path) -> None:
    ref = tmp_path / "ref.tum"
    est = tmp_path / "est.tum"
    _write_tum(ref, x_offset=0.0, t_offset=0.0)
    _write_tum(est, x_offset=1.0, t_offset=0.01)

    out_dir = tmp_path / "out_rpe"
    bundle = tmp_path / "rpe_figs.json"
    args = rpe_tool._build_parser().parse_args(
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
            "--t_max_diff",
            "0.05",
            "--t_offset",
            "-0.01",
            "--out_dir",
            str(out_dir),
            "--serialize_plot",
            str(bundle),
        ]
    )
    assert rpe_tool.run(args) == 0
    assert bundle.exists()

    save_plot = tmp_path / "rerender.png"
    fig_args = fig_tool.build_parser().parse_args([str(bundle), "--save_plot", str(save_plot)])
    assert fig_tool.run(fig_args) == 0
    assert (tmp_path / "rerender_raw.png").exists()
    assert (tmp_path / "rerender_map.png").exists()
