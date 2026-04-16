from epa import ipython_tool


def test_build_user_namespace_keys() -> None:
    ns = ipython_tool.build_user_namespace()
    required = {
        "np",
        "plt",
        "Path",
        "calibration",
        "evaluation",
        "io_utils",
        "math_utils",
        "time_alignment",
        "pipeline_modular",
        "ape_tool",
        "rpe_tool",
        "traj_tool",
        "res_tool",
        "metrics_res_tool",
        "plot_summary_tool",
    }
    assert required.issubset(set(ns.keys()))


def test_run_list_mode(capsys) -> None:
    parser = ipython_tool.build_parser()
    args = parser.parse_args(["--list"])
    assert ipython_tool.run(args, ipython_args=[]) == 0
    out = capsys.readouterr().out
    assert "np" in out
    assert "ape_tool" in out

