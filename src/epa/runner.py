from argparse import Namespace
from pathlib import Path

from .config import PipelineOptions
from .core.pipeline_modular import run_pipeline_modular


def run(ns: Namespace) -> int:
    opts = PipelineOptions(
        gt_csv=getattr(ns, "gt_csv", "example_data/example_groundtruth.csv"),
        gt_format=getattr(ns, "gt_format", "csv"),
        gt_topic=getattr(ns, "gt_topic", "") or "",
        est_path=getattr(ns, "est_path", "") or "",
        est_format=getattr(ns, "est_format", "auto"),
        est_topic=getattr(ns, "est_topic", "") or "",
        dt_resample=getattr(ns, "dt_resample", 0.001),
        synthetic=bool(getattr(ns, "synthetic", False)),
        quat_interp=getattr(ns, "quat_interp", "linear"),
        rpe_delta=getattr(ns, "rpe_delta", 1.0),
        rpe_delta_unit=getattr(ns, "rpe_delta_unit", "f"),
        rpe_delta_tol=getattr(ns, "rpe_delta_tol", 0.1),
        rpe_all_pairs=bool(getattr(ns, "rpe_all_pairs", False)),
        rpe_pairs_from_reference=bool(getattr(ns, "rpe_pairs_from_reference", False)),
        t_max_diff=getattr(ns, "t_max_diff", 0.02),
        t_offset=getattr(ns, "t_offset", 0.0),
        t_start=getattr(ns, "t_start", None),
        t_end=getattr(ns, "t_end", None),
        eval_align=getattr(ns, "eval_align", "none"),
        eval_n_to_align=getattr(ns, "eval_n_to_align", -1),
        eval_project_to_plane=getattr(ns, "eval_project_to_plane", "none"),
        ape_pose_relation=getattr(ns, "ape_pose_relation", "trans_part"),
        rpe_pose_relation=getattr(ns, "rpe_pose_relation", "trans_part"),
        plot=bool(getattr(ns, "plot", True)),
        plot_x_dimension=getattr(ns, "plot_x_dimension", "seconds"),
        plot_ape_relation=getattr(ns, "plot_ape_relation", "translation_part"),
        plot_rpe_relation=getattr(ns, "plot_rpe_relation", "translation_part"),
        save_results=getattr(ns, "save_results", "") or "",
    )
    invocation_root = Path.cwd().resolve()
    if bool(getattr(ns, "dry_run", False)):
        print("Dry run command:")
        print(
            "python -m epa.cli "
            f"--gt-csv {opts.gt_csv} --est-format {opts.est_format} ..."
        )
        return 0

    run_pipeline_modular(ns, script_dir=invocation_root)
    return 0
