from argparse import Namespace
from pathlib import Path

from .bridge_legacy import run_legacy_pipeline
from .config import PipelineOptions, project_root_from_file
from .core.pipeline_modular import run_pipeline_modular


def run(ns: Namespace) -> int:
    opts = PipelineOptions(
        gt_csv=getattr(ns, "gt_csv", "gt.csv"),
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
    )
    repo_root = project_root_from_file(Path(__file__))
    engine = getattr(ns, "engine", "legacy")
    if engine == "modular":
        if bool(ns.dry_run):
            print("Dry run command:")
            print(
                "python -m vicon_ws.cli --engine modular "
                f"--gt-csv {opts.gt_csv} --est-format {opts.est_format} ..."
            )
            return 0
        run_pipeline_modular(ns, script_dir=repo_root)
        return 0
    return run_legacy_pipeline(
        legacy_argv=opts.to_legacy_argv(), repo_root=repo_root, dry_run=bool(ns.dry_run)
    )
