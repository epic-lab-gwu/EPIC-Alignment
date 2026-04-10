from argparse import Namespace
from pathlib import Path

from .bridge_legacy import run_legacy_pipeline
from .config import PipelineOptions, project_root_from_file
from .core.pipeline_modular import run_pipeline_modular


def run(ns: Namespace) -> int:
    opts = PipelineOptions(
        gt_csv=ns.gt_csv,
        est_path=ns.est_path or "",
        est_format=ns.est_format,
        dt_resample=ns.dt_resample,
        synthetic=bool(ns.synthetic),
        quat_interp=ns.quat_interp,
        rpe_delta=ns.rpe_delta,
        rpe_delta_unit=ns.rpe_delta_unit,
        rpe_delta_tol=ns.rpe_delta_tol,
        rpe_all_pairs=bool(ns.rpe_all_pairs),
        rpe_pairs_from_reference=bool(ns.rpe_pairs_from_reference),
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
