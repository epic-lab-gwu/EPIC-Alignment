from argparse import Namespace
from pathlib import Path

from .core.pipeline_modular import run_pipeline_modular


def run(ns: Namespace) -> int:
    invocation_root = Path.cwd().resolve()
    if bool(getattr(ns, "dry_run", False)):
        print("Dry run command:")
        print(
            "python -m epa.cli "
            f"{getattr(ns, 'gt_csv', '<gt_file>')} "
            f"{getattr(ns, 'est_path', '<est_file>')} "
            f"--gt-format {getattr(ns, 'gt_format', 'auto')} "
            f"--est-format {getattr(ns, 'est_format', 'auto')} ..."
        )
        return 0

    run_pipeline_modular(ns, script_dir=invocation_root)
    return 0
