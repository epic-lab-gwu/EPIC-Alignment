from argparse import Namespace
from pathlib import Path

from .core.pipeline_modular import run_pipeline_modular


def run(ns: Namespace) -> int:
    invocation_root = Path.cwd().resolve()
    if bool(getattr(ns, "dry_run", False)):
        print("Dry run command:")
        print(
            "python -m epa.cli "
            f"--gt-csv {getattr(ns, 'gt_csv', 'example_data/example_groundtruth.csv')} "
            f"--est-format {getattr(ns, 'est_format', 'auto')} ..."
        )
        return 0

    run_pipeline_modular(ns, script_dir=invocation_root)
    return 0
