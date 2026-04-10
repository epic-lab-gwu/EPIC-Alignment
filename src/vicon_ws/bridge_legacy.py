import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import project_root_from_file


def run_legacy_pipeline(
    legacy_argv: List[str], repo_root: Optional[Path] = None, dry_run: bool = False
) -> int:
    root = repo_root or project_root_from_file(Path(__file__))
    pipeline_py = root / "pipeline.py"
    if not pipeline_py.exists():
        raise FileNotFoundError(f"Legacy pipeline not found: {pipeline_py}")

    cmd = [sys.executable, str(pipeline_py)] + legacy_argv
    if dry_run:
        print("Dry run command:")
        print(" ".join(cmd))
        return 0

    completed = subprocess.run(cmd, cwd=str(root), check=False)
    return int(completed.returncode)

