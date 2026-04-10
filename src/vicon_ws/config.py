from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PipelineOptions:
    gt_csv: str = "gt.csv"
    est_path: str = ""
    est_format: str = "auto"
    dt_resample: float = 0.001
    synthetic: bool = False
    quat_interp: str = "linear"
    rpe_delta: float = 1.0
    rpe_delta_unit: str = "f"
    rpe_delta_tol: float = 0.1
    rpe_all_pairs: bool = False
    rpe_pairs_from_reference: bool = False

    def to_legacy_argv(self) -> List[str]:
        argv = [
            "--gt-csv",
            self.gt_csv,
            "--est-format",
            self.est_format,
            "--dt-resample",
            str(self.dt_resample),
            "--quat-interp",
            self.quat_interp,
            "--rpe-delta",
            str(self.rpe_delta),
            "--rpe-delta-unit",
            self.rpe_delta_unit,
            "--rpe-delta-tol",
            str(self.rpe_delta_tol),
        ]
        if self.est_path:
            argv.extend(["--est-path", self.est_path])
        if self.synthetic:
            argv.append("--synthetic")
        if self.rpe_all_pairs:
            argv.append("--rpe-all-pairs")
        if self.rpe_pairs_from_reference:
            argv.append("--rpe-pairs-from-reference")
        return argv


def project_root_from_file(file_path: Path) -> Path:
    # src/vicon_ws/config.py -> repo root
    return file_path.resolve().parents[2]
