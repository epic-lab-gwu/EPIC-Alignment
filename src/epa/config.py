from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PipelineOptions:
    gt_csv: str = ""
    gt_format: str = "auto"
    gt_topic: str = ""
    est_path: str = ""
    est_format: str = "auto"
    est_topic: str = ""
    dt_resample: float = 0.001
    synthetic: bool = False
    quat_interp: str = "linear"
    rpe_delta: float = 1.0
    rpe_delta_unit: str = "f"
    rpe_delta_tol: float = 0.1
    rpe_all_pairs: bool = False
    rpe_pairs_from_reference: bool = False
    t_max_diff: float = 0.02
    t_offset: float = 0.0
    t_start: float | None = None
    t_end: float | None = None
    eval_align: str = "none"
    eval_n_to_align: int = -1
    eval_project_to_plane: str = "none"
    ape_pose_relation: str = "trans_part"
    rpe_pose_relation: str = "trans_part"
    plot: bool = True
    plot_x_dimension: str = "seconds"
    plot_ape_relation: str = "translation_part"
    plot_rpe_relation: str = "translation_part"
    debug: bool = False
    save_results: str = ""
    downsample_hz: float = 100.0
    no_downsample: bool = False

    def to_legacy_argv(self) -> List[str]:
        argv = [
            "--gt-csv",
            self.gt_csv,
            "--gt-format",
            self.gt_format,
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
            "--t-max-diff",
            str(self.t_max_diff),
            "--t-offset",
            str(self.t_offset),
            "--eval-align",
            self.eval_align,
            "--eval-n-to-align",
            str(self.eval_n_to_align),
            "--eval-project-to-plane",
            self.eval_project_to_plane,
            "--ape-pose-relation",
            self.ape_pose_relation,
            "--rpe-pose-relation",
            self.rpe_pose_relation,
            "--plot-x-dimension",
            self.plot_x_dimension,
            "--plot-ape-relation",
            self.plot_ape_relation,
            "--plot-rpe-relation",
            self.plot_rpe_relation,
            "--downsample-hz",
            str(self.downsample_hz),
        ]
        if self.save_results:
            argv.extend(["--save-results", self.save_results])
        if self.gt_topic:
            argv.extend(["--gt-topic", self.gt_topic])
        if self.est_path:
            argv.extend(["--est-path", self.est_path])
        if self.est_topic:
            argv.extend(["--est-topic", self.est_topic])
        if self.synthetic:
            argv.append("--synthetic")
        if self.rpe_all_pairs:
            argv.append("--rpe-all-pairs")
        if self.rpe_pairs_from_reference:
            argv.append("--rpe-pairs-from-reference")
        if self.no_downsample:
            argv.append("--no-downsample")
        if self.t_start is not None:
            argv.extend(["--t-start", str(self.t_start)])
        if self.t_end is not None:
            argv.extend(["--t-end", str(self.t_end)])
        if self.plot:
            argv.append("--plot")
        else:
            argv.append("--no-plot")
        if self.debug:
            argv.append("--debug")
        return argv


def project_root_from_file(file_path: Path) -> Path:
    # src/epa/config.py -> repo root
    return file_path.resolve().parents[2]
