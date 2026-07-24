from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Timestamped SE(3) trajectory represented in EPA's normalized array form."""

    t: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    name: str = ""


@dataclass(frozen=True)
class TimeAlignmentResult:
    calculated_offset: float
    time_metrics: dict[str, Any]
    step1_forced_candidate: bool = False
    step1_force_reason: str = ""


@dataclass(frozen=True)
class SolveEvalTrajectory:
    t_est_sync: np.ndarray
    t_gt: np.ndarray
    pos_gt: np.ndarray
    quat_gt: np.ndarray
    pr_sync: np.ndarray
    qr_sync: np.ndarray
    pos_gt_solve: np.ndarray
    quat_gt_solve: np.ndarray
    pr_solve: np.ndarray
    qr_solve: np.ndarray
    overlap_info: dict[str, Any] = field(default_factory=dict)
    downsample_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryAlignmentResult:
    R_calc: np.ndarray
    t_calc: np.ndarray
    Rw_calc: np.ndarray
    tw_calc: np.ndarray
    pr_corrected: np.ndarray
    pr_final: np.ndarray
    q_step2: np.ndarray
    q_step3: np.ndarray
    alignment_selection: dict[str, Any]


@dataclass(frozen=True)
class PipelineResult:
    out_dir: Path
    metrics: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)
