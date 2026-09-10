#!/usr/bin/env python3
"""Run calibration sensitivity for SE(3)-original and SE3R, with and without calibration.

Scientific question
-------------------
When timestamp/extrinsic perturbations are present, how strongly do trajectory-
evaluation errors depend on motion profile, simulated VIO accuracy, perturbation
magnitude, global-alignment method, and whether EPA calibration is enabled?

Experiment setup
----------------
1. Motion templates: eleven real ground-truth trajectories spanning Hot3D
   small egocentric motion, EuRoC indoor/aggressive flight, KITTI driving, and
   AEA exercise motion.
2. Preprocessing: timestamps start at zero; poses are expressed relative to the
   first pose; every trajectory is cropped to 80 s and resampled to 10 Hz using
   linear position interpolation and quaternion SLERP. No spatial scaling is
   applied. KITTI timestamps assume the dataset's 10 Hz rate.
3. Simulated VIO: a deterministic, smoothed SE(3) random walk is added to each
   GT trajectory. Translation drift is added in the common world frame and
   orientation drift is right-multiplied as Exp(delta_theta). The complete
   traces are normalized to four accuracy targets: oracle (0, 0 deg), high
   (0.1% S, 0.1 deg), medium (0.5% S, 0.5 deg), and low (2.0% S, 2.0 deg),
   where S=max(bounding-box diagonal, 0.1*path length). Five fixed seeds are
   reused across all calibration-error conditions in each matched block.
4. Injected perturbations: time offsets (1, 5, 10, 20, 30, 50 ms), extrinsic
   rotations (5, 20, 45 deg), extrinsic translations (0.05, 0.30, 0.50 m), one
   combined case (10 ms + 20 deg + 0.30 m), and a no-perturbation control.
5. Evaluation: every case is evaluated four ways: (a) SE(3)-original without
   calibration, (b) SE3R without calibration, (c) SE(3)-original with EPA
   calibration, and (d) SE3R with EPA calibration. The calibrated policies run
   the current EPA pipeline end to end, including timestamp and extrinsic
   estimation. APE is translational RMSE in metres; ARE is rotational RMSE in
   degrees.
6. Heatmap statistic: for each alignment mode, metric, motion group, accuracy,
   and perturbation, report the mean matched relative change over trajectories
   and seeds: 100*(error_condition-error_control)/error_control. Oracle rows are
   retained in source data but omitted from relative heatmaps because their
   denominators are numerically zero.

Outputs
-------
- figures/fig1_se3_original_without_calibration.{pdf,svg,png,tiff}
- figures/fig2_se3r_without_calibration.{pdf,svg,png,tiff}
- figures/fig3_se3_original_with_calibration.{pdf,svg,png,tiff}
- figures/fig4_se3r_with_calibration.{pdf,svg,png,tiff}
- data/case_metrics.csv
- data/heatmap_values.csv
- data/simulated_vio_rmse.csv       (Table 1 source data)
- setup.md, config.json, validation.json, and QA sidecars

The experiment is controlled simulation on real GT motion templates. It tests
error sensitivity and mitigation, not real-VIO generalization.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from scipy.spatial.transform import Rotation as R

import run_pilot as core
import run_relative_baseline as design


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "calibration_sensitivity_results"
FIGURE_AUDIT_SCRIPTS = ROOT

ACCURACIES = ("high", "medium", "low")
ACCURACY_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}
GROUP_ORDER = ("hot3d", "euroc", "kitti", "euroc_aggressive", "aea")
GROUP_LABELS = {
    "hot3d": "Hot3D / small",
    "euroc": "EuRoC / indoor",
    "kitti": "KITTI / driving",
    "euroc_aggressive": "EuRoC V2_03 / aggressive",
    "aea": "AEA / exercise",
}
DISPLAY_CONDITIONS = (
    "time_1ms",
    "time_10ms",
    "time_50ms",
    "rotation_5deg",
    "rotation_20deg",
    "rotation_45deg",
    "translation_0p05m",
    "translation_0p30m",
    "translation_0p50m",
    "combined_realistic",
)
CONDITION_LABELS = {
    "time_1ms": "T 1 ms",
    "time_10ms": "T 10 ms",
    "time_50ms": "T 50 ms",
    "rotation_5deg": "R 5 deg",
    "rotation_20deg": "R 20 deg",
    "rotation_45deg": "R 45 deg",
    "translation_0p05m": "X 0.05 m",
    "translation_0p30m": "X 0.30 m",
    "translation_0p50m": "X 0.50 m",
    "combined_realistic": "Combined",
}
MODES = (
    "se3_position",
    "se3r_rotation_first",
    "se3_position_calibrated",
    "se3r_rotation_first_calibrated",
)
NO_CALIBRATION_MODES = ("se3_position", "se3r_rotation_first")
CALIBRATION_MODES = {
    "se3_position_calibrated": "se3",
    "se3r_rotation_first_calibrated": "se3r",
}
NOISE_SEED_SCOPE = "profile_sequence_accuracy_seed"
MODE_LABELS = {
    "se3_position": "SE(3)-original without calibration",
    "se3r_rotation_first": "SE3R without calibration",
    "se3_position_calibrated": "SE(3)-original with calibration",
    "se3r_rotation_first_calibrated": "SE3R with calibration",
}
FIGURE_NAMES = {
    "se3_position": "fig1_se3_original_without_calibration",
    "se3r_rotation_first": "fig2_se3r_without_calibration",
    "se3_position_calibrated": "fig3_se3_original_with_calibration",
    "se3r_rotation_first_calibrated": "fig4_se3r_with_calibration",
}
METRICS = ("ape", "are")
METRIC_LABELS = {"ape": "Relative APE change", "are": "Relative ARE change"}

FIGURE_CONTRACT = {
    "core_conclusion": (
        "EPA calibration should reduce sensitivity to timestamp/extrinsic perturbations; "
        "the residual depends on motion, VIO accuracy, and global-alignment method."
    ),
    "results_level_question": (
        "Where do known perturbations change trajectory-evaluation error, and how much "
        "of that sensitivity remains after EPA calibration?"
    ),
    "archetype": "quantitative grid",
    "backend": "Python/matplotlib",
    "final_size": "183 mm x 211 mm per policy figure",
    "panels": {
        "a": "matched relative translation RMSE (APE)",
        "b": "matched relative orientation RMSE (ARE)",
    },
    "replicate_unit": "trajectory; five deterministic random-walk seeds per trajectory",
    "center": "mean over available trajectories and five seeds",
    "spread": "SD is retained in source data; not encoded in heatmap color",
    "reviewer_risks": [
        "Synthetic random-walk errors are not native VIO failure distributions.",
        "Hot3D/EuRoC/KITTI have three trajectories per group; aggressive EuRoC/AEA have one.",
        "KITTI timestamps use an explicit 10 Hz assumption.",
        "Relative metrics can be unstable near zero; oracle controls are excluded.",
        "Four figures must share one color normalization for visual comparability.",
    ],
}


def resolve_epa_calibration_modes(epa_repo: Path) -> dict[str, str]:
    """Map conceptual policies to the public mode names in this EPA checkout.

    EPA's main branch names the rotation-first policy ``se3`` and the legacy
    position-only policy ``se3-original``. The in-development naming cleanup
    uses ``se3r`` and ``se3`` respectively. Reading ``PUBLIC_ALIGN_MODES`` keeps
    this experiment reviewable and runnable on either scheme without guessing
    from a version number.
    """
    modes_path = epa_repo / "src" / "epa" / "alignment" / "modes.py"
    if not modes_path.is_file():
        raise FileNotFoundError(f"EPA alignment modes not found: {modes_path}")
    tree = ast.parse(modes_path.read_text(encoding="utf-8"), filename=str(modes_path))
    public_modes: tuple[str, ...] | None = None
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id == "PUBLIC_ALIGN_MODES" and node.value is not None:
            value = ast.literal_eval(node.value)
            public_modes = tuple(str(item) for item in value)
            break
    if public_modes is None:
        raise RuntimeError(f"Could not resolve PUBLIC_ALIGN_MODES from {modes_path}")
    if "se3r" in public_modes:
        return {
            "se3_position_calibrated": "se3",
            "se3r_rotation_first_calibrated": "se3r",
        }
    if "se3-original" in public_modes:
        return {
            "se3_position_calibrated": "se3-original",
            "se3r_rotation_first_calibrated": "se3",
        }
    raise RuntimeError(f"Unsupported EPA public alignment modes: {public_modes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four calibration-sensitivity heatmaps for SE(3)-original and SE3R.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--epa-repo",
        type=Path,
        default=ROOT.parents[1],
        help="EPA repository whose current calibration implementation is evaluated.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, max(1, os.cpu_count() or 1)),
        help="Concurrent EPA terminal-only subprocesses.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory. Without this flag, existing outputs are preserved.",
    )
    parser.add_argument(
        "--skip-tiff",
        action="store_true",
        help="Skip 600-dpi TIFF export; PDF, SVG, and PNG are still written.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful calibration cases in data/calibration_cache.jsonl.",
    )
    parser.add_argument(
        "--print-setup",
        action="store_true",
        help="Print the complete scientific setup and exit without running the experiment.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array))


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_group(profile: core.Profile) -> str:
    if profile.sequence == "V2_03_difficult":
        return "euroc_aggressive"
    if profile.sequence == "loc1111":
        return "aea"
    return {"small": "hot3d", "medium": "euroc", "large": "kitti"}[profile.name]


def evaluate_position_only_se3(
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    r_gt: R,
    t_est: np.ndarray,
    p_est: np.ndarray,
    r_est: R,
) -> tuple[int, float, float]:
    """Evaluate recorded-time poses with position-only Kabsch SE(3) alignment."""
    mask = (t_est >= t_gt[0]) & (t_est <= t_gt[-1])
    query = t_est[mask]
    if len(query) < 3:
        raise ValueError("Insufficient timestamp overlap for position-only SE(3)")
    p_ref, r_ref = core.interpolate_pose(t_gt, p_gt, r_gt, query)
    world_r, world_t = core.kabsch_se3(p_est[mask], p_ref)
    p_aligned = world_r.apply(p_est[mask]) + world_t
    r_aligned = world_r * r_est[mask]
    ape = float(np.sqrt(np.mean(np.sum(np.square(p_ref - p_aligned), axis=1))))
    are = float(
        np.sqrt(np.mean(np.square((r_ref.inv() * r_aligned).magnitude())))
        * 180.0
        / np.pi
    )
    return len(query), ape, are


def evaluate_rotation_first_se3r(
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    r_gt: R,
    t_est: np.ndarray,
    p_est: np.ndarray,
    r_est: R,
) -> tuple[int, float, float]:
    """Evaluate recorded-time poses with the current rotation-first SE3R alignment."""
    return design.calibration_off_rotation_first(t_gt, p_gt, r_gt, t_est, p_est, r_est)


EVALUATORS: dict[
    str,
    Callable[[np.ndarray, np.ndarray, R, np.ndarray, np.ndarray, R], tuple[int, float, float]],
] = {
    "se3_position": evaluate_position_only_se3,
    "se3r_rotation_first": evaluate_rotation_first_se3r,
}

EPA_WORKER_PATTERN = re.compile(r"^EPA_WORKER_JSON=(?P<payload>\{.*\})$", re.MULTILINE)
EPA_OFFSET_PATTERN = re.compile(r"Calculated Time Offset: (?P<offset>[-+0-9.eE]+) s")
EPA_ALIGNMENT_PATTERN = re.compile(
    r"Final alignment mode=(?P<mode>\S+) inliers=(?P<inliers>\d+) rejected=(?P<rejected>\d+)"
)


def run_epa_calibration_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one current EPA calibration policy and parse its terminal metrics."""
    started = time.perf_counter()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-epa-worker",
        "--gt-csv",
        task["gt_path"],
        "--gt-format",
        "tum",
        "--est-path",
        task["est_path"],
        "--est-format",
        "tum",
        "--mode",
        task["epa_mode"],
        "--dt-resample",
        "0.001",
        "--offset-search-window-s",
        "0.5",
        "--t-max-diff",
        "0.06",
        "--no-downsample",
        "--no-plot",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(task["epa_repo"]) / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    env["MPLBACKEND"] = "Agg"
    process = subprocess.run(
        command,
        cwd=task["epa_repo"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = process.stdout + "\n" + process.stderr
    worker_match = EPA_WORKER_PATTERN.search(combined)
    worker_payload = json.loads(worker_match.group("payload")) if worker_match else None
    offset_match = EPA_OFFSET_PATTERN.search(combined)
    alignment_match = EPA_ALIGNMENT_PATTERN.search(combined)
    result: dict[str, Any] = {
        "cache_key": task["cache_key"],
        "case_id": task["case_id"],
        "column_mode": task["column_mode"],
        "epa_mode": task["epa_mode"],
        "returncode": process.returncode,
        "runtime_s": time.perf_counter() - started,
    }
    if process.returncode != 0 or worker_payload is None or alignment_match is None:
        result.update(
            {
                "status": "failed",
                "error": combined.strip()[-3000:],
            }
        )
        return result
    result.update(
        {
            "status": "ok",
            "ape_rmse_m": float(worker_payload["ape_rmse_m"]),
            "are_rmse_deg": float(worker_payload["are_rmse_deg"]),
            "estimated_offset_ms": (
                1000.0 * float(offset_match.group("offset")) if offset_match else math.nan
            ),
            "step3_alignment_mode": alignment_match.group("mode"),
            "step3_inlier_count": int(alignment_match.group("inliers")),
            "step3_rejected_count": int(alignment_match.group("rejected")),
        }
    )
    return result


def read_calibration_cache(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return cached
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed cache line {line_number}: {path}") from exc
        if item.get("status") == "ok":
            cached[str(item["cache_key"])] = item
    return cached


def run_calibrated_policies(
    rows: list[dict[str, Any]],
    input_root: Path,
    epa_repo: Path,
    workers: int,
    cache_path: Path,
    resume: bool,
) -> None:
    """Populate calibration-on metrics in-place with resumable EPA evaluations."""
    if not (epa_repo / "src" / "epa" / "cli.py").is_file():
        raise FileNotFoundError(f"EPA CLI not found under {epa_repo}")
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    epa_fingerprint = hashlib.sha256(
        (
            sha256(epa_repo / "src" / "epa" / "cli.py")
            + sha256(epa_repo / "src" / "epa" / "core" / "trajectory_alignment.py")
            + "full-precision-worker-v1"
        ).encode("utf-8")
    ).hexdigest()[:16]
    cached = read_calibration_cache(cache_path) if resume else {}
    tasks: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        for column_mode, epa_mode in CALIBRATION_MODES.items():
            cache_key = f"{case_id}::{epa_mode}::{epa_fingerprint}"
            if cache_key in cached:
                by_key[cache_key] = cached[cache_key]
                continue
            case_dir = input_root / case_id
            tasks.append(
                {
                    "cache_key": cache_key,
                    "case_id": case_id,
                    "column_mode": column_mode,
                    "epa_mode": epa_mode,
                    "gt_path": str(case_dir / "gt.tum"),
                    "est_path": str(case_dir / "est.tum"),
                    "epa_repo": str(epa_repo),
                }
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if resume and cache_path.exists() else "w"
    completed = 0
    failures: list[dict[str, Any]] = []
    with cache_path.open(file_mode, encoding="utf-8") as cache_stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(run_epa_calibration_task, task): task for task in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                cache_stream.write(json.dumps(result, sort_keys=True) + "\n")
                cache_stream.flush()
                completed += 1
                if result["status"] == "ok":
                    by_key[result["cache_key"]] = result
                else:
                    failures.append(result)
                if completed % 250 == 0 or completed == len(tasks):
                    print(
                        f"EPA calibration progress: {completed}/{len(tasks)} new tasks "
                        f"({len(cached)} cached, {len(failures)} failed)",
                        flush=True,
                    )
    if failures:
        failure_path = cache_path.parent / "calibration_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"EPA calibration failed for {len(failures)} tasks; see {failure_path}"
        )

    for row in rows:
        case_id = str(row["case_id"])
        for column_mode, epa_mode in CALIBRATION_MODES.items():
            result = by_key[f"{case_id}::{epa_mode}::{epa_fingerprint}"]
            row[f"{column_mode}_pair_count"] = int(row["sample_count"])
            row[f"{column_mode}_ape_rmse_m"] = result["ape_rmse_m"]
            row[f"{column_mode}_are_rmse_deg"] = result["are_rmse_deg"]
            row[f"{column_mode}_estimated_offset_ms"] = result["estimated_offset_ms"]
            row[f"{column_mode}_step3_alignment_mode"] = result["step3_alignment_mode"]
            row[f"{column_mode}_step3_inlier_count"] = result["step3_inlier_count"]
            row[f"{column_mode}_step3_rejected_count"] = result["step3_rejected_count"]
            row[f"{column_mode}_runtime_s"] = result["runtime_s"]


def run_internal_epa_worker(argv: list[str]) -> int:
    """Expose full-precision EPA metrics while retaining terminal-only execution."""
    from epa import cli as epa_cli
    from epa.core import pipeline_modular

    captured: dict[str, Any] = {}
    original_metrics = pipeline_modular._compute_pose_metrics_by_stage

    def capture_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
        metrics = original_metrics(*args, **kwargs)
        captured["pose_metrics"] = metrics
        return metrics

    pipeline_modular._compute_pose_metrics_by_stage = capture_metrics
    returncode = epa_cli.main(argv)
    pose_metrics = captured.get("pose_metrics")
    if not pose_metrics:
        raise RuntimeError("EPA worker did not capture pose metrics")
    payload = {
        "ape_rmse_m": float(
            pose_metrics["ape"]["step3"]["translation_part"]["rmse"]
        ),
        "are_rmse_deg": float(
            pose_metrics["ape"]["step3"]["rotation_angle_deg"]["rmse"]
        ),
    }
    print("EPA_WORKER_JSON=" + json.dumps(payload, separators=(",", ":")))
    return int(returncode or 0)


def prepare_profiles() -> dict[str, dict[str, Any]]:
    prepared: dict[str, dict[str, Any]] = {}
    for profile in design.PROFILES:
        if not profile.path.is_file():
            raise FileNotFoundError(profile.path)
        t_gt, p_gt, r_gt = core.normalize_and_resample(profile)
        if len(t_gt) < 3 or not np.all(np.diff(t_gt) > 0):
            raise ValueError(f"Non-monotone or insufficient timestamps: {profile.sequence}")
        prepared[core.profile_key(profile)] = {
            "profile": profile,
            "t_gt": t_gt,
            "p_gt": p_gt,
            "r_gt": r_gt,
            "descriptors": core.trajectory_descriptors(t_gt, p_gt, r_gt),
            "source_sha256": sha256(profile.path),
        }
    return prepared


def generate_case_metrics(
    prepared: dict[str, dict[str, Any]], input_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in design.PROFILES:
        item = prepared[core.profile_key(profile)]
        t_gt = item["t_gt"]
        p_gt = item["p_gt"]
        r_gt = item["r_gt"]
        descriptors = item["descriptors"]
        scale_m = float(descriptors["characteristic_scale_m"])
        for accuracy in core.ACCURACIES:
            for seed in design.SEEDS:
                p_vio, r_vio, realized_t, realized_r = core.simulated_vio(
                    p_gt, r_gt, profile, accuracy, seed, scale_m
                )
                target_t = accuracy.translation_fraction * scale_m
                if not math.isclose(realized_t, target_t, rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError(
                        f"Translation RMS normalization failed for {profile.sequence}/{accuracy.name}/{seed}"
                    )
                if not math.isclose(
                    realized_r, accuracy.rotation_deg, rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise RuntimeError(
                        f"Rotation RMS normalization failed for {profile.sequence}/{accuracy.name}/{seed}"
                    )
                for calibration in design.CALIBRATIONS:
                    t_est, p_est, r_est, _, _ = core.inject_calibration(
                        t_gt, p_vio, r_vio, calibration
                    )
                    case_id = core.clean(
                        f"{profile.name}__{profile.sequence}__{accuracy.name}__"
                        f"{calibration.name}__seed_{seed}"
                    )
                    row: dict[str, Any] = {
                        "case_id": case_id,
                        "motion_group": profile_group(profile),
                        "profile": profile.name,
                        "dataset": profile.dataset,
                        "sequence": profile.sequence,
                        "accuracy": accuracy.name,
                        "condition": calibration.name,
                        "seed": seed,
                        "source_path": str(profile.path),
                        "source_sha256": item["source_sha256"],
                        "sample_count": int(descriptors["sample_count"]),
                        "duration_s": descriptors["duration_s"],
                        "rate_hz": descriptors["rate_hz"],
                        "path_length_m": descriptors["path_length_m"],
                        "bbox_diagonal_m": descriptors["bbox_diagonal_m"],
                        "characteristic_scale_m": scale_m,
                        "target_translation_rms_m": target_t,
                        "target_translation_rms_fraction": accuracy.translation_fraction,
                        "target_rotation_rms_deg": accuracy.rotation_deg,
                        "realized_translation_rms_m": realized_t,
                        "realized_rotation_rms_deg": realized_r,
                        "injected_offset_ms": calibration.offset_s * 1000.0,
                        "injected_rotation_deg": calibration.rotation_deg,
                        "injected_translation_m": calibration.translation_m,
                    }
                    pair_counts: list[int] = []
                    for mode, evaluator in EVALUATORS.items():
                        pair_count, ape, are = evaluator(
                            t_gt, p_gt, r_gt, t_est, p_est, r_est
                        )
                        pair_counts.append(pair_count)
                        row[f"{mode}_pair_count"] = pair_count
                        row[f"{mode}_ape_rmse_m"] = ape
                        row[f"{mode}_are_rmse_deg"] = are
                    if pair_counts[0] != pair_counts[1]:
                        raise RuntimeError(f"Alignment modes used different pose pairs: {row['case_id']}")
                    case_dir = input_root / case_id
                    core.write_tum(case_dir / "gt.tum", t_gt, p_gt, r_gt)
                    core.write_tum(case_dir / "est.tum", t_est, p_est, r_est)
                    rows.append(row)
    return rows


def validate_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = (
        len(design.PROFILES)
        * len(core.ACCURACIES)
        * len(design.CALIBRATIONS)
        * len(design.SEEDS)
    )
    ids = [row["case_id"] for row in rows]
    controls = [row for row in rows if row["condition"] == "none"]
    numeric_fields = [
        f"{mode}_{metric}_rmse_{'m' if metric == 'ape' else 'deg'}"
        for mode in MODES
        for metric in METRICS
    ]
    checks = {
        "expected_case_count": expected,
        "case_count": len(rows),
        "unique_case_count": len(set(ids)),
        "expected_control_count": len(design.PROFILES) * len(core.ACCURACIES) * len(design.SEEDS),
        "control_count": len(controls),
        "finite_metric_count": sum(
            math.isfinite(float(row[field])) for row in rows for field in numeric_fields
        ),
        "expected_finite_metric_count": len(rows) * len(numeric_fields),
        "calibrated_se3_standard_count": sum(
            row.get("se3_position_calibrated_step3_alignment_mode")
            in {"standard", "robust_trimmed"}
            for row in rows
        ),
        "calibrated_se3r_rotation_first_count": sum(
            str(row.get("se3r_rotation_first_calibrated_step3_alignment_mode", "")).startswith(
                "rotation_first"
            )
            for row in rows
        ),
        "finite_estimated_offset_count": sum(
            math.isfinite(float(row[f"{mode}_estimated_offset_ms"]))
            for row in rows
            for mode in CALIBRATION_MODES
        ),
    }
    checks["passed"] = (
        checks["case_count"] == expected
        and checks["unique_case_count"] == expected
        and checks["control_count"] == checks["expected_control_count"]
        and checks["finite_metric_count"] == checks["expected_finite_metric_count"]
        and checks["calibrated_se3_standard_count"] == expected
        and checks["calibrated_se3r_rotation_first_count"] == expected
        and checks["finite_estimated_offset_count"] == 2 * expected
    )
    if not checks["passed"]:
        raise RuntimeError(f"Experiment validation failed: {checks}")
    return checks


def add_relative_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        (row["sequence"], row["accuracy"], row["seed"]): row
        for row in rows
        if row["condition"] == "none"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        baseline = baselines[(row["sequence"], row["accuracy"], row["seed"])]
        relative_valid = row["accuracy"] != "oracle"
        item["relative_valid"] = int(relative_valid)
        for mode in MODES:
            for metric, unit in (("ape", "m"), ("are", "deg")):
                field = f"{mode}_{metric}_rmse_{unit}"
                denominator = float(baseline[field])
                item[f"{mode}_{metric}_baseline_{unit}"] = denominator
                item[f"{mode}_{metric}_relative_change_pct"] = (
                    100.0 * (float(row[field]) - denominator) / denominator
                    if relative_valid and denominator > 1e-12
                    else math.nan
                )
        output.append(item)
    return output


def aggregate_heatmaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["relative_valid"] and row["condition"] in DISPLAY_CONDITIONS:
            grouped[(row["motion_group"], row["accuracy"], row["condition"])].append(row)
    output: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        for accuracy in ACCURACIES:
            for condition in DISPLAY_CONDITIONS:
                selected = grouped[(group, accuracy, condition)]
                if not selected:
                    raise RuntimeError(f"Missing heatmap cell: {group}/{accuracy}/{condition}")
                for mode in MODES:
                    for metric in METRICS:
                        values = [
                            float(row[f"{mode}_{metric}_relative_change_pct"])
                            for row in selected
                        ]
                        output.append(
                            {
                                "alignment_mode": mode,
                                "alignment_label": MODE_LABELS[mode],
                                "metric": metric,
                                "motion_group": group,
                                "motion_group_label": GROUP_LABELS[group],
                                "accuracy": accuracy,
                                "condition": condition,
                                "mean_relative_change_pct": mean(values),
                                "sd_relative_change_pct": sample_sd(values),
                                "n": len(values),
                            }
                        )
    return output


def aggregate_baseline_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "none":
            grouped[(row["sequence"], row["accuracy"])].append(row)
    output: list[dict[str, Any]] = []
    for profile in design.PROFILES:
        for accuracy in core.ACCURACIES:
            selected = grouped[(profile.sequence, accuracy.name)]
            if len(selected) != len(design.SEEDS):
                raise RuntimeError(
                    f"Incomplete Table 1 cell: {profile.sequence}/{accuracy.name}"
                )
            row: dict[str, Any] = {
                "motion_group": profile_group(profile),
                "dataset": profile.dataset,
                "sequence": profile.sequence,
                "accuracy": accuracy.name,
                "n_seeds": len(selected),
                "target_translation_rms_m": mean(
                    float(item["target_translation_rms_m"]) for item in selected
                ),
                "target_rotation_rms_deg": mean(
                    float(item["target_rotation_rms_deg"]) for item in selected
                ),
                "realized_translation_rms_m_mean": mean(
                    float(item["realized_translation_rms_m"]) for item in selected
                ),
                "realized_translation_rms_m_sd": sample_sd(
                    float(item["realized_translation_rms_m"]) for item in selected
                ),
                "realized_rotation_rms_deg_mean": mean(
                    float(item["realized_rotation_rms_deg"]) for item in selected
                ),
                "realized_rotation_rms_deg_sd": sample_sd(
                    float(item["realized_rotation_rms_deg"]) for item in selected
                ),
            }
            for mode in MODES:
                for metric, unit in (("ape", "m"), ("are", "deg")):
                    field = f"{mode}_{metric}_rmse_{unit}"
                    values = [float(item[field]) for item in selected]
                    row[f"{field}_mean"] = mean(values)
                    row[f"{field}_sd"] = sample_sd(values)
            output.append(row)
    return output


def matrix(aggregates: list[dict[str, Any]], mode: str, metric: str) -> np.ndarray:
    lookup = {
        (row["motion_group"], row["accuracy"], row["condition"]): float(
            row["mean_relative_change_pct"]
        )
        for row in aggregates
        if row["alignment_mode"] == mode and row["metric"] == metric
    }
    return np.asarray(
        [
            [lookup[(group, accuracy, condition)] for condition in DISPLAY_CONDITIONS]
            for group in GROUP_ORDER
            for accuracy in ACCURACIES
        ],
        dtype=float,
    )


def compact_pct(value: float) -> str:
    sign = "+" if value > 0.05 else ("-" if value < -0.05 else "")
    magnitude = abs(value)
    if magnitude >= 10000:
        body = f"{magnitude / 1000:.0f}k"
    elif magnitude >= 1000:
        body = f"{magnitude / 1000:.1f}k"
    elif magnitude >= 100:
        body = f"{magnitude:.0f}"
    elif magnitude >= 10:
        body = f"{magnitude:.1f}"
    else:
        body = f"{magnitude:.2f}"
    return f"{sign}{body}%"


def panel_label(ax: plt.Axes, label: str) -> None:
    from matplotlib.transforms import ScaledTranslation

    offset = ScaledTranslation(-12 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        label,
        transform=ax.transAxes + offset,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    cmap: LinearSegmentedColormap,
    norm: SymLogNorm,
) -> mpl.collections.QuadMesh:
    # Draw one axis-aligned vector rectangle per cell. Matplotlib's PDF image
    # resampling can produce diagonal white wedges in some viewers when an
    # ``imshow`` raster is displayed at a fractional zoom level.
    x_edges = np.arange(values.shape[1] + 1, dtype=float) - 0.5
    y_edges = np.arange(values.shape[0] + 1, dtype=float) - 0.5
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        values,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.65,
        antialiased=False,
        rasterized=False,
        snap=True,
    )
    ax.set_xlim(-0.5, values.shape[1] - 0.5)
    ax.set_ylim(values.shape[0] - 0.5, -0.5)
    ax.set_title(title, loc="left", fontsize=7, fontweight="bold", pad=7)
    ax.set_xticks(range(len(DISPLAY_CONDITIONS)))
    ax.set_xticklabels([CONDITION_LABELS[item] for item in DISPLAY_CONDITIONS])
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("right")
        label.set_va("center")
        label.set_rotation_mode("anchor")
    row_labels = [
        f"{GROUP_LABELS[group]} - {ACCURACY_LABELS[accuracy]}"
        for group in GROUP_ORDER
        for accuracy in ACCURACIES
    ]
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis="both", labelsize=5.2, length=0, pad=3)
    ax.set_xticks(np.arange(-0.5, len(DISPLAY_CONDITIONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y in (2.5, 5.5, 8.5, 11.5):
        ax.axhline(y, color="#59636D", linewidth=0.7)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                x,
                y,
                compact_pct(value),
                ha="center",
                va="center",
                fontsize=5.0,
                color="white" if luminance < 0.56 else "#252A2E",
            )
    return image


def add_colorbar(
    fig: plt.Figure,
    cax: plt.Axes,
    image: mpl.cm.ScalarMappable,
    extent: float,
) -> None:
    colorbar = fig.colorbar(image, cax=cax)
    candidates = np.asarray(
        (-100000, -10000, -1000, -100, -10, 0, 10, 100, 1000, 10000, 100000),
        dtype=float,
    )
    ticks = candidates[(candidates >= -extent) & (candidates <= extent)]
    if len(ticks) < 3:
        ticks = np.asarray((-extent, 0.0, extent))
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(
        [
            f"{value / 1000:.0f}k" if abs(value) >= 1000 else f"{value:.0f}"
            for value in ticks
        ]
    )
    colorbar.set_label("Relative change from matched baseline (%)", fontsize=5.2, labelpad=4)
    colorbar.ax.tick_params(labelsize=5.0, width=0.5, length=2)
    colorbar.outline.set_linewidth(0.5)


def save_figure(
    fig: plt.Figure,
    axes: list[plt.Axes],
    colorbar_axis: plt.Axes,
    output_base: Path,
    skip_tiff: bool,
) -> None:
    import sys

    sys.path.insert(0, str(FIGURE_AUDIT_SCRIPTS))
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as exc:
        raise RuntimeError(
            f"Missing panel-alignment QA helper under {FIGURE_AUDIT_SCRIPTS}"
        ) from exc

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(output_base) + ".alignment.json",
        overlay_svg=str(output_base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        axes=axes,
        panel_ids={axis: label for axis, label in zip(axes, ("a", "b"))},
        exclude_axes=[colorbar_axis],
        column_groups=[["a", "b"]],
        strict=True,
    )
    fig.savefig(str(output_base) + ".pdf")
    fig.savefig(str(output_base) + ".svg")
    fig.savefig(str(output_base) + ".png", dpi=300)
    if not skip_tiff:
        fig.savefig(str(output_base) + ".tiff", dpi=600)
    plt.close(fig)


def render_figures(
    aggregates: list[dict[str, Any]], figures_dir: Path, skip_tiff: bool
) -> list[Path]:
    matrices = {
        (mode, metric): matrix(aggregates, mode, metric)
        for mode in MODES
        for metric in METRICS
    }
    all_values = np.concatenate([values.ravel() for values in matrices.values()])
    if not np.all(np.isfinite(all_values)):
        raise RuntimeError("Heatmap matrix contains non-finite values")
    extent = max(25.0, float(np.max(np.abs(all_values))))
    cmap = LinearSegmentedColormap.from_list(
        "relative_change", ("#2B6CB0", "#F7F7F7", "#C44E3F")
    )
    norm = SymLogNorm(linthresh=10.0, linscale=0.9, vmin=-extent, vmax=extent, base=10)

    figures_dir.mkdir(parents=True, exist_ok=True)
    bases: list[Path] = []
    for mode in MODES:
        fig = plt.figure(figsize=(7.2, 8.3), constrained_layout=False)
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.0, 0.026),
            height_ratios=(1.0, 1.0),
            left=0.235,
            right=0.90,
            bottom=0.15,
            top=0.88,
            wspace=0.08,
            hspace=0.32,
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[1, 0])
        cax = fig.add_subplot(grid[:, 1])
        image = draw_heatmap(ax_a, matrices[(mode, "ape")], METRIC_LABELS["ape"], cmap, norm)
        draw_heatmap(ax_b, matrices[(mode, "are")], METRIC_LABELS["are"], cmap, norm)
        panel_label(ax_a, "a")
        panel_label(ax_b, "b")
        add_colorbar(fig, cax, image, extent)
        fig.suptitle(
            f"Calibration sensitivity - {MODE_LABELS[mode]}",
            x=0.235,
            y=0.955,
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
        fig.text(
            0.235,
            0.035,
            "Cells show mean matched relative change from the same-sequence, same-accuracy, same-seed no-perturbation baseline.\n"
            "Hot3D, EuRoC, and KITTI: n=15 per cell (3 trajectories x 5 seeds); EuRoC V2_03 and AEA: n=5.\n"
            "Blue indicates lower error; red indicates higher error. All four figures use the identical color scale.\n"
            "Combined = 10 ms + 20 deg + 0.30 m. Synthetic random walk is not a native VIO error distribution.",
            ha="left",
            va="bottom",
            fontsize=5.1,
            color="#4D5660",
            linespacing=1.25,
        )
        base = figures_dir / FIGURE_NAMES[mode]
        save_figure(fig, [ax_a, ax_b], cax, base, skip_tiff)
        bases.append(base)
    return bases


def setup_markdown(prepared: dict[str, dict[str, Any]], case_count: int) -> str:
    profile_lines = []
    for profile in design.PROFILES:
        descriptors = prepared[core.profile_key(profile)]["descriptors"]
        profile_lines.append(
            f"| {GROUP_LABELS[profile_group(profile)]} | `{profile.sequence}` | "
            f"{descriptors['path_length_m']:.2f} m | {descriptors['bbox_diagonal_m']:.2f} m |"
        )
    accuracy_lines = [
        f"| {accuracy.name} | {100.0 * accuracy.translation_fraction:.1f}% of S | "
        f"{accuracy.rotation_deg:.1f} deg |"
        for accuracy in core.ACCURACIES
    ]
    condition_lines = [
        f"| `{condition.name}` | {1000.0 * condition.offset_s:.0f} ms | "
        f"{condition.rotation_deg:.0f} deg | {condition.translation_m:.2f} m |"
        for condition in design.CALIBRATIONS
    ]
    return f"""# Calibration sensitivity experiment setup

## Material Passport

- Origin: local EPIC-Alignment experiment
- Type: controlled simulation on real GT motion templates
- Verification status: locally generated; scientific interpretation remains evidence-bounded
- Case count: {case_count}

## Research question

How do known timestamp/extrinsic perturbations affect APE and ARE across motion
profiles, simulated VIO accuracies, and global-alignment methods, and how much of
that sensitivity remains after EPA calibration?

## Controlled inputs

Every trajectory is expressed relative to its first pose, cropped to 80 s, and
resampled to 10 Hz. Position uses linear interpolation; orientation uses quaternion
SLERP. No spatial scaling is applied. KITTI timestamps assume 10 Hz.

| Motion group | Sequence | 80-s path length | Bounding-box diagonal |
|---|---|---:|---:|
{chr(10).join(profile_lines)}

## Simulated VIO generation

The estimate is generated from GT using a smoothed, cumulative SE(3) random walk:

`p_vio(t) = p_gt(t) + delta_p(t)`

`R_vio(t) = R_gt(t) Exp(delta_theta(t))`

Translation drift is in the common world frame. Rotation drift is right-multiplied.
Both complete traces are normalized to their RMS targets. Define
`S = max(bounding-box diagonal, 0.1 * path length)`.

| Accuracy | Translation RMS target | Rotation RMS target |
|---|---:|---:|
{chr(10).join(accuracy_lines)}

Five fixed seeds are used: {', '.join(str(seed) for seed in design.SEEDS)}. Within
each `(sequence, accuracy, seed)` block, every calibration-error condition uses the
identical simulated VIO realization.

## Injected calibration errors

| Condition | Time offset | Extrinsic rotation | Extrinsic translation |
|---|---:|---:|---:|
{chr(10).join(condition_lines)}

Rotation axis is normalized `[1, -2, 3]`; translation direction is normalized
`[1, -0.4, 0.2]`.

## Evaluation policies

The exact same injected trajectory is evaluated under four policies:

1. **Fig. 1, SE(3)-original without calibration:** determine global rotation and
   translation by position-only Kabsch alignment.
2. **Fig. 2, SE3R without calibration:** determine global rotation from paired
   orientations first, then solve translation with that rotation fixed.
3. **Fig. 3, SE(3)-original with calibration:** run the current EPA pipeline with
   `--mode {CALIBRATION_MODES['se3_position_calibrated']}`; EPA estimates time offset and extrinsics before a position-only
   Umeyama world alignment.
4. **Fig. 4, SE3R with calibration:** run the current EPA pipeline with
   `--mode {CALIBRATION_MODES['se3r_rotation_first_calibrated']}`; EPA estimates time offset and extrinsics before an orientation-first
   world rotation and fixed-rotation translation.

The calibrated command uses `--dt-resample 0.001`,
`--offset-search-window-s 0.5`, `--t-max-diff 0.06`, `--no-downsample`, and
`--no-plot`. Terminal-only mode changes output generation, not calibration or metrics.

APE is translational RMSE in metres. ARE is orientation-angle RMSE in degrees.
Each heatmap cell is the mean over the available trajectories in its motion group and
five seeds. Relative change is calculated against the matched no-perturbation baseline
under the same alignment and calibration policy:

`100 * (RMSE_condition - RMSE_control) / RMSE_control`.

Oracle controls remain in `case_metrics.csv` and `simulated_vio_rmse.csv`, but are
omitted from relative heatmaps because a numerical-zero denominator is undefined.

## Figure contract

```json
{json.dumps(FIGURE_CONTRACT, indent=2)}
```

## Evidence boundary

This experiment tests sensitivity to controlled synthetic perturbations and whether
the current EPA pipeline mitigates their effect. It does not show that the random-walk
model matches the full error distribution of a real VIO system. A lower calibrated
APE/ARE is also not, by itself, proof that every estimated calibration parameter equals
its injected truth; parameter-recovery evidence belongs in a separate table.
"""


def write_config(
    output_dir: Path,
    prepared: dict[str, dict[str, Any]],
    epa_repo: Path,
    workers: int,
) -> None:
    config = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "figure_contract": FIGURE_CONTRACT,
        "preprocessing": {
            "duration_s": core.WINDOW_S,
            "rate_hz": core.RATE_HZ,
            "position_interpolation": "linear",
            "orientation_interpolation": "quaternion SLERP",
            "spatial_scaling": False,
            "kitti_timestamp_assumption_hz": 10.0,
        },
        "profiles": [
            {
                **asdict(profile),
                "path": str(profile.path),
                "source_sha256": prepared[core.profile_key(profile)]["source_sha256"],
            }
            for profile in design.PROFILES
        ],
        "accuracies": [asdict(accuracy) for accuracy in core.ACCURACIES],
        "calibration_conditions": [asdict(item) for item in design.CALIBRATIONS],
        "seeds": list(design.SEEDS),
        "noise_seed_scope": NOISE_SEED_SCOPE,
        "alignment_modes": {
            "se3_position": "position-only Kabsch SE(3), calibration disabled",
            "se3r_rotation_first": "orientation-first rotation, calibration disabled",
            "se3_position_calibrated": (
                f"current EPA --mode {CALIBRATION_MODES['se3_position_calibrated']}, calibration enabled"
            ),
            "se3r_rotation_first_calibrated": (
                f"current EPA --mode {CALIBRATION_MODES['se3r_rotation_first_calibrated']}, calibration enabled"
            ),
        },
        "epa": {
            "repo": str(epa_repo),
            "cli_sha256": sha256(epa_repo / "src" / "epa" / "cli.py"),
            "trajectory_alignment_sha256": sha256(
                epa_repo / "src" / "epa" / "core" / "trajectory_alignment.py"
            ),
            "workers": workers,
            "arguments": [
                "--dt-resample", "0.001",
                "--offset-search-window-s", "0.5",
                "--t-max-diff", "0.06",
                "--no-downsample",
                "--no-plot",
            ],
        },
        "heatmap_display_conditions": list(DISPLAY_CONDITIONS),
        "heatmap_statistic": "mean matched relative RMSE change (%)",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def prepare_output(output_dir: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("Use only one of --overwrite or --resume")
    if output_dir.exists() and any(output_dir.iterdir()):
        if resume:
            return
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite, --resume, or choose another path."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.print_setup:
        print(__doc__)
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    epa_repo = args.epa_repo.expanduser().resolve()
    CALIBRATION_MODES.clear()
    CALIBRATION_MODES.update(resolve_epa_calibration_modes(epa_repo))
    prepare_output(output_dir, args.overwrite, args.resume)
    # The seed must identify the physical trajectory, not only its broad motion
    # group. This freezes the full-design behavior used by the prior main and
    # relative-baseline runs and prevents different sequences from sharing an
    # unintended random-walk realization.
    core.NOISE_SEED_SCOPE = NOISE_SEED_SCOPE
    prepared = prepare_profiles()
    write_config(output_dir, prepared, epa_repo, args.workers)

    with tempfile.TemporaryDirectory(prefix="calibration-sensitivity-inputs-") as temp_dir:
        input_root = Path(temp_dir)
        rows = generate_case_metrics(prepared, input_root)
        run_calibrated_policies(
            rows=rows,
            input_root=input_root,
            epa_repo=epa_repo,
            workers=args.workers,
            cache_path=output_dir / "data" / "calibration_cache.jsonl",
            resume=args.resume,
        )
    validation = validate_cases(rows)
    relative_rows = add_relative_metrics(rows)
    aggregates = aggregate_heatmaps(relative_rows)
    heatmap_values = [abs(float(row["mean_relative_change_pct"])) for row in aggregates]
    validation["heatmap_shared_color_extent_pct"] = max(25.0, max(heatmap_values))
    validation["heatmap_shared_color_normalization"] = True
    baseline_table = aggregate_baseline_table(rows)

    data_dir = output_dir / "data"
    write_csv(data_dir / "case_metrics.csv", relative_rows)
    write_csv(data_dir / "heatmap_values.csv", aggregates)
    write_csv(data_dir / "simulated_vio_rmse.csv", baseline_table)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    (output_dir / "setup.md").write_text(
        setup_markdown(prepared, len(rows)), encoding="utf-8"
    )

    configure_plotting()
    figure_bases = render_figures(aggregates, output_dir / "figures", args.skip_tiff)
    print(f"Validated {len(rows)} matched cases across four evaluation policies.")
    print(f"Table 1 source: {data_dir / 'simulated_vio_rmse.csv'}")
    for base in figure_bases:
        print(f"Heatmap group: {base}.pdf")
    print(f"Complete output: {output_dir}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--internal-epa-worker":
        raise SystemExit(run_internal_epa_worker(sys.argv[2:]))
    raise SystemExit(main())
