#!/usr/bin/env python3
"""Run the motion-profile x simulated-VIO calibration-impact pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R, Slerp


ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(
    os.environ.get("EPA_CALIBRATION_DATA_ROOT", ROOT / "data")
).expanduser().resolve()
WINDOW_S = 80.0
RATE_HZ = 10.0
DT_S = 1.0 / RATE_HZ
SEEDS = (1101, 2202, 3303)
NOISE_SEED_SCOPE = "profile_accuracy_seed"
ROT_AXIS = np.asarray([1.0, -2.0, 3.0], dtype=float)
ROT_AXIS /= np.linalg.norm(ROT_AXIS)
TRANS_DIRECTION = np.asarray([1.0, -0.4, 0.2], dtype=float)
TRANS_DIRECTION /= np.linalg.norm(TRANS_DIRECTION)


@dataclass(frozen=True)
class Profile:
    name: str
    dataset: str
    sequence: str
    path: Path
    fmt: str
    assumed_rate_hz: float | None = None


@dataclass(frozen=True)
class Accuracy:
    name: str
    translation_fraction: float
    rotation_deg: float


@dataclass(frozen=True)
class Calibration:
    name: str
    offset_s: float = 0.0
    rotation_deg: float = 0.0
    translation_m: float = 0.0


PROFILES = (
    Profile(
        "small", "Hot3D", "P0009_1b21bb01",
        RUN_ROOT / "hot3d_gt1" / "P0009_1b21bb01.txt", "tum",
    ),
    Profile(
        "medium", "EuRoC", "V1_02_medium",
        RUN_ROOT / "AlignAnything2" / "GT" / "euroc_mav" / "V1_02_medium.txt", "tum",
    ),
    Profile(
        "large", "KITTI", "05",
        RUN_ROOT / "kitti_gt" / "05.txt", "kitti", assumed_rate_hz=10.0,
    ),
)

ACCURACIES = (
    Accuracy("oracle", 0.0, 0.0),
    Accuracy("high", 0.001, 0.1),
    Accuracy("medium", 0.005, 0.5),
    Accuracy("low", 0.020, 2.0),
)

CALIBRATIONS = (
    Calibration("none"),
    Calibration("time_100ms", offset_s=0.100),
    Calibration("rotation_20deg", rotation_deg=20.0),
    Calibration("translation_0p30m", translation_m=0.30),
    Calibration("combined", offset_s=0.100, rotation_deg=20.0, translation_m=0.30),
)

FIELDS = (
    "case_id", "profile", "dataset", "sequence", "accuracy", "calibration", "seed",
    "status", "runtime_s", "error", "source_path", "source_sha256",
    "sample_count", "duration_s", "rate_hz", "path_length_m", "bbox_diagonal_m",
    "characteristic_scale_m", "median_speed_mps", "p90_speed_mps",
    "cumulative_rotation_deg", "p90_angular_speed_deg_s", "angular_speed_variance",
    "target_translation_rms_m", "target_translation_rms_fraction",
    "target_rotation_rms_deg", "realized_translation_rms_m", "realized_rotation_rms_deg",
    "injected_offset_ms", "injected_rotation_deg", "injected_translation_m",
    "off_pair_count", "ape_off_m", "are_off_deg", "epa_raw_ape_m", "epa_raw_are_deg",
    "ape_on_m", "are_on_deg", "delta_ape_m", "delta_ape_norm", "delta_are_deg",
    "ape_relative_improvement", "are_relative_improvement", "ape_harmed", "are_harmed",
    "estimated_offset_ms", "offset_error_ms", "extrinsic_rotation_error_deg",
    "extrinsic_translation_error_m", "xcorr_peak_normalized", "xcorr_psr",
    "offset_match_ratio_gate", "step1_forced_candidate", "offset_fallback_used",
    "failure_diagnosis_level", "failure_hard_reasons", "failure_soft_reasons",
    "trajectory_quality", "trajectory_diagnosis_level", "trajectory_hard_reasons",
    "trajectory_soft_reasons", "calibration_confidence", "calibration_confidence_reasons",
    "coverage_reliability", "orientation_reliability", "sr_reliability",
    "diagnosis_tags", "metrics_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epa-repo", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pilot")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(profile: str, accuracy: str, seed: int, sequence: str = "") -> int:
    if NOISE_SEED_SCOPE == "profile_sequence_accuracy_seed":
        token_text = f"{profile}|{sequence}|{accuracy}|{seed}"
    else:
        token_text = f"{profile}|{accuracy}|{seed}"
    token = token_text.encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def profile_key(profile: Profile) -> str:
    return f"{profile.name}|{profile.sequence}"


def load_source(profile: Profile) -> tuple[np.ndarray, np.ndarray, R]:
    values = np.loadtxt(profile.path, comments="#")
    values = np.atleast_2d(values)
    if profile.fmt == "tum":
        if values.shape[1] < 8:
            raise ValueError(f"TUM source has {values.shape[1]} columns: {profile.path}")
        timestamps = values[:, 0].astype(float)
        position = values[:, 1:4].astype(float)
        rotation = R.from_quat(values[:, 4:8].astype(float))
    elif profile.fmt == "kitti":
        if values.shape[1] != 12 or profile.assumed_rate_hz is None:
            raise ValueError(f"Invalid KITTI source or missing rate assumption: {profile.path}")
        matrices = values.reshape(-1, 3, 4)
        timestamps = np.arange(len(values), dtype=float) / profile.assumed_rate_hz
        position = matrices[:, :, 3]
        rotation = R.from_matrix(matrices[:, :, :3])
    else:
        raise ValueError(f"Unsupported format: {profile.fmt}")

    order = np.argsort(timestamps, kind="stable")
    timestamps, position, rotation = timestamps[order], position[order], rotation[order]
    keep = np.r_[True, np.diff(timestamps) > 0]
    return timestamps[keep], position[keep], rotation[keep]


def normalize_and_resample(profile: Profile) -> tuple[np.ndarray, np.ndarray, R]:
    timestamps, position, rotation = load_source(profile)
    timestamps = timestamps - timestamps[0]
    if timestamps[-1] + 1e-9 < WINDOW_S - DT_S:
        raise ValueError(f"{profile.sequence} is shorter than the requested 80 s window")

    r0_inv = rotation[0].inv()
    position = r0_inv.apply(position - position[0])
    rotation = r0_inv * rotation
    sample_times = np.arange(0.0, WINDOW_S, DT_S)
    position_interp = np.column_stack(
        [np.interp(sample_times, timestamps, position[:, axis]) for axis in range(3)]
    )
    rotation_interp = Slerp(timestamps, rotation)(sample_times)
    return sample_times, position_interp, rotation_interp


def trajectory_descriptors(t: np.ndarray, p: np.ndarray, q: R) -> dict[str, float]:
    dp = np.diff(p, axis=0)
    dt = np.diff(t)
    distance = np.linalg.norm(dp, axis=1)
    speed = distance / dt
    angular = (q[:-1].inv() * q[1:]).magnitude() * 180.0 / np.pi / dt
    path_length = float(np.sum(distance))
    bbox = float(np.linalg.norm(np.max(p, axis=0) - np.min(p, axis=0)))
    return {
        "sample_count": float(len(t)),
        "duration_s": float(t[-1] - t[0] + DT_S),
        "rate_hz": RATE_HZ,
        "path_length_m": path_length,
        "bbox_diagonal_m": bbox,
        "characteristic_scale_m": max(bbox, 0.1 * path_length),
        "median_speed_mps": float(np.median(speed)),
        "p90_speed_mps": float(np.percentile(speed, 90)),
        "cumulative_rotation_deg": float(np.sum(angular * dt)),
        "p90_angular_speed_deg_s": float(np.percentile(angular, 90)),
        "angular_speed_variance": float(np.var(angular)),
    }


def normalize_trace(trace: np.ndarray, target_rms: float) -> np.ndarray:
    trace = trace - trace[0]
    current = float(np.sqrt(np.mean(np.sum(np.square(trace), axis=1))))
    if target_rms == 0.0:
        return np.zeros_like(trace)
    if current <= 1e-15:
        raise ValueError("Degenerate random-walk trace")
    return trace * (target_rms / current)


def simulated_vio(
    p_gt: np.ndarray,
    r_gt: R,
    profile: Profile,
    accuracy: Accuracy,
    seed: int,
    scale_m: float,
) -> tuple[np.ndarray, R, float, float]:
    target_translation = accuracy.translation_fraction * scale_m
    target_rotation_rad = np.deg2rad(accuracy.rotation_deg)
    if target_translation == 0.0 and target_rotation_rad == 0.0:
        return p_gt.copy(), R.from_quat(r_gt.as_quat()), 0.0, 0.0

    rng = np.random.default_rng(stable_seed(profile.name, accuracy.name, seed, profile.sequence))
    sigma_samples = 5.0
    trans_increments = gaussian_filter1d(
        rng.normal(size=(len(p_gt), 3)), sigma_samples, axis=0, mode="nearest"
    )
    rot_increments = gaussian_filter1d(
        rng.normal(size=(len(p_gt), 3)), sigma_samples, axis=0, mode="nearest"
    )
    trans_walk = normalize_trace(np.cumsum(trans_increments, axis=0), target_translation)
    rot_walk = normalize_trace(np.cumsum(rot_increments, axis=0), target_rotation_rad)
    p_vio = p_gt + trans_walk
    r_vio = r_gt * R.from_rotvec(rot_walk)
    realized_t = float(np.sqrt(np.mean(np.sum(np.square(trans_walk), axis=1))))
    realized_r = float(
        np.sqrt(np.mean(np.square((r_gt.inv() * r_vio).magnitude()))) * 180.0 / np.pi
    )
    return p_vio, r_vio, realized_t, realized_r


def inject_calibration(
    t: np.ndarray, p: np.ndarray, q: R, calibration: Calibration
) -> tuple[np.ndarray, np.ndarray, R, R, np.ndarray]:
    r_ext = R.from_rotvec(ROT_AXIS * np.deg2rad(calibration.rotation_deg))
    t_ext = TRANS_DIRECTION * calibration.translation_m
    t_est = t + calibration.offset_s
    p_est = p + q.apply(t_ext)
    q_est = q * r_ext
    return t_est, p_est, q_est, r_ext, t_ext


def interpolate_pose(t: np.ndarray, p: np.ndarray, q: R, query: np.ndarray) -> tuple[np.ndarray, R]:
    p_query = np.column_stack([np.interp(query, t, p[:, axis]) for axis in range(3)])
    return p_query, Slerp(t, q)(query)


def kabsch_se3(source: np.ndarray, target: np.ndarray) -> tuple[R, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    matrix = vt.T @ u.T
    if np.linalg.det(matrix) < 0:
        vt[-1] *= -1
        matrix = vt.T @ u.T
    rotation = R.from_matrix(matrix)
    translation = target_mean - rotation.apply(source_mean)
    return rotation, translation


def calibration_off_metrics(
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    r_gt: R,
    t_est: np.ndarray,
    p_est: np.ndarray,
    r_est: R,
) -> tuple[int, float, float]:
    mask = (t_est >= t_gt[0]) & (t_est <= t_gt[-1])
    query = t_est[mask]
    if len(query) < 3:
        raise ValueError("Insufficient uncalibrated timestamp overlap")
    p_ref, r_ref = interpolate_pose(t_gt, p_gt, r_gt, query)
    world_r, world_t = kabsch_se3(p_est[mask], p_ref)
    p_aligned = world_r.apply(p_est[mask]) + world_t
    r_aligned = world_r * r_est[mask]
    ape = float(np.sqrt(np.mean(np.sum(np.square(p_ref - p_aligned), axis=1))))
    are = float(np.sqrt(np.mean(np.square((r_ref.inv() * r_aligned).magnitude()))) * 180.0 / np.pi)
    return len(query), ape, are


def write_tum(path: Path, t: np.ndarray, p: np.ndarray, q: R) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.column_stack([t, p, q.as_quat()])
    np.savetxt(path, values, fmt="%.12f %.9f %.9f %.9f %.10f %.10f %.10f %.10f")


def nested(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def joined(value: Any) -> str:
    return ";".join(str(item) for item in value) if isinstance(value, list) else str(value or "")


def clean(value: str) -> str:
    return value.replace(".", "p").replace("/", "-")


def make_case_id(profile: Profile, accuracy: Accuracy, calibration: Calibration, seed: int) -> str:
    return clean(f"{profile.name}__{accuracy.name}__{calibration.name}__seed_{seed}")


def all_cases() -> list[tuple[Profile, Accuracy, Calibration, int]]:
    return [
        (profile, accuracy, calibration, seed)
        for profile in PROFILES
        for accuracy in ACCURACIES
        for calibration in CALIBRATIONS
        for seed in SEEDS
    ]


def run_case(
    profile: Profile,
    accuracy: Accuracy,
    calibration: Calibration,
    seed: int,
    epa_repo: Path,
    output_dir: Path,
    prepared: dict[str, tuple[np.ndarray, np.ndarray, R, dict[str, float]]],
) -> dict[str, Any]:
    case_id = make_case_id(profile, accuracy, calibration, seed)
    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update({
        "case_id": case_id,
        "profile": profile.name,
        "dataset": profile.dataset,
        "sequence": profile.sequence,
        "accuracy": accuracy.name,
        "calibration": calibration.name,
        "seed": seed,
        "status": "failed",
        "source_path": str(profile.path),
        "source_sha256": sha256(profile.path),
        "target_translation_rms_fraction": accuracy.translation_fraction,
        "target_rotation_rms_deg": accuracy.rotation_deg,
        "injected_offset_ms": calibration.offset_s * 1000.0,
        "injected_rotation_deg": calibration.rotation_deg,
        "injected_translation_m": calibration.translation_m,
    })
    t_gt, p_gt, r_gt, descriptors = prepared[profile_key(profile)]
    row.update(descriptors)
    row["target_translation_rms_m"] = accuracy.translation_fraction * descriptors["characteristic_scale_m"]

    started = time.perf_counter()
    try:
        p_vio, r_vio, realized_t, realized_r = simulated_vio(
            p_gt, r_gt, profile, accuracy, seed, descriptors["characteristic_scale_m"]
        )
        row["realized_translation_rms_m"] = realized_t
        row["realized_rotation_rms_deg"] = realized_r
        t_est, p_est, r_est, r_ext_true, t_ext_true = inject_calibration(
            t_gt, p_vio, r_vio, calibration
        )
        pair_count, ape_off, are_off = calibration_off_metrics(
            t_gt, p_gt, r_gt, t_est, p_est, r_est
        )
        row.update({"off_pair_count": pair_count, "ape_off_m": ape_off, "are_off_deg": are_off})

        input_dir = output_dir / "inputs" / case_id
        gt_path, est_path = input_dir / "gt.tum", input_dir / "est.tum"
        write_tum(gt_path, t_gt, p_gt, r_gt)
        write_tum(est_path, t_est, p_est, r_est)
        raw_dir = output_dir / "raw" / case_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(epa_repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["MPLBACKEND"] = "Agg"
        env["MPLCONFIGDIR"] = str(output_dir / "mplconfig")
        with tempfile.TemporaryDirectory(prefix="motion-calibration-") as temp_dir:
            command = [
                sys.executable, "-m", "epa.cli",
                "--gt-csv", str(gt_path), "--gt-format", "tum",
                "--est-path", str(est_path), "--est-format", "tum",
                "--mode", "se3", "--dt-resample", "0.001",
                "--offset-search-window-s", "0.5", "--t-max-diff", "0.06",
                "--no-downsample", "--output-root", temp_dir, "--run-label", case_id,
            ]
            (raw_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
            process = subprocess.run(
                command, cwd=epa_repo, env=env, text=True, capture_output=True, check=False
            )
            (raw_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
            (raw_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
            metrics_paths = sorted(Path(temp_dir).glob("run_*/metrics.json"))
            if process.returncode != 0 or len(metrics_paths) != 1:
                message = process.stderr.strip()[-3000:] or process.stdout.strip()[-3000:]
                raise RuntimeError(message or "EPA did not produce exactly one metrics.json")
            metrics_copy = raw_dir / "metrics.json"
            shutil.copy2(metrics_paths[0], metrics_copy)

        payload = json.loads(metrics_copy.read_text(encoding="utf-8"))
        ape_on = float(nested(payload, "pose_metrics", "ape", "step3", "translation_part", "rmse"))
        are_on = float(nested(payload, "pose_metrics", "ape", "step3", "rotation_angle_deg", "rmse"))
        epa_raw_ape = float(nested(payload, "pose_metrics", "ape", "raw", "translation_part", "rmse"))
        epa_raw_are = float(nested(payload, "pose_metrics", "ape", "raw", "rotation_angle_deg", "rmse"))
        params = nested(payload, "estimated_params", default={})
        r_ext_est = R.from_matrix(np.asarray(params["R_ext"], dtype=float))
        t_ext_est = np.asarray(params["t_ext"], dtype=float)
        offset_est_ms = float(params["offset_s"]) * 1000.0
        diagnosis = nested(payload, "failure_diagnosis", default={})
        tags = nested(payload, "case_diagnostics", "diagnosis_tags", default=[])

        row.update({
            "status": "ok",
            "epa_raw_ape_m": epa_raw_ape,
            "epa_raw_are_deg": epa_raw_are,
            "ape_on_m": ape_on,
            "are_on_deg": are_on,
            "delta_ape_m": ape_off - ape_on,
            "delta_ape_norm": (ape_off - ape_on) / descriptors["characteristic_scale_m"],
            "delta_are_deg": are_off - are_on,
            "ape_relative_improvement": (ape_off - ape_on) / max(ape_off, 1e-9),
            "are_relative_improvement": (are_off - are_on) / max(are_off, 1e-9),
            "ape_harmed": int(ape_on > ape_off + 1e-6),
            "are_harmed": int(are_on > are_off + 1e-4),
            "estimated_offset_ms": offset_est_ms,
            "offset_error_ms": abs(offset_est_ms - calibration.offset_s * 1000.0),
            "extrinsic_rotation_error_deg": float(
                (r_ext_est.inv() * r_ext_true).magnitude() * 180.0 / np.pi
            ),
            "extrinsic_translation_error_m": float(np.linalg.norm(t_ext_est - t_ext_true)),
            "xcorr_peak_normalized": nested(payload, "time_alignment", "xcorr_peak_normalized"),
            "xcorr_psr": nested(payload, "time_alignment", "xcorr_psr"),
            "offset_match_ratio_gate": nested(payload, "time_alignment", "offset_match_ratio_gate"),
            "step1_forced_candidate": int(bool(nested(payload, "time_alignment", "step1_forced_candidate_code", default=0))),
            "offset_fallback_used": int(bool(nested(payload, "time_alignment", "offset_fallback_used", default=0))),
            "failure_diagnosis_level": diagnosis.get("failure_diagnosis_level", ""),
            "failure_hard_reasons": joined(diagnosis.get("failure_diagnosis_hard_reasons", [])),
            "failure_soft_reasons": joined(diagnosis.get("failure_diagnosis_soft_reasons", [])),
            "trajectory_quality": diagnosis.get("trajectory_quality", ""),
            "trajectory_diagnosis_level": diagnosis.get("trajectory_diagnosis_level", ""),
            "trajectory_hard_reasons": joined(diagnosis.get("trajectory_diagnosis_hard_reasons", [])),
            "trajectory_soft_reasons": joined(diagnosis.get("trajectory_diagnosis_soft_reasons", [])),
            "calibration_confidence": diagnosis.get("calibration_confidence", ""),
            "calibration_confidence_reasons": joined(diagnosis.get("calibration_confidence_reasons", [])),
            "coverage_reliability": diagnosis.get("coverage_reliability", ""),
            "orientation_reliability": diagnosis.get("orientation_reliability", ""),
            "sr_reliability": diagnosis.get("sr_reliability", ""),
            "diagnosis_tags": joined(tags),
            "metrics_json": str(metrics_copy.relative_to(output_dir.parent)),
        })
    except Exception as exc:  # Preserve failed cases for audit; never retry silently.
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["runtime_s"] = time.perf_counter() - started
    return row


def load_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=False)
    return result.stdout.strip()


def write_manifest(output_dir: Path, epa_repo: Path, prepared: dict[str, Any]) -> None:
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "design": (
            f"{len(PROFILES)} sequences x {len(ACCURACIES)} accuracy levels x "
            f"{len(CALIBRATIONS)} calibration conditions x {len(SEEDS)} seeds"
        ),
        "window_s": WINDOW_S,
        "rate_hz": RATE_HZ,
        "mode": "se3",
        "seeds": list(SEEDS),
        "noise_seed_scope": NOISE_SEED_SCOPE,
        "rotation_axis": ROT_AXIS.tolist(),
        "translation_direction": TRANS_DIRECTION.tolist(),
        "accuracies": [accuracy.__dict__ for accuracy in ACCURACIES],
        "calibrations": [calibration.__dict__ for calibration in CALIBRATIONS],
        "profiles": [
            {
                **profile.__dict__,
                "path": str(profile.path),
                "sha256": sha256(profile.path),
                "descriptors": prepared[profile_key(profile)][3],
            }
            for profile in PROFILES
        ],
        "calibration_off_definition": "recorded-time association plus global Kabsch SE3; no EPA time/extrinsic correction",
        "calibration_on_definition": "EPA pose_metrics.ape.step3",
        "epa": {
            "repo": str(epa_repo),
            "commit": git_value(epa_repo, "rev-parse", "HEAD"),
            "status_porcelain": git_value(epa_repo, "status", "--porcelain").splitlines(),
            "cli_parameters": {
                "dt_resample_s": 0.001,
                "offset_search_window_s": 0.5,
                "t_max_diff_s": 0.06,
                "no_downsample": True,
            },
        },
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "command": shlex.join(sys.argv),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    epa_repo = args.epa_repo.resolve()
    output_dir = args.output_dir.resolve()
    for profile in PROFILES:
        if not profile.path.is_file():
            raise FileNotFoundError(profile.path)
    if not (epa_repo / "src" / "epa").is_dir():
        raise FileNotFoundError(f"EPA repository not found: {epa_repo}")

    prepared: dict[str, tuple[np.ndarray, np.ndarray, R, dict[str, float]]] = {}
    for profile in PROFILES:
        t, p, q = normalize_and_resample(profile)
        prepared[profile_key(profile)] = (t, p, q, trajectory_descriptors(t, p, q))
    write_manifest(output_dir, epa_repo, prepared)

    cases = all_cases()
    if args.limit > 0:
        cases = cases[: args.limit]
    results_path = output_dir / "results.csv"
    existing = {} if args.overwrite else load_existing(results_path)
    rows: list[dict[str, Any]] = list(existing.values())
    pending = [case for case in cases if make_case_id(*case) not in existing]
    print(f"output={output_dir}")
    print(f"total_selected={len(cases)} already_complete={len(cases)-len(pending)} pending={len(pending)}")
    for index, (profile, accuracy, calibration, seed) in enumerate(pending, start=1):
        case_id = make_case_id(profile, accuracy, calibration, seed)
        print(f"[{index}/{len(pending)}] START {case_id}", flush=True)
        row = run_case(profile, accuracy, calibration, seed, epa_repo, output_dir, prepared)
        rows = [old for old in rows if old.get("case_id") != case_id]
        rows.append(row)
        rows.sort(key=lambda item: item.get("case_id", ""))
        write_rows(results_path, rows)
        print(
            f"[{index}/{len(pending)}] {row['status'].upper()} {case_id} "
            f"runtime={float(row['runtime_s']):.2f}s error={row['error']}",
            flush=True,
        )
    failures = sum(row.get("status") != "ok" for row in rows if row.get("case_id") in {make_case_id(*c) for c in cases})
    print(f"finished={len(cases)} failures={failures} results={results_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
