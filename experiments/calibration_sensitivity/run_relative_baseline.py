#!/usr/bin/env python3
"""Run the relative-baseline motion-profile calibration-impact experiment.

The design replaces the old spatial-scale normalization with a matched baseline:
for every sequence, VIO-accuracy level, and random-walk seed, the baseline is the
same simulated trajectory with no injected calibration perturbation and with
calibration disabled.  Each non-control row records both calibration-off and
EPA calibration-on metrics, so the two policies can be compared without
changing the simulated VIO realization.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

import run_main as main_design
import run_pilot as core


ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(
    os.environ.get("EPA_CALIBRATION_DATA_ROOT", ROOT / "data")
).expanduser().resolve()

PROFILES = main_design.MAIN_PROFILES + (
    core.Profile(
        "euroc_dynamic",
        "EuRoC",
        "V2_03_difficult",
        RUN_ROOT / "AlignAnything2" / "GT" / "euroc_mav" / "V2_03_difficult.txt",
        "tum",
    ),
    core.Profile(
        "aea_exercise",
        "AEA",
        "loc1111",
        RUN_ROOT / "aea_gt" / "loc1111.txt",
        "tum",
    ),
)

# Intermediate time offsets are retained for the dose-response analysis; the
# manuscript heatmap will highlight 1, 10, and 50 ms as representative levels.
CALIBRATIONS = (
    core.Calibration("none"),
    core.Calibration("time_1ms", offset_s=0.001),
    core.Calibration("time_5ms", offset_s=0.005),
    core.Calibration("time_10ms", offset_s=0.010),
    core.Calibration("time_20ms", offset_s=0.020),
    core.Calibration("time_30ms", offset_s=0.030),
    core.Calibration("time_50ms", offset_s=0.050),
    core.Calibration("rotation_5deg", rotation_deg=5.0),
    core.Calibration("rotation_20deg", rotation_deg=20.0),
    core.Calibration("rotation_45deg", rotation_deg=45.0),
    core.Calibration("translation_0p01m", translation_m=0.01),
    core.Calibration("translation_0p03m", translation_m=0.03),
    core.Calibration("translation_0p05m", translation_m=0.05),
    core.Calibration(
        "combined_realistic", offset_s=0.010, rotation_deg=20.0, translation_m=0.05
    ),
)

SEEDS = main_design.MAIN_SEEDS
EVAL_T_START_S = core.DT_S
EVAL_T_END_S = core.WINDOW_S - 2.0 * core.DT_S


def calibration_off_rotation_first(
    t_gt: np.ndarray,
    p_gt: np.ndarray,
    r_gt: R,
    t_est: np.ndarray,
    p_est: np.ndarray,
    r_est: R,
) -> tuple[int, float, float]:
    """Evaluate without calibration using the current orientation-first SE(3)."""
    query = np.arange(
        EVAL_T_START_S,
        EVAL_T_END_S + 0.5 * core.DT_S,
        core.DT_S,
    )
    if len(query) < 3:
        raise ValueError("Insufficient uncalibrated timestamp overlap")
    p_ref, r_ref = core.interpolate_pose(t_gt, p_gt, r_gt, query)
    p_src, r_src = core.interpolate_pose(t_est, p_est, r_est, query)
    world_r = (r_ref * r_src.inv()).mean()
    world_t = np.mean(p_ref - world_r.apply(p_src), axis=0)
    p_aligned = world_r.apply(p_src) + world_t
    r_aligned = world_r * r_src
    ape = float(np.sqrt(np.mean(np.sum(np.square(p_ref - p_aligned), axis=1))))
    are = float(
        np.sqrt(np.mean(np.square((r_ref.inv() * r_aligned).magnitude())))
        * 180.0
        / np.pi
    )
    return len(query), ape, are


def configure_core() -> None:
    core.PROFILES = PROFILES
    core.CALIBRATIONS = CALIBRATIONS
    core.SEEDS = SEEDS
    core.NOISE_SEED_SCOPE = "profile_sequence_accuracy_seed"
    core.calibration_off_metrics = calibration_off_rotation_first

    def case_id(
        profile: core.Profile,
        accuracy: core.Accuracy,
        calibration: core.Calibration,
        seed: int,
    ) -> str:
        return core.clean(
            f"{profile.name}__{profile.sequence}__{accuracy.name}__"
            f"{calibration.name}__seed_{seed}"
        )

    core.make_case_id = case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epa-repo", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "relative_baseline_se3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--smoke-matrix",
        action="store_true",
        help="Run matched control/time/rotation/translation cases for one profile.",
    )
    return parser.parse_args()


def selected_cases(smoke: bool) -> list[tuple[core.Profile, core.Accuracy, core.Calibration, int]]:
    if not smoke:
        return core.all_cases()
    profile = PROFILES[0]
    accuracy = next(item for item in core.ACCURACIES if item.name == "medium")
    names = {"none", "time_10ms", "rotation_20deg", "translation_0p03m"}
    seed = SEEDS[0]
    return [
        (profile, accuracy, calibration, seed)
        for calibration in CALIBRATIONS
        if calibration.name in names
    ]


def main() -> int:
    configure_core()
    args = parse_args()
    epa_repo = args.epa_repo.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    for profile in PROFILES:
        if not profile.path.is_file():
            raise FileNotFoundError(profile.path)
    if not (epa_repo / "src" / "epa").is_dir():
        raise FileNotFoundError(f"EPA repository not found: {epa_repo}")

    prepared: dict[str, tuple[np.ndarray, np.ndarray, R, dict[str, float]]] = {}
    for profile in PROFILES:
        t, p, q = core.normalize_and_resample(profile)
        prepared[core.profile_key(profile)] = (t, p, q, core.trajectory_descriptors(t, p, q))
    core.write_manifest(output_dir, epa_repo, prepared)
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["calibration_off_definition"] = (
        "recorded-time association plus orientation-first SE(3) world alignment; "
        "no EPA time/extrinsic calibration"
    )
    config["baseline_definition"] = (
        "same sequence, VIO accuracy, and random-walk seed; no perturbation; "
        "calibration disabled"
    )
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cases = selected_cases(args.smoke_matrix)
    expected = len(PROFILES) * len(core.ACCURACIES) * len(CALIBRATIONS) * len(SEEDS)
    if not args.smoke_matrix and len(cases) != expected:
        raise RuntimeError(f"Expected {expected} cases, got {len(cases)}")
    if args.limit > 0:
        cases = cases[: args.limit]

    case_ids = [core.make_case_id(*case) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Duplicate case IDs in relative-baseline design")

    results_path = output_dir / "results.csv"
    existing = {} if args.overwrite else core.load_existing(results_path)
    rows: list[dict[str, Any]] = list(existing.values())
    pending = [case for case in cases if core.make_case_id(*case) not in existing]
    print(f"output={output_dir}")
    print(f"design_cases={expected if not args.smoke_matrix else len(cases)}")
    print(f"selected={len(cases)} complete={len(cases) - len(pending)} pending={len(pending)}")

    def record(case, row, index: int) -> None:
        nonlocal rows
        case_id = core.make_case_id(*case)
        rows = [old for old in rows if old.get("case_id") != case_id]
        rows.append(row)
        rows.sort(key=lambda item: item.get("case_id", ""))
        core.write_rows(results_path, rows)
        print(
            f"[{index}/{len(pending)}] {row['status'].upper()} {case_id} "
            f"runtime={float(row['runtime_s']):.3f}s error={row['error']}",
            flush=True,
        )

    if args.workers == 1:
        for index, case in enumerate(pending, start=1):
            record(case, core.run_case(*case, epa_repo, output_dir, prepared), index)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_case = {
                executor.submit(core.run_case, *case, epa_repo, output_dir, prepared): case
                for case in pending
            }
            for index, future in enumerate(as_completed(future_to_case), start=1):
                case = future_to_case[future]
                record(case, future.result(), index)

    selected_ids = set(case_ids)
    selected_rows = [row for row in rows if row.get("case_id") in selected_ids]
    failures = sum(row.get("status") != "ok" for row in selected_rows)
    print(f"finished={len(selected_rows)} failures={failures} results={results_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
