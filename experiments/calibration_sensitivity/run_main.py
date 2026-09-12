#!/usr/bin/env python3
"""Run the frozen 1,980-case motion-profile calibration-impact experiment."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

import run_pilot as core


ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(
    os.environ.get("EPA_CALIBRATION_DATA_ROOT", ROOT / "data")
).expanduser().resolve()

MAIN_PROFILES = (
    core.Profile("small", "Hot3D", "P0012_915e71c6", RUN_ROOT / "hot3d_gt2" / "P0012_915e71c6.txt", "tum"),
    core.Profile("small", "Hot3D", "P0009_1b21bb01", RUN_ROOT / "hot3d_gt1" / "P0009_1b21bb01.txt", "tum"),
    core.Profile("small", "Hot3D", "P0011_451a7734", RUN_ROOT / "hot3d_gt1" / "P0011_451a7734.txt", "tum"),
    core.Profile("medium", "EuRoC", "V1_01_easy_original", RUN_ROOT / "AlignAnything2" / "GT" / "euroc_mav" / "V1_01_easy_original.txt", "tum"),
    core.Profile("medium", "EuRoC", "V1_02_medium", RUN_ROOT / "AlignAnything2" / "GT" / "euroc_mav" / "V1_02_medium.txt", "tum"),
    core.Profile("medium", "EuRoC", "V2_02_medium", RUN_ROOT / "AlignAnything2" / "GT" / "euroc_mav" / "V2_02_medium.txt", "tum"),
    core.Profile("large", "KITTI", "03", RUN_ROOT / "kitti_gt" / "03.txt", "kitti", assumed_rate_hz=10.0, gravity_aligned=False),
    core.Profile("large", "KITTI", "05", RUN_ROOT / "kitti_gt" / "05.txt", "kitti", assumed_rate_hz=10.0, gravity_aligned=False),
    core.Profile("large", "KITTI", "00", RUN_ROOT / "kitti_gt" / "00.txt", "kitti", assumed_rate_hz=10.0, gravity_aligned=False),
)

MAIN_CALIBRATIONS = (
    core.Calibration("none"),
    core.Calibration("time_20ms", offset_s=0.020),
    core.Calibration("time_100ms", offset_s=0.100),
    core.Calibration("time_300ms", offset_s=0.300),
    core.Calibration("rotation_5deg", rotation_deg=5.0),
    core.Calibration("rotation_20deg", rotation_deg=20.0),
    core.Calibration("rotation_45deg", rotation_deg=45.0),
    core.Calibration("translation_0p05m", translation_m=0.05),
    core.Calibration("translation_0p30m", translation_m=0.30),
    core.Calibration("translation_0p50m", translation_m=0.50),
    core.Calibration("combined", offset_s=0.100, rotation_deg=20.0, translation_m=0.30),
)

MAIN_SEEDS = (1101, 2202, 3303, 4404, 5505)


def configure_core() -> None:
    core.PROFILES = MAIN_PROFILES
    core.CALIBRATIONS = MAIN_CALIBRATIONS
    core.SEEDS = MAIN_SEEDS
    core.NOISE_SEED_SCOPE = "profile_sequence_accuracy_seed"

    def main_case_id(
        profile: core.Profile,
        accuracy: core.Accuracy,
        calibration: core.Calibration,
        seed: int,
    ) -> str:
        return core.clean(
            f"{profile.name}__{profile.sequence}__{accuracy.name}__{calibration.name}__seed_{seed}"
        )

    core.make_case_id = main_case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epa-repo", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "main")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Independent EPA cases to run concurrently.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N selected cases.")
    parser.add_argument(
        "--smoke-matrix", action="store_true",
        help="Run nine representative cases, one per sequence, instead of the full factorial.",
    )
    return parser.parse_args()


def selected_cases(smoke_matrix: bool) -> list[tuple[core.Profile, core.Accuracy, core.Calibration, int]]:
    full = core.all_cases()
    if not smoke_matrix:
        return full
    cases = []
    for index, profile in enumerate(MAIN_PROFILES):
        accuracy = core.ACCURACIES[index % len(core.ACCURACIES)]
        calibration = MAIN_CALIBRATIONS[(index * 2) % len(MAIN_CALIBRATIONS)]
        seed = MAIN_SEEDS[index % len(MAIN_SEEDS)]
        cases.append((profile, accuracy, calibration, seed))
    return cases


def main() -> int:
    configure_core()
    args = parse_args()
    epa_repo = args.epa_repo.resolve()
    output_dir = args.output_dir.resolve()
    for profile in MAIN_PROFILES:
        if not profile.path.is_file():
            raise FileNotFoundError(profile.path)
    if not (epa_repo / "src" / "epa").is_dir():
        raise FileNotFoundError(f"EPA repository not found: {epa_repo}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    prepared: dict[str, tuple[np.ndarray, np.ndarray, R, dict[str, float]]] = {}
    for profile in MAIN_PROFILES:
        t, p, q = core.normalize_and_resample(profile)
        prepared[core.profile_key(profile)] = (t, p, q, core.trajectory_descriptors(t, p, q))
    core.write_manifest(output_dir, epa_repo, prepared)

    cases = selected_cases(args.smoke_matrix)
    if not args.smoke_matrix and len(cases) != 1980:
        raise RuntimeError(f"Expected 1,980 cases, got {len(cases)}")
    case_ids = [core.make_case_id(*case) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Main design contains duplicate case IDs")
    if args.limit > 0:
        cases = cases[: args.limit]
        case_ids = case_ids[: args.limit]

    results_path = output_dir / "results.csv"
    existing = {} if args.overwrite else core.load_existing(results_path)
    rows: list[dict[str, Any]] = list(existing.values())
    pending = [case for case in cases if core.make_case_id(*case) not in existing]
    print(f"output={output_dir}")
    print(f"design_cases={1980 if not args.smoke_matrix else 9}")
    print(f"total_selected={len(cases)} already_complete={len(cases)-len(pending)} pending={len(pending)}")
    if args.workers == 1:
        completed_rows = (
            core.run_case(profile, accuracy, calibration, seed, epa_repo, output_dir, prepared)
            for profile, accuracy, calibration, seed in pending
        )
        for index, (case, row) in enumerate(zip(pending, completed_rows), start=1):
            case_id = core.make_case_id(*case)
            rows = [old for old in rows if old.get("case_id") != case_id]
            rows.append(row)
            rows.sort(key=lambda item: item.get("case_id", ""))
            core.write_rows(results_path, rows)
            print(
                f"[{index}/{len(pending)}] {row['status'].upper()} {case_id} "
                f"runtime={float(row['runtime_s']):.2f}s error={row['error']}", flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_case = {
                executor.submit(core.run_case, profile, accuracy, calibration, seed,
                                epa_repo, output_dir, prepared): (profile, accuracy, calibration, seed)
                for profile, accuracy, calibration, seed in pending
            }
            for index, future in enumerate(as_completed(future_to_case), start=1):
                case = future_to_case[future]
                case_id = core.make_case_id(*case)
                row = future.result()
                rows = [old for old in rows if old.get("case_id") != case_id]
                rows.append(row)
                rows.sort(key=lambda item: item.get("case_id", ""))
                core.write_rows(results_path, rows)
                print(
                    f"[{index}/{len(pending)}] {row['status'].upper()} {case_id} "
                    f"runtime={float(row['runtime_s']):.2f}s error={row['error']}", flush=True,
                )
    selected_ids = set(case_ids)
    failures = sum(row.get("status") != "ok" for row in rows if row.get("case_id") in selected_ids)
    print(f"finished={len(cases)} failures={failures} results={results_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
