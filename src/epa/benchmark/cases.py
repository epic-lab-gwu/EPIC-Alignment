from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _default_data_root() -> Path:
    return Path(os.getenv("EPA_DATA_ROOT", "~/epa_data")).expanduser()


def _default_cases_root() -> str:
    override = os.getenv("EPA_CASES_ROOT", "").strip() or os.getenv("EPA_ALIGNANYTHING_ROOT", "").strip()
    if override:
        return override
    return str(_default_data_root() / "benchmark_cases")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dataset: str
    method: str
    sequence: str
    gt_path: Path
    est_path: Path


def _safe_case_id(dataset: str, sequence: str, method: str) -> str:
    raw = f"{dataset}_{sequence}_{method}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")


def _case_id_qualifier(rel_pose: Path, method: str, sequence: str) -> str:
    parts = rel_pose.parts
    if "pose" in parts:
        pose_idx = parts.index("pose")
        qualifier_parts = list(parts[1:pose_idx])
    else:
        qualifier_parts = list(parts[1:-1])
        if qualifier_parts and qualifier_parts[-1] == sequence:
            qualifier_parts = qualifier_parts[:-1]
        if qualifier_parts and qualifier_parts[-1] == method:
            qualifier_parts = qualifier_parts[:-1]
    return "_".join(part for part in qualifier_parts if part)


_GT_SUFFIXES = (".txt", ".tum", ".csv")
_EST_SUFFIXES = (".txt", ".tum", ".csv")
_EST_STEM_SUFFIXES = ("_poses", "_pose", "_trajectory", "_traj")
_GENERIC_EST_STEMS = {"poses", "pose", "trajectory", "traj"}


def _strip_est_suffix(stem: str) -> str:
    for suffix in _EST_STEM_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return "" if stem in _GENERIC_EST_STEMS else stem


def _is_est_trajectory_file(path: Path) -> bool:
    if path.suffix.lower() not in _EST_SUFFIXES:
        return False
    stem = path.stem
    if path.suffix.lower() == ".txt" and not (
        stem in _GENERIC_EST_STEMS or any(stem.endswith(suffix) for suffix in _EST_STEM_SUFFIXES)
    ):
        return False
    try:
        _sniff_trajectory_columns(path)
    except Exception:
        return False
    return True


def _with_supported_gt_suffixes(path_without_suffix: Path) -> list[Path]:
    return [path_without_suffix.with_suffix(suffix) for suffix in _GT_SUFFIXES]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _glob_gt(gt_root: Path, pattern: str) -> Path | None:
    matches: list[Path] = []
    for suffix in _GT_SUFFIXES:
        matches.extend(p for p in gt_root.glob(f"{pattern}{suffix}") if p.is_file())
    if matches:
        return sorted(matches)[0]
    return None


def _case_candidates(rel_pose: Path) -> list[tuple[str, str, str]]:
    parts = rel_pose.parts
    if len(parts) < 2:
        return []

    dataset = parts[0]
    dirs = list(parts[1:-1])
    file_base = _strip_est_suffix(Path(parts[-1]).stem)
    candidates: list[tuple[str, str, str]] = []

    def add(method: str, sequence: str) -> None:
        method = str(method).strip()
        sequence = str(sequence).strip()
        if method and sequence:
            candidates.append((dataset, method, sequence))

    if "pose" in parts:
        pose_idx = parts.index("pose")
        after_pose = list(parts[pose_idx + 1 : -1])
        if len(after_pose) >= 2:
            add(after_pose[0], after_pose[1])
        if len(after_pose) >= 1 and file_base:
            add(after_pose[0], file_base)
            add(file_base, after_pose[-1])

    if len(dirs) >= 2:
        add(dirs[-2], dirs[-1])
        add(dirs[-1], dirs[-2])
    if len(dirs) >= 1 and file_base:
        add(dirs[-1], file_base)
        add(file_base, dirs[-1])
    if file_base:
        add("default", file_base)

    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _sniff_delimiter(path: Path) -> str | None:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                return ","
            return None
    raise ValueError(f"No numeric rows found in trajectory file: {path}")


def _sniff_trajectory_columns(path: Path) -> int:
    delimiter = _sniff_delimiter(path)
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split(delimiter) if delimiter == "," else line.split()
            values = [float(col.strip()) for col in cols if str(col).strip()]
            if len(values) < 8:
                raise ValueError(f"Trajectory must have at least 8 columns: {path}")
            return len(values)
    raise ValueError(f"No numeric rows found in trajectory file: {path}")


def load_pose_table(path: Path) -> np.ndarray:
    delimiter = _sniff_delimiter(path)
    arr = np.loadtxt(path, comments="#", delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 8:
        raise ValueError(f"Trajectory must have at least 8 columns: {path}")

    data = np.asarray(arr[:, :8], dtype=float)
    finite_mask = np.isfinite(data).all(axis=1)
    data = data[finite_mask]
    if data.shape[0] < 2:
        raise ValueError(f"Trajectory has fewer than 2 valid rows: {path}")

    q = data[:, 4:8]
    q_norm = np.linalg.norm(q, axis=1)
    valid_q = q_norm > 1e-12
    data = data[valid_q]
    q = q[valid_q] / q_norm[valid_q, None]
    data[:, 4:8] = q
    if data.shape[0] < 2:
        raise ValueError(f"Trajectory has fewer than 2 valid quaternion rows: {path}")

    order = np.argsort(data[:, 0], kind="mergesort")
    data = data[order]
    t = data[:, 0]
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    data = data[keep]
    if data.shape[0] < 2:
        raise ValueError(f"Trajectory has fewer than 2 unique timestamps: {path}")
    return data


def _write_tum(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data[:, :8], fmt="%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f")


def _resolve_gt_for_case(gt_root: Path, rel_pose: Path, sequence: str) -> Path | None:
    parts = rel_pose.parts
    dataset = parts[0]

    if dataset == "euroc_mav":
        cand = _first_existing(_with_supported_gt_suffixes(gt_root / "euroc_mav" / sequence))
        if cand is not None:
            return cand

    if dataset == "grand_tour":
        match = _glob_gt(gt_root / "grand_tour", f"**/{sequence}")
        if match is not None:
            return match

    if dataset == "lamaria":
        subset = parts[1] if len(parts) > 1 else ""
        if subset:
            cand = _first_existing(_with_supported_gt_suffixes(gt_root / "lamaria" / subset / sequence))
            if cand is not None:
                return cand
        match = _glob_gt(gt_root / "lamaria", f"**/{sequence}")
        if match is not None:
            return match

    if dataset == "uzh_fpv":
        subset = parts[1] if len(parts) > 1 else ""
        if subset:
            cand = _first_existing(_with_supported_gt_suffixes(gt_root / f"uzhfpv_{subset}" / sequence))
            if cand is not None:
                return cand
        match = _glob_gt(gt_root, f"uzhfpv*/{sequence}")
        if match is not None:
            return match

    return _glob_gt(gt_root, f"**/{sequence}")


def _format_cases_root_error(
    cases_root: Path,
    *,
    missing_entries: list[str],
) -> str:
    env_data_root = os.getenv("EPA_DATA_ROOT", "").strip() or "(unset)"
    env_cases_root = (
        os.getenv("EPA_CASES_ROOT", "").strip()
        or os.getenv("EPA_ALIGNANYTHING_ROOT", "").strip()
        or "(unset)"
    )
    suggested_data_root = _default_data_root()
    suggested_cases_root = Path(_default_cases_root()).expanduser()

    detail_lines = [f"- Missing required path: {cases_root / entry}" for entry in missing_entries]
    if not cases_root.exists():
        detail_lines.insert(0, f"- Base path does not exist: {cases_root}")

    cmd_example = f"epa_bench {suggested_cases_root}"

    lines = [
        f"Invalid --cases-root: {cases_root}",
        "Expected a cases root containing both benchmark/ and GT/.",
        *detail_lines,
        "",
        "Current environment:",
        f"- EPA_DATA_ROOT={env_data_root}",
        f"- EPA_CASES_ROOT={env_cases_root}",
        "",
        "Example fix:",
        f"  export EPA_DATA_ROOT={suggested_data_root}",
        f"  export EPA_CASES_ROOT={suggested_cases_root}",
        f"  {cmd_example}",
    ]
    return "\n".join(lines)


def discover_cases(cases_root: Path) -> tuple[list[BenchmarkCase], list[dict[str, str]]]:
    bench_root = cases_root / "benchmark"
    gt_root = cases_root / "GT"
    missing_entries: list[str] = []
    if not bench_root.exists():
        missing_entries.append("benchmark")
    if not gt_root.exists():
        missing_entries.append("GT")
    if missing_entries:
        raise FileNotFoundError(
            _format_cases_root_error(
                cases_root=cases_root,
                missing_entries=missing_entries,
            )
        )

    selected_cases: list[tuple[str, str, str, Path, Path, Path]] = []
    unresolved: list[dict[str, str]] = []
    for est_path in sorted(p for p in bench_root.rglob("*") if p.is_file() and _is_est_trajectory_file(p)):
        rel_pose = est_path.relative_to(bench_root)
        candidates = _case_candidates(rel_pose)
        if not candidates:
            continue

        selected: tuple[str, str, str, Path] | None = None
        first_candidate = candidates[0]
        for dataset, method, sequence in candidates:
            gt_path = _resolve_gt_for_case(gt_root, rel_pose, sequence)
            if gt_path is not None:
                selected = (dataset, method, sequence, gt_path)
                break

        if selected is None:
            dataset, method, sequence = first_candidate
            gt_path = None
        else:
            dataset, method, sequence, gt_path = selected

        if gt_path is None:
            unresolved.append(
                {
                    "est_relpath": str(rel_pose),
                    "sequence": sequence,
                    "method": method,
                    "dataset": dataset,
                }
            )
            continue
        case_id = _safe_case_id(dataset, sequence, method)
        selected_cases.append((case_id, dataset, method, sequence, gt_path, est_path, rel_pose))

    case_id_counts: dict[str, int] = {}
    for case_id, *_ in selected_cases:
        case_id_counts[case_id] = case_id_counts.get(case_id, 0) + 1

    cases: list[BenchmarkCase] = []
    for case_id, dataset, method, sequence, gt_path, est_path, rel_pose in selected_cases:
        if case_id_counts[case_id] > 1:
            qualifier = _case_id_qualifier(rel_pose, method, sequence)
            if qualifier:
                case_id = _safe_case_id(dataset, f"{qualifier}_{sequence}", method)
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                dataset=dataset,
                method=method,
                sequence=sequence,
                gt_path=gt_path,
                est_path=est_path,
            )
        )
    return cases, unresolved


