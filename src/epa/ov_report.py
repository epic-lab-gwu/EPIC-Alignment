from __future__ import annotations

import math
from pathlib import Path


PairTable = dict[str, dict[str, tuple[float, float]]]
SegmentPairTable = dict[str, dict[float, dict[str, tuple[float, float]]]]


def fmt_num(value: float, ndigits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{ndigits}f}"


def source_counts(counts: dict[str, int]) -> str:
    epa_count = int(counts.get("epa_step3", 0))
    epa_eval_count = int(counts.get("epa_eval_align", 0))
    epa_posyaw_count = int(counts.get("epa_posyaw", 0))
    failed_count = int(counts.get("failed", 0))
    known = {"epa_step3", "epa_eval_align", "epa_posyaw", "failed"}
    unknown_count = sum(int(v) for k, v in counts.items() if k not in known)
    parts = [f"epa_step3={epa_count}", f"epa_eval={epa_eval_count}", f"failed={failed_count}"]
    if epa_posyaw_count:
        parts.insert(2, f"epa_posyaw={epa_posyaw_count}")
    if unknown_count:
        parts.append(f"unknown={unknown_count}")
    return ", ".join(parts)


def source_details(items: list[str]) -> str:
    if not items:
        return "none"
    return ", ".join(items)


def _latex_name(text: str) -> str:
    return text.replace("_", "\\_")


def _finite_pair_average(values: list[tuple[float, float]]) -> tuple[float, float]:
    finite = [(o, p) for o, p in values if math.isfinite(float(o)) and math.isfinite(float(p))]
    if not finite:
        return float("nan"), float("nan")
    return (
        sum(float(o) for o, _ in finite) / len(finite),
        sum(float(p) for _, p in finite) / len(finite),
    )


def _print_pair_table(
    title: str,
    gt_files: list[Path],
    algo_names: list[str],
    table: PairTable,
) -> None:
    print("============================================")
    print(title)
    print("============================================")
    for gt in gt_files:
        print(f" & \\textbf{{{_latex_name(gt.stem)}}}", end="")
    print(" & \\textbf{Average} \\\\hline")

    for algo in algo_names:
        print(_latex_name(algo), end="")
        row_values: list[tuple[float, float]] = []
        for gt in gt_files:
            ds = gt.stem
            if ds not in table[algo]:
                print(" & - / -", end="")
                continue
            ori, pos = table[algo][ds]
            print(f" & {fmt_num(ori)} / {fmt_num(pos)}", end="")
            row_values.append((float(ori), float(pos)))
        avg_ori, avg_pos = _finite_pair_average(row_values)
        print(f" & {fmt_num(avg_ori)} / {fmt_num(avg_pos)} \\\\")
    print("============================================")


def _print_sr_table(
    gt_files: list[Path],
    algo_names: list[str],
    table: dict[str, dict[str, tuple[float, float]]],
) -> None:
    print("============================================")
    print("DRIFT-VALID SUCCESS RATE LATEX TABLE (% PATH LENGTH)")
    print("============================================")
    for gt in gt_files:
        print(f" & \\textbf{{{_latex_name(gt.stem)}}}", end="")
    print(" & \\textbf{Average} \\\\hline")

    for algo in algo_names:
        print(_latex_name(algo), end="")
        values: list[float] = []
        for gt in gt_files:
            ds = gt.stem
            if ds not in table[algo]:
                print(" & -", end="")
                continue
            sr_dist, _ = table[algo][ds]
            print(f" & {fmt_num(float(sr_dist) * 100.0, 2)}", end="")
            if math.isfinite(float(sr_dist)):
                values.append(float(sr_dist))
        avg = sum(values) / len(values) if values else float("nan")
        print(f" & {fmt_num(avg * 100.0, 2)} \\\\")
    print("============================================")


def print_comparison_tables(
    *,
    gt_files: list[Path],
    algo_names: list[str],
    ate_table: PairTable,
    time_rpe_table: PairTable,
    success_table: dict[str, dict[str, tuple[float, float]]],
    valid_ate_table: PairTable,
    valid_rpe_table: SegmentPairTable,
    valid_time_rpe_table: PairTable,
    segments: list[float],
) -> None:
    # All averages ignore missing/nan cells, matching the old OV-Eval table style.
    _print_pair_table("FULL TRAJECTORY ATE LATEX TABLE (ROT DEG / TRANS M)", gt_files, algo_names, ate_table)
    _print_pair_table(
        "FULL TRAJECTORY 1S TIME RPE LATEX TABLE (ROT DEG / TRANS M)",
        gt_files,
        algo_names,
        time_rpe_table,
    )
    _print_sr_table(gt_files, algo_names, success_table)
    _print_pair_table("DRIFT-VALID ONLY ATE LATEX TABLE (ROT DEG / TRANS M)", gt_files, algo_names, valid_ate_table)
    for seg in segments:
        seg_table = {algo: valid_rpe_table[algo][seg] for algo in algo_names}
        _print_pair_table(
            f"DRIFT-VALID ONLY {int(seg)}m SEGMENT RPE LATEX TABLE (ROT DEG / TRANS M)",
            gt_files,
            algo_names,
            seg_table,
        )
    _print_pair_table(
        "DRIFT-VALID ONLY 1S TIME RPE LATEX TABLE (ROT DEG / TRANS M)",
        gt_files,
        algo_names,
        valid_time_rpe_table,
    )
