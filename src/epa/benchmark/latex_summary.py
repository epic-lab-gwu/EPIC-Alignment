from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


LEGACY_KEY_MAP = {
    "epa_status": "vicon_status",
    "epa_offset_est_s": "vicon_offset_est_s",
    "epa_ate_rmse_raw_m": "vicon_ate_rmse_raw_m",
    "epa_ate_rmse_step3_m": "vicon_ate_rmse_step3_m",
    "epa_improve_pct": "vicon_improve_pct",
    "epa_xcorr_peak": "vicon_xcorr_peak",
    "epa_xcorr_psr": "vicon_xcorr_psr",
    "epa_omega_improve_pct": "vicon_omega_improve_pct",
    "epa_matches_equivalent": "vicon_matches_equivalent",
}


NUMERIC_KEYS = {
    "epa_offset_est_s",
    "evo_offset_s",
    "epa_ate_rmse_raw_m",
    "epa_ate_rmse_step3_m",
    "evo_ape_raw_rmse_m",
    "evo_ape_se3_rmse_m",
    "epa_improve_pct",
    "evo_improve_pct",
    "epa_xcorr_peak",
    "epa_xcorr_psr",
    "epa_omega_improve_pct",
    "epa_matches_equivalent",
    "evo_matches",
    "evo_sweep_evals",
}

NL = r"\\"


def _to_float(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_summary_rows(summary_csv: Path) -> list[dict[str, object]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, object]] = []
        for row in reader:
            casted: dict[str, object] = dict(row)

            # Backward compatibility: old benchmark summaries used vicon_* keys.
            for new_key, legacy_key in LEGACY_KEY_MAP.items():
                new_val = str(casted.get(new_key, "")).strip()
                legacy_val = str(casted.get(legacy_key, "")).strip()
                if not new_val and legacy_val:
                    casted[new_key] = legacy_val

            for key in NUMERIC_KEYS:
                casted[key] = _to_float(casted.get(key, ""))
            rows.append(casted)
    return rows


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _fmt_float(value: float, ndigits: int = 3, nan_token: str = "--") -> str:
    if not math.isfinite(value):
        return nan_token
    return f"{value:.{ndigits}f}"


def _row_ok(row: dict[str, object]) -> bool:
    return str(row.get("epa_status", "")) == "ok" and str(row.get("evo_status", "")) == "ok"


def _float_from(row: dict[str, object], key: str) -> float:
    return _to_float(row.get(key, float("nan")))


def _epa_better_pct(row: dict[str, object]) -> float:
    epa = _float_from(row, "epa_ate_rmse_step3_m")
    evo = _float_from(row, "evo_ape_se3_rmse_m")
    if not (math.isfinite(epa) and math.isfinite(evo) and evo > 0.0):
        return float("nan")
    return (evo - epa) / evo * 100.0


def _case_score(row: dict[str, object]) -> tuple[float, float, float]:
    epa = _float_from(row, "epa_ate_rmse_step3_m")
    evo = _float_from(row, "evo_ape_se3_rmse_m")
    if math.isfinite(epa) and math.isfinite(evo):
        align = max(epa, evo)
        gap = abs(epa - evo)
    elif math.isfinite(epa):
        align = epa
        gap = 0.0
    elif math.isfinite(evo):
        align = evo
        gap = 0.0
    else:
        align = float("-inf")
        gap = float("-inf")

    epa_imp = _float_from(row, "epa_improve_pct")
    evo_imp = _float_from(row, "evo_improve_pct")
    improve = max(epa_imp if math.isfinite(epa_imp) else float("-inf"), evo_imp if math.isfinite(evo_imp) else float("-inf"))
    return align, gap, improve


def _select_main_rows(rows: list[dict[str, object]], max_rows: int) -> tuple[list[dict[str, object]], bool, int]:
    comparable = [row for row in rows if _row_ok(row)]
    base_rows = comparable if comparable else list(rows)
    total = len(base_rows)
    if total <= max_rows:
        return base_rows, False, total

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in base_rows:
        grouped[str(row.get("dataset", "unknown"))].append(row)

    dataset_order = sorted(grouped.keys(), key=lambda key: (-len(grouped[key]), key))
    queues: dict[str, deque[dict[str, object]]] = {
        dataset: deque(sorted(grouped[dataset], key=_case_score, reverse=True))
        for dataset in dataset_order
    }

    selected: list[dict[str, object]] = []
    while len(selected) < max_rows:
        progressed = False
        for dataset in dataset_order:
            if len(selected) >= max_rows:
                break
            q = queues[dataset]
            if q:
                selected.append(q.popleft())
                progressed = True
        if not progressed:
            break
    return selected, True, total


def _short_case(case_id: str, max_chars: int = 44) -> str:
    if len(case_id) <= max_chars:
        return case_id
    return case_id[: max_chars - 3] + "..."


def _build_main_table_tex(rows: list[dict[str, object]], *, total_cases: int, truncated: bool) -> str:
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        f"Case & Dataset & Method & EPA RMSE$\\downarrow$ & EVO RMSE$\\downarrow$ & EPA Imp.(\\%)$\\uparrow$ & EVO Imp.(\\%)$\\uparrow$ & EPA Better(\\%)$\\uparrow$ {NL}",
        "\\midrule",
    ]

    if not rows:
        lines.append(f"(none) & -- & -- & -- & -- & -- & -- & -- {NL}")
    else:
        for row in rows:
            case = _latex_escape(_short_case(str(row.get("case", ""))))
            dataset = _latex_escape(str(row.get("dataset", "")))
            method = _latex_escape(str(row.get("method", "")))
            epa_aligned = _fmt_float(_float_from(row, "epa_ate_rmse_step3_m"), 3)
            evo_aligned = _fmt_float(_float_from(row, "evo_ape_se3_rmse_m"), 3)
            epa_imp = _fmt_float(_float_from(row, "epa_improve_pct"), 1)
            evo_imp = _fmt_float(_float_from(row, "evo_improve_pct"), 1)
            better = _fmt_float(_epa_better_pct(row), 1)
            lines.append(
                f"{case} & {dataset} & {method} & {epa_aligned} & {evo_aligned} & {epa_imp} & {evo_imp} & {better} {NL}"
            )

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if truncated:
        lines.append("\\vspace{2pt}")
        lines.append(
            f"\\caption{{Main-paper per-case metrics. Showing {len(rows)} of {total_cases} comparable cases; use Appendix table for full details.}}"
        )
    else:
        lines.append("\\caption{Main-paper per-case metrics (all comparable cases).}")
    lines.append("\\label{tab:epa_main_metrics}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _build_dataset_table_tex(rows: list[dict[str, object]]) -> str:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if _row_ok(row):
            grouped[str(row.get("dataset", "unknown"))].append(row)

    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        f"Dataset & N & EPA Mean RMSE$\\downarrow$ & EVO Mean RMSE$\\downarrow$ & EPA Win Rate(\\%)$\\uparrow$ & Median EPA Better(\\%)$\\uparrow$ {NL}",
        "\\midrule",
    ]

    all_rows: list[dict[str, object]] = []
    for dataset in sorted(grouped.keys()):
        d_rows = grouped[dataset]
        all_rows.extend(d_rows)
        epa_vals = np.asarray([_float_from(row, "epa_ate_rmse_step3_m") for row in d_rows], dtype=float)
        evo_vals = np.asarray([_float_from(row, "evo_ape_se3_rmse_m") for row in d_rows], dtype=float)
        better_vals = np.asarray([_epa_better_pct(row) for row in d_rows], dtype=float)

        valid_pair = np.isfinite(epa_vals) & np.isfinite(evo_vals)
        wins = int(np.sum((epa_vals < evo_vals) & valid_pair))
        n_valid = int(np.sum(valid_pair))
        win_rate = (wins / n_valid * 100.0) if n_valid > 0 else float("nan")

        lines.append(
            f"{_latex_escape(dataset)} & {len(d_rows)} & {_fmt_float(float(np.nanmean(epa_vals)), 3)} & {_fmt_float(float(np.nanmean(evo_vals)), 3)} & {_fmt_float(win_rate, 1)} & {_fmt_float(float(np.nanmedian(better_vals)), 1)} {NL}"
        )

    if all_rows:
        epa_all = np.asarray([_float_from(row, "epa_ate_rmse_step3_m") for row in all_rows], dtype=float)
        evo_all = np.asarray([_float_from(row, "evo_ape_se3_rmse_m") for row in all_rows], dtype=float)
        better_all = np.asarray([_epa_better_pct(row) for row in all_rows], dtype=float)
        valid_pair = np.isfinite(epa_all) & np.isfinite(evo_all)
        wins = int(np.sum((epa_all < evo_all) & valid_pair))
        n_valid = int(np.sum(valid_pair))
        win_rate = (wins / n_valid * 100.0) if n_valid > 0 else float("nan")

        lines.extend(
            [
                "\\midrule",
                f"All & {len(all_rows)} & {_fmt_float(float(np.nanmean(epa_all)), 3)} & {_fmt_float(float(np.nanmean(evo_all)), 3)} & {_fmt_float(win_rate, 1)} & {_fmt_float(float(np.nanmedian(better_all)), 1)} {NL}",
            ]
        )
    else:
        lines.append(f"(none) & 0 & -- & -- & -- & -- {NL}")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Dataset-level aggregate metrics for comparable cases (EPA vs EVO).}",
            "\\label{tab:epa_dataset_metrics}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_appendix_table_tex(rows: list[dict[str, object]]) -> str:
    lines = [
        "% Requires: \\usepackage{booktabs,longtable}",
        "\\small",
        "\\begin{longtable}{lll ll rrrr}",
        f"\\caption{{Full per-case benchmark table (appendix).}}\\label{{tab:epa_appendix_full}}{NL}",
        "\\toprule",
        f"Case & Dataset & Method & EPA & EVO & EPA RMSE$\\downarrow$ & EVO RMSE$\\downarrow$ & EPA Offset(s) & EVO Offset(s) {NL}",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        f"Case & Dataset & Method & EPA & EVO & EPA RMSE$\\downarrow$ & EVO RMSE$\\downarrow$ & EPA Offset(s) & EVO Offset(s) {NL}",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{9}}{{r}}{{\\emph{{Continued on next page}}}}{NL}",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]

    if not rows:
        lines.append(f"(none) & -- & -- & -- & -- & -- & -- & -- & -- {NL}")
    else:
        for row in rows:
            case = _latex_escape(_short_case(str(row.get("case", "")), max_chars=52))
            dataset = _latex_escape(str(row.get("dataset", "")))
            method = _latex_escape(str(row.get("method", "")))
            epa_status = _latex_escape(str(row.get("epa_status", "")))
            evo_status = _latex_escape(str(row.get("evo_status", "")))
            epa_rmse = _fmt_float(_float_from(row, "epa_ate_rmse_step3_m"), 3)
            evo_rmse = _fmt_float(_float_from(row, "evo_ape_se3_rmse_m"), 3)
            epa_offset = _fmt_float(_float_from(row, "epa_offset_est_s"), 3)
            evo_offset = _fmt_float(_float_from(row, "evo_offset_s"), 3)
            lines.append(
                f"{case} & {dataset} & {method} & {epa_status} & {evo_status} & {epa_rmse} & {evo_rmse} & {epa_offset} & {evo_offset} {NL}"
            )

    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def write_latex_tables(
    rows: list[dict[str, object]],
    output_dir: Path,
    *,
    max_main_rows: int = 18,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows, truncated, total_cases = _select_main_rows(rows, max_rows=max(1, int(max_main_rows)))
    main_path = output_dir / "main_table.tex"
    dataset_path = output_dir / "dataset_table.tex"
    appendix_path = output_dir / "appendix_full_table.tex"

    main_path.write_text(
        _build_main_table_tex(main_rows, total_cases=total_cases, truncated=truncated),
        encoding="utf-8",
    )
    dataset_path.write_text(_build_dataset_table_tex(rows), encoding="utf-8")
    appendix_path.write_text(_build_appendix_table_tex(rows), encoding="utf-8")

    return {
        "main_table": main_path,
        "dataset_table": dataset_path,
        "appendix_full_table": appendix_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready LaTeX tables from benchmark summary.csv."
    )
    parser.add_argument("--summary-csv", required=True, help="Path to benchmark summary.csv")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for .tex files (default: <summary_dir>/paper_tables)",
    )
    parser.add_argument(
        "--max-main-rows",
        type=int,
        default=18,
        help="Maximum number of per-case rows kept in main_table.tex",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    summary_csv = Path(args.summary_csv).resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary csv not found: {summary_csv}")

    rows = load_summary_rows(summary_csv)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (summary_csv.parent / "paper_tables")
    produced = write_latex_tables(rows, out_dir, max_main_rows=max(1, int(args.max_main_rows)))

    print(f"Generated LaTeX tables in: {out_dir}")
    for key, path in produced.items():
        print(f"- {key}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
