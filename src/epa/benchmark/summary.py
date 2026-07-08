from __future__ import annotations

import csv
import html
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np

from epa.alignment.modes import PUBLIC_ALIGN_MODES, public_align_mode

def _as_float(value: object) -> float:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return float("nan")
    return float("nan")



def _fmt(value: object, ndigits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return ""
        return f"{x:.{ndigits}f}"
    text = str(value)
    if text.strip().lower() in {"none", "nan"}:
        return ""
    try:
        x = float(text)
    except ValueError:
        return str(value)
    if math.isnan(x):
        return ""
    return f"{x:.{ndigits}f}"



PUBLIC_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case",
    "dataset",
    "method",
    "mode",
    "status",
    "sr_distance_pct",
    "sr_time_pct",
    "sr_reliability",
    "ape_rmse_m",
    "rpe_1s_trans_rmse_m",
    "rpe_1s_rot_rmse_deg",
    "valid_distance_m",
    "total_distance_m",
    "sim3_scale",
    "sim3_reliable",
    "notes",
)


def _public_mode_for_row(row: dict[str, object]) -> str:
    for key in ("epa_eval_align", "mode", "method"):
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        mode = public_align_mode(value, default="")
        if mode in PUBLIC_ALIGN_MODES:
            return mode
    return "se3"


def _public_note_for_row(row: dict[str, object]) -> str:
    notes: list[str] = []
    sr_status = str(row.get("epa_sr_reliability_status", "") or "").strip()
    if sr_status and sr_status.lower() not in {"ok", "true"}:
        notes.append(f"SR {sr_status}")
    if str(row.get("epa_sim3_reliable", "") or "").lower() == "false":
        notes.append("Sim3 unreliable")
    if str(row.get("epa_orientation_unstable", "") or "").lower() == "true":
        notes.append("orientation unstable")
    diagnosis = str(row.get("epa_case_diagnosis_primary", "") or "").strip()
    if diagnosis:
        notes.append(diagnosis)
    if str(row.get("epa_global_gate_failed", "") or "").lower() == "true":
        notes.append("global gate failed")
    return "; ".join(dict.fromkeys(notes))


def _public_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    public_rows: list[dict[str, object]] = []
    for row in rows:
        sr_dist = _as_float(row.get("epa_sr_distance"))
        sr_time = _as_float(row.get("epa_sr_time"))
        public_rows.append(
            {
                "case": row.get("case", ""),
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "mode": _public_mode_for_row(row),
                "status": row.get("status", ""),
                "sr_distance_pct": sr_dist * 100.0 if math.isfinite(sr_dist) else float("nan"),
                "sr_time_pct": sr_time * 100.0 if math.isfinite(sr_time) else float("nan"),
                "sr_reliability": row.get("epa_sr_reliability_status", ""),
                "ape_rmse_m": _as_float(row.get("epa_ate_rmse_step3_m")),
                "rpe_1s_trans_rmse_m": _as_float(row.get("epa_rpe_time_1s_trans_rmse_m")),
                "rpe_1s_rot_rmse_deg": _as_float(
                    row.get("epa_rpe_time_1s_rot_rmse_deg", row.get("epa_orientation_rpe_time_1s_rmse_deg"))
                ),
                "valid_distance_m": _as_float(row.get("epa_valid_distance_m")),
                "total_distance_m": _as_float(row.get("epa_total_distance_m")),
                "sim3_scale": _as_float(row.get("epa_sim3_scale")),
                "sim3_reliable": row.get("epa_sim3_reliable", ""),
                "notes": _public_note_for_row(row),
            }
        )
    return public_rows


def _write_public_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    public_rows = _public_summary_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PUBLIC_SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(public_rows)


def _write_summary_md(rows: list[dict[str, object]], path: Path) -> None:
    total = len(rows)
    epa_ok = sum(1 for row in rows if row.get("status") == "ok")
    has_evo = any(row.get("evo_status") == "ok" for row in rows)
    lines = [
        "# EPA Independent Benchmark",
        "",
        f"- total cases: {total}",
        f"- epa_ok: {epa_ok}",
        "",
        "## Table 1: EPA Metrics",
        "",
        "| case | status | raw_rmse_m (↓) | aligned_rmse_m (↓) | improve_pct (↑) | offset_s (N/A) | matches (↑) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {status} | {vr} | {va} | {vi} | {vo} | {vm} |".format(
                case=row.get("case", ""),
                status=row.get("epa_status", ""),
                vr=_fmt(row.get("epa_ate_rmse_raw_m"), 3),
                va=_fmt(row.get("epa_ate_rmse_step3_m"), 3),
                vi=_fmt(row.get("epa_improve_pct"), 2),
                vo=_fmt(row.get("epa_offset_est_s"), 3),
                vm=_fmt(row.get("epa_matches_equivalent"), 0),
            )
        )
    lines.extend(
        [
            "",
            "## Table 2: Public Summary",
            "",
            "| case | mode | status | SR_dist_% (↑) | SR_time_% (↑) | APE_RMSE_m (↓) | 1s_RPE_m (↓) | valid_distance_m | notes |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {mode} | {status} | {sr_dist} | {sr_time} | {ape} | {rpe} | {valid_dist} | {notes} |".format(
                case=row.get("case", ""),
                mode=_public_mode_for_row(row),
                status=row.get("status", ""),
                sr_dist=_fmt(_as_float(row.get("epa_sr_distance")) * 100.0, 2),
                sr_time=_fmt(_as_float(row.get("epa_sr_time")) * 100.0, 2),
                ape=_fmt(row.get("epa_ate_rmse_step3_m"), 3),
                rpe=_fmt(row.get("epa_rpe_time_1s_trans_rmse_m"), 3),
                valid_dist=_fmt(row.get("epa_valid_distance_m"), 3),
                notes=_public_note_for_row(row),
            )
        )
    lines.extend(
        [
            "",
            "## Table 3: Advanced Diagnostics",
            "",
            "| case | reliability | global_gate | internal_alignment | stable_solve_% | rejected_% | sim3_scale | sim3_reliable | orientation_unstable |",
            "|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        gate_value = _fmt(row.get("epa_global_gate_value_m"), 3)
        gate_text = (
            f"{'fail' if str(row.get('epa_global_gate_failed', '')).lower() == 'true' else 'pass'} {gate_value} m"
            if gate_value
            else ""
        )
        lines.append(
            "| {case} | {reliability} | {gate} | {mode} | {stable_pct} | {reject_pct} | {sim3_scale} | {sim3_reliable} | {orientation_unstable} |".format(
                case=row.get("case", ""),
                reliability=row.get("epa_sr_reliability_status", ""),
                gate=gate_text,
                mode=row.get("epa_step3_alignment_mode", ""),
                stable_pct=_fmt(_as_float(row.get("epa_step3_stable_solve_ratio")) * 100.0, 2),
                reject_pct=_fmt(_as_float(row.get("epa_step3_rejection_ratio")) * 100.0, 2),
                sim3_scale=_fmt(row.get("epa_sim3_scale"), 6),
                sim3_reliable=row.get("epa_sim3_reliable", ""),
                orientation_unstable=row.get("epa_orientation_unstable", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Table 4: epa-only Metrics",
            "",
            "| case | epa_xcorr_peak (↑) | epa_xcorr_psr (↑) | epa_omega_improve_pct (↑) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {peak} | {psr} | {omega} |".format(
                case=row.get("case", ""),
                peak=_fmt(row.get("epa_xcorr_peak"), 3),
                psr=_fmt(row.get("epa_xcorr_psr"), 2),
                omega=_fmt(row.get("epa_omega_improve_pct"), 2),
            )
        )
    if has_evo:
        lines.extend(
            [
                "",
                "## Table 5: Legacy evo Baseline",
                "",
                "| case | evo_status | evo_aligned_rmse_m (↓) | evo_improve_pct (↑) | evo_offset_s (N/A) | evo_sweep_evals (N/A) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {case} | {status} | {rmse} | {improve} | {offset} | {sweeps} |".format(
                    case=row.get("case", ""),
                    status=row.get("evo_status", ""),
                    rmse=_fmt(row.get("evo_ape_se3_rmse_m"), 3),
                    improve=_fmt(row.get("evo_improve_pct"), 2),
                    offset=_fmt(row.get("evo_offset_s"), 3),
                    sweeps=_fmt(row.get("evo_sweep_evals"), 0),
                )
            )
    lines.extend(
        [
            "",
            "Note: `↑` means larger is better, `↓` means smaller is better, and `N/A` marks process parameters without a universal quality direction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rel_link(path_text: object, base_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_dir():
        candidate = path / "plots" / "step3_alignment_map.png"
        if candidate.exists():
            path = candidate
        else:
            path = path / "metrics.json"
    try:
        href = os.path.relpath(path, start=base_dir)
    except ValueError:
        href = str(path)
    label = path.name or str(path)
    return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'


def _interactive_thumb(path_text: object, base_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    run_path = Path(text)
    img = run_path / "plots" / "step3_alignment_map.png"
    interactive = run_path / "interactive_report.html"
    if interactive.exists() and not img.exists():
        try:
            href = os.path.relpath(interactive, start=base_dir)
        except ValueError:
            href = str(interactive)
        return f'<a href="{html.escape(href, quote=True)}">interactive_report.html</a>'
    if not img.exists():
        return _rel_link(path_text, base_dir)
    try:
        img_href = os.path.relpath(img, start=base_dir)
    except ValueError:
        img_href = str(img)
    try:
        target_href = os.path.relpath(interactive if interactive.exists() else img, start=base_dir)
    except ValueError:
        target_href = str(interactive if interactive.exists() else img)
    label = "interactive_report.html" if interactive.exists() else "step3_alignment_map.png"
    return (
        f'<a class="thumbLink" href="{html.escape(target_href, quote=True)}">'
        f'<img class="thumb" src="{html.escape(img_href, quote=True)}" alt="step3 alignment">'
        f'<span class="plotCount">{html.escape(label)}</span>'
        "</a>"
    )


def _summary_filter_options(rows: list[dict[str, object]], key: str) -> str:
    values = sorted({str(row.get(key, "") or "") for row in rows if str(row.get(key, "") or "")})
    options = ['<option value="">all</option>']
    options.extend(
        f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>'
        for value in values
    )
    return "".join(options)


def _write_summary_html(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    method_labels: list[str] = []
    for row in rows:
        method = str(row.get("method", "") or "method")
        if method not in method_labels:
            method_labels.append(method)
    method_hint = " / ".join(method_labels) if method_labels else "methods"
    matrix_layout = len(method_labels) > 3

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("case", "") or "case"), []).append(row)

    def _row_for_method(case_rows: list[dict[str, object]], method: str) -> dict[str, object] | None:
        for item in case_rows:
            if str(item.get("method", "") or "method") == method:
                return item
        return None

    def _metric_values(
        case_rows: list[dict[str, object]],
        key: str,
        *,
        fallback_key: str = "",
        scale: float = 1.0,
        ndigits: int = 3,
        unit: str = "",
    ) -> str:
        vals: list[str] = []
        for method in method_labels:
            item = _row_for_method(case_rows, method)
            if item is None:
                vals.append("-")
                continue
            value = _as_float(item.get(key))
            if fallback_key and not math.isfinite(value):
                value = _as_float(item.get(fallback_key))
            value *= float(scale)
            text = _fmt(value, ndigits)
            if unit and text:
                text = f"{text}%" if unit == "%" else f"{text} {unit}"
            vals.append(text or "-")
        return " / ".join(vals)

    def _text_values(case_rows: list[dict[str, object]], key: str) -> str:
        vals: list[str] = []
        for method in method_labels:
            item = _row_for_method(case_rows, method)
            if item is None:
                vals.append("-")
                continue
            text = str(item.get(key, "") or "").strip()
            vals.append(text or "-")
        return " / ".join(vals)

    def _valid_distance_values(case_rows: list[dict[str, object]]) -> str:
        vals: list[str] = []
        for method in method_labels:
            item = _row_for_method(case_rows, method)
            if item is None:
                vals.append("-")
                continue
            valid = _fmt(item.get("epa_valid_distance_m"), 1)
            total = _fmt(item.get("epa_total_distance_m"), 1)
            if valid and total:
                vals.append(f"{valid}/{total} m")
            elif valid:
                vals.append(f"{valid} m")
            else:
                vals.append("-")
        return " / ".join(vals)

    def _sr_key(axis: str, scope: str) -> str:
        if scope == "gated":
            return f"epa_sr_{axis}_gated"
        return f"epa_sr_{axis}"

    def _sr_values(case_rows: list[dict[str, object]], *, axis: str, scope: str = "local") -> str:
        vals: list[str] = []
        key = _sr_key(axis, scope)
        for method in method_labels:
            item = _row_for_method(case_rows, method)
            if item is None:
                vals.append("-")
                continue
            value = _as_float(item.get(key))
            vals.append(_fmt(value * 100.0, 1) + "%" if math.isfinite(value) else "-")
        return " | ".join(vals)

    def _sr_text(row: dict[str, object], *, axis: str, scope: str = "local") -> str:
        value = _as_float(row.get(_sr_key(axis, scope)))
        return _fmt(value * 100.0, 1) + "%" if math.isfinite(value) else "-"

    def _valid_distance_text(row: dict[str, object]) -> str:
        valid = _fmt(row.get("epa_valid_distance_m"), 1)
        total = _fmt(row.get("epa_total_distance_m"), 1)
        if valid and total:
            return f"{valid}/{total} m"
        if valid:
            return f"{valid} m"
        return "-"

    def _global_gate_text(row: dict[str, object]) -> str:
        failed = str(row.get("epa_global_gate_failed", "") or "").lower() == "true"
        value = _fmt(row.get("epa_global_gate_value_m"), 3)
        if value:
            return f"{'fail' if failed else 'pass'} {value} m"
        return "fail" if failed else "pass"

    def _row_status(row: dict[str, object]) -> tuple[str, str]:
        if (
            str(row.get("status", "")) != "ok"
            or str(row.get("epa_case_status", "")) == "globally_failed"
            or str(row.get("epa_sr_reliability_status", "")).lower() == "failed"
        ):
            return "failed", "bad"
        if (
            str(row.get("epa_case_status", "")) != "valid_segment"
            or str(row.get("epa_sr_reliability_status", "")).lower() == "warning"
            or str(row.get("epa_sim3_reliable", "")).lower() == "false"
            or str(row.get("epa_sim3_warning", "")).strip()
            or str(row.get("epa_orientation_unstable", "")).lower() == "true"
            or str(row.get("epa_case_diagnosis_primary", "")).strip()
        ):
            return "warning", "warn"
        return "ok", ""

    def _case_status(case_rows: list[dict[str, object]]) -> tuple[str, str]:
        if any(str(row.get("status", "")) != "ok" or str(row.get("epa_case_status", "")) == "globally_failed" for row in case_rows):
            return "failed", "bad"
        if any(
            str(row.get("epa_case_status", "")) != "valid_segment"
            or str(row.get("epa_sr_reliability_status", "")).lower() in {"failed", "warning"}
            or str(row.get("epa_sim3_reliable", "")).lower() == "false"
            or str(row.get("epa_sim3_warning", "")).strip()
            or str(row.get("epa_orientation_unstable", "")).lower() == "true"
            or str(row.get("epa_case_diagnosis_primary", "")).strip()
            for row in case_rows
        ):
            return "warning", "warn"
        return "ok", ""

    report_dir = path.parent / "reports"

    def _safe_report_slug(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")
        return slug[:140] or "report"

    def _short_report_href(row: dict[str, object]) -> str:
        run_dir = Path(str(row.get("epa_run_dir", "") or ""))
        interactive = run_dir / "interactive_report.html"
        if not interactive.exists():
            return ""
        case_name = _safe_report_slug(str(row.get("case", "") or "case"))
        method = _safe_report_slug(str(row.get("method", "") or "method"))
        report_dir.mkdir(parents=True, exist_ok=True)
        target = report_dir / f"{case_name}__{method}.html"
        try:
            shutil.copy2(interactive, target)
            return os.path.relpath(target, path.parent)
        except Exception:
            try:
                return os.path.relpath(interactive, path.parent)
            except ValueError:
                return str(interactive)

    def _case_link(case_name: str, case_rows: list[dict[str, object]]) -> str:
        target = ""
        preferred_rows = sorted(
            case_rows,
            key=lambda row: (
                "health" not in str(row.get("method", "")).lower(),
                "original" in str(row.get("method", "")).lower(),
            ),
        )
        for row in preferred_rows:
            target = _short_report_href(row)
            if target:
                break
        if not target:
            return html.escape(case_name)
        return f'<a class="caseLink" href="{html.escape(target, quote=True)}">{html.escape(case_name)}</a>'

    def _run_link(row: dict[str, object], label: str) -> str:
        target = _short_report_href(row)
        if target:
            return f'<a class="methodPill" href="{html.escape(target, quote=True)}">{html.escape(label)}</a>'
        return f'<span class="methodPill">{html.escape(label)}</span>'

    def _diagnosis_text(case_rows: list[dict[str, object]]) -> str:
        code_to_sentence = {
            "time_alignment_weak": "Time sync is weak; inspect replay before trusting small differences.",
            "step3_no_reliable_stable_segment": "No reliable stable segment was found for alignment.",
            "global_gate_too_large": "Global position error is too large; this case is likely failed.",
            "trajectory_jump": "Trajectory has jump or divergence; SR may be misleading.",
            "scale_or_unit_suspect": "Scale or unit looks suspicious; do not trust position SR alone.",
            "gt_mapping_suspect": "GT mapping may be imperfect for this sequence.",
            "alignment_poor": "Alignment quality is poor; inspect replay.",
            "poor_align": "Alignment quality is poor; inspect replay.",
            "orientation_unstable": "Orientation is unstable; translation SR alone is not enough.",
            "sim3_unreliable": "Sim3 is not reliable here; inspect replay before trusting SR.",
            "sim3_may_mask_failure": "Sim3 may be masking a real trajectory failure; treat SR as unreliable.",
        }
        ordered_codes = [
            "sim3_may_mask_failure",
            "sim3_unreliable",
            "trajectory_jump",
            "global_gate_too_large",
            "time_alignment_weak",
            "step3_no_reliable_stable_segment",
            "orientation_unstable",
            "scale_or_unit_suspect",
            "gt_mapping_suspect",
            "alignment_poor",
            "poor_align",
        ]
        codes: set[str] = set()
        has_unmapped_warning = False
        for row in case_rows:
            for key in (
                "epa_case_diagnosis_primary",
                "epa_case_diagnosis_summary",
                "epa_case_diagnosis_tags",
            ):
                text = str(row.get(key, "") or "")
                for token in re.split(r"[,;\s]+", text):
                    token = token.strip()
                    if token:
                        codes.add(token)
            if str(row.get("epa_sim3_reliable", "") or "").lower() == "false":
                codes.add("sim3_unreliable")
            if str(row.get("epa_sim3_may_mask_failure", "") or "").lower() == "true":
                codes.add("sim3_may_mask_failure")
            if str(row.get("epa_orientation_unstable", "") or "").lower() == "true":
                codes.add("orientation_unstable")
            if str(row.get("epa_sim3_warning", "") or "").strip() or str(row.get("epa_orientation_warning", "") or "").strip():
                has_unmapped_warning = True
        notes = [code_to_sentence[code] for code in ordered_codes if code in codes and code in code_to_sentence]
        if not notes and has_unmapped_warning:
            notes.append("Alignment warning is present; inspect replay before trusting SR.")
        if not notes:
            return "No obvious alignment issue was detected."
        deduped = list(dict.fromkeys(notes))
        return " ".join(deduped[:3])

    def _diagnosis_brief(row: dict[str, object]) -> str:
        summary = " ".join(
            str(row.get(key, "") or "")
            for key in (
                "epa_case_diagnosis_primary",
                "epa_case_diagnosis_summary",
                "epa_case_diagnosis_tags",
            )
        )
        sr_status = str(row.get("epa_sr_reliability_status", "") or "").lower()
        if sr_status == "failed":
            return "SR gated failed"
        if str(row.get("epa_sim3_reliable", "") or "").lower() == "false":
            return "Sim3 unreliable"
        if str(row.get("epa_global_gate_failed", "") or "").lower() == "true":
            return "Global gate failed"
        if "trajectory_jump" in summary:
            return "Trajectory jump"
        if "time_alignment_weak" in summary:
            return "Time sync weak"
        if "scale_or_unit_suspect" in summary:
            return "Scale/unit suspect"
        if "gt_mapping_suspect" in summary:
            return "GT mapping suspect"
        if str(row.get("epa_orientation_unstable", "") or "").lower() == "true":
            return "Orientation unstable"
        return "OK"

    def _advanced_details(row: dict[str, object]) -> str:
        stable_pct = _fmt(_as_float(row.get("epa_step3_stable_solve_ratio")) * 100.0, 2)
        reject_pct = _fmt(_as_float(row.get("epa_step3_rejection_ratio")) * 100.0, 2)
        warning_parts = [
            str(row.get("epa_sr_warning_explanation", "") or "").strip(),
            str(row.get("epa_sim3_warning", "") or "").strip(),
            str(row.get("epa_orientation_warning", "") or "").strip(),
        ]
        detail_rows = [
            ("SR reliability", str(row.get("epa_sr_reliability_status", "") or "-")),
            ("Global gate", _global_gate_text(row)),
            ("Internal alignment", str(row.get("epa_step3_alignment_mode", "") or "-")),
            ("Stable solve", f"{stable_pct}%" if stable_pct else "-"),
            ("Rejected", f"{reject_pct}%" if reject_pct else "-"),
            ("Sim3 scale", _fmt(row.get("epa_sim3_scale"), 6) or "-"),
            ("Sim3 reliable", str(row.get("epa_sim3_reliable", "") or "-")),
            ("Sim3 path", _sim3_path_text(row)),
            ("Orientation unstable", str(row.get("epa_orientation_unstable", "") or "-")),
            ("Warnings", " ".join(part for part in warning_parts if part) or "-"),
        ]
        body = "".join(
            f'<div><span class="muted">{html.escape(label)}:</span> {html.escape(value)}</div>'
            for label, value in detail_rows
        )
        return f'<details class="advanced"><summary>details</summary>{body}</details>'

    def _diagnosis_title(row: dict[str, object]) -> str:
        notes = [
            _diagnosis_text([row]),
            str(row.get("epa_sr_warning_explanation", "") or "").strip(),
            str(row.get("epa_sim3_warning", "") or "").strip(),
            str(row.get("epa_orientation_warning", "") or "").strip(),
        ]
        return " ".join(note for note in notes if note)

    def _sim3_path_text(row: dict[str, object]) -> str:
        stable = str(row.get("epa_sim3_stable_anchor_used", "") or "").strip().lower()
        fallback = str(row.get("epa_sim3_fallback_used", "") or "").strip().lower()
        parts: list[str] = []
        if stable:
            parts.append(f"stable={stable}")
        if not fallback:
            return ", ".join(parts) if parts else "-"
        if fallback == "true":
            reason = str(row.get("epa_sim3_fallback_reason", "") or "").strip()
            parts.append(f"v2=true: {reason}" if reason else "v2=true")
        else:
            parts.append(f"v2={fallback}")
        return ", ".join(parts)

    total = len(grouped)
    table_rows: list[str] = []
    status_counts = {"ok": 0, "warning": 0, "failed": 0}
    for case_name, case_rows in grouped.items():
        status_label, row_class = _case_status(case_rows)
        status_counts[status_label] = status_counts.get(status_label, 0) + 1
        dataset = str(case_rows[0].get("dataset", ""))
        search_text = " ".join(str(row.get(key, "")) for row in case_rows for key in (
            "case", "dataset", "method", "epa_case_status", "epa_case_diagnosis_summary",
            "epa_case_diagnosis_primary", "epa_sim3_reliable", "epa_sim3_warning",
            "epa_step3_alignment_mode", "epa_orientation_warning", "epa_sr_reliability_status",
            "epa_sr_warning_explanation", "epa_sim3_stable_anchor_used", "epa_sim3_fallback_used",
            "epa_sim3_fallback_reason",
        ))
        pill = {
            "ok": "okPill",
            "warning": "warnPill",
            "failed": "badPill",
        }.get(status_label, "warnPill")
        if matrix_layout:
            for idx, row in enumerate(case_rows):
                row_status, row_class = _row_status(row)
                rel_status = str(row.get("epa_sr_reliability_status", "") or row_status)
                rel_pill = {"ok": "okPill", "warning": "warnPill", "failed": "badPill"}.get(
                    rel_status.lower(), "warnPill"
                )
                case_cell = (
                    '<td class="caseCell">'
                    '<button class="caseMarker" title="Mark this case" type="button"></button>'
                    f'<div>{_case_link(case_name, case_rows)}'
                    f'<div class="caseSub"><span class="pill {pill}">{html.escape(status_label)}</span>'
                    f'<span class="muted"> {html.escape(dataset)}</span></div></div>'
                    '</td>'
                    if idx == 0
                    else '<td class="caseCell caseRepeat"><button class="caseMarker" title="Mark this case" type="button"></button></td>'
                )
                method = str(row.get("method", "") or "method")
                diag_title = _diagnosis_title(row)
                diag = _diagnosis_brief(row)
                table_rows.append(
                    f'<tr class="{html.escape(row_class)} {"caseStart" if idx == 0 else ""}" '
                    f'data-case="{html.escape(case_name, quote=True)}" '
                    f'data-search="{html.escape((search_text + " " + method + " " + diag_title).lower(), quote=True)}" '
                    f'data-status="{html.escape(row_status, quote=True)}">'
                    f'{case_cell}'
                    f'<td>{_run_link(row, method)}</td>'
                    f'<td><span class="metricVals">{html.escape(_public_mode_for_row(row))}</span></td>'
                    f'<td><span class="metricVals">{html.escape(_sr_text(row, axis="distance", scope="local"))}</span></td>'
                    f'<td><span class="metricVals">{html.escape(_sr_text(row, axis="time", scope="local"))}</span></td>'
                    f'<td><span class="metricVals">{html.escape((_fmt(row.get("epa_ate_rmse_step3_m"), 3) or "-") + " m")}</span></td>'
                    f'<td><span class="metricVals">{html.escape((_fmt(row.get("epa_rpe_time_1s_trans_rmse_m"), 3) or "-") + " m")}'
                    f'<div class="muted">rot {html.escape((_fmt(row.get("epa_rpe_time_1s_rot_rmse_deg"), 3) or _fmt(row.get("epa_orientation_rpe_time_1s_rmse_deg"), 3) or "-"))} deg</div></span></td>'
                    f'<td><span class="metricVals">{html.escape(_valid_distance_text(row))}</span></td>'
                    f'<td title="{html.escape(diag_title, quote=True)}">{html.escape(diag)}'
                    f'<div class="muted"><span class="pill {rel_pill}">{html.escape(rel_status)}</span></div></td>'
                    f'<td>{_advanced_details(row)}</td>'
                    '</tr>'
                )
        else:
            first_row = case_rows[0]
            table_rows.append(
                f'<tr class="{html.escape(row_class)}" data-case="{html.escape(case_name, quote=True)}" '
                f'data-search="{html.escape(search_text.lower(), quote=True)}" data-status="{html.escape(status_label, quote=True)}">'
                '<td class="caseCell">'
                '<button class="caseMarker" title="Mark this case" type="button"></button>'
                f'<div>{_case_link(case_name, case_rows)}'
                f'<div class="caseSub"><span class="pill {pill}">{html.escape(status_label)}</span>'
                f'<span class="muted"> {html.escape(dataset)}</span></div></div>'
                '</td>'
                f'<td><span class="metricVals">{html.escape(_text_values(case_rows, "method"))}</span></td>'
                f'<td><span class="metricVals">{html.escape(" / ".join(_public_mode_for_row(row) for row in case_rows))}</span></td>'
                f'<td><span class="metricVals">{html.escape(_sr_values(case_rows, axis="distance", scope="local"))}</span></td>'
                f'<td><span class="metricVals">{html.escape(_sr_values(case_rows, axis="time", scope="local"))}</span></td>'
                f'<td><span class="metricVals">{html.escape(_metric_values(case_rows, "epa_ate_rmse_step3_m", ndigits=3, unit="m"))}</span></td>'
                f'<td><span class="metricVals">{html.escape(_metric_values(case_rows, "epa_rpe_time_1s_trans_rmse_m", ndigits=3, unit="m"))}</span></td>'
                f'<td><span class="metricVals">{html.escape(_metric_values(case_rows, "epa_rpe_time_1s_rot_rmse_deg", fallback_key="epa_orientation_rpe_time_1s_rmse_deg", ndigits=3, unit="deg"))}</span></td>'
                f'<td><span class="metricVals">{html.escape(_valid_distance_values(case_rows))}</span></td>'
                f'<td>{html.escape(_diagnosis_text(case_rows))}<div class="muted">{html.escape(_text_values(case_rows, "epa_sr_reliability_status"))}</div></td>'
                f'<td>{_advanced_details(first_row)}</td>'
                '</tr>'
            )

    css = """
:root {
  --bg:#f7f9fc; --panel:#ffffff; --text:#172033; --muted:#667085; --line:#dde5f0;
  --head:#f1f5f9; --ok:#e8f7ef; --okText:#166534; --warn:#fff6df; --warnText:#92400e;
  --bad:#ffecec; --badText:#b42318; --accent:#2563eb; --mark:#e11d48;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
.page { padding:24px; }
.topbar { display:flex; justify-content:space-between; gap:16px; align-items:flex-end; margin-bottom:16px; }
h1 { font-size:22px; line-height:1.2; margin:0 0 6px; letter-spacing:0; }
.summary { color:var(--muted); margin:0; font-size:13px; }
.stats { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
.stat { border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:8px 10px; min-width:74px; text-align:center; }
.stat b { display:block; font-size:18px; }
.stat span { color:var(--muted); font-size:12px; }
.toolbar { display:flex; gap:8px; align-items:center; margin-bottom:12px; }
input { width:min(420px, 100%); height:34px; border:1px solid var(--line); border-radius:8px; padding:0 10px; background:#fff; color:var(--text); }
button { height:34px; border:1px solid var(--line); background:#fff; color:#344054; border-radius:8px; padding:0 10px; cursor:pointer; }
button:hover { border-color:#b6c2d2; background:#f8fafc; }
.tableWrap { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:auto; box-shadow:0 1px 2px rgba(15,23,42,.04); }
table { border-collapse:separate; border-spacing:0; width:100%; min-width:1120px; font-size:13px; }
th, td { border-bottom:1px solid var(--line); padding:10px 12px; vertical-align:top; text-align:left; }
th { position:sticky; top:0; z-index:1; background:var(--head); color:#475467; font-weight:650; }
tbody tr:last-child td { border-bottom:0; }
tr.warn { background:#fffaf0; }
tr.bad { background:#fff7f7; }
tr.caseStart td { border-top:2px solid #cbd5e1; }
.caseCell { display:flex; gap:9px; align-items:flex-start; min-width:230px; }
.caseRepeat { min-width:130px; }
.caseMarker { width:14px; height:14px; min-width:14px; padding:0; border-radius:50%; margin-top:2px; border:1px solid #cbd5e1; background:#fff; }
.caseMarker.marked { background:var(--mark); border-color:var(--mark); box-shadow:0 0 0 3px rgba(225,29,72,.12); }
.caseLink { color:var(--accent); text-decoration:underline; text-underline-offset:3px; font-weight:650; }
.caseSub { margin-top:6px; }
	.metricVals { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; line-height:1.45; white-space:nowrap; }
	.muted { color:var(--muted); font-size:12px; line-height:1.45; }
	.auditLine { margin-bottom:3px; }
	.advanced summary { cursor:pointer; color:var(--accent); font-weight:650; }
	.advanced div { margin-top:3px; line-height:1.35; }
	.pill { display:inline-flex; align-items:center; height:22px; padding:0 8px; border-radius:999px; font-size:12px; font-weight:650; }
.methodPill { display:inline-flex; align-items:center; height:24px; padding:0 8px; border-radius:7px; background:#eef2ff; color:#344054; font-weight:650; white-space:nowrap; text-decoration:none; }
.okPill { background:var(--ok); color:var(--okText); }
.warnPill { background:#fdecc8; color:var(--warnText); }
.badPill { background:#ffd6d6; color:var(--badText); }
.hint { color:var(--muted); font-size:12px; margin-top:10px; }
"""
    if matrix_layout:
        summary_text = (
            f"Matrix layout: <b>{len(method_labels)}</b> methods, one row per method. "
            "The main table shows public metrics; internal diagnostics are under details."
        )
        header_cells = [
            "<th style='width:250px'>case</th>",
            "<th style='width:132px'>method</th>",
            "<th>mode</th>",
            "<th>local SR distance</th>",
            "<th>local SR time</th>",
            "<th>APE RMSE</th>",
            "<th>1s RPE<br><span class='muted'>trans + rot</span></th>",
            "<th>valid distance</th>",
            "<th style='width:190px'>notes</th>",
            "<th style='width:210px'>details</th>",
        ]
    else:
        summary_text = (
            f"Metrics are grouped as <b>{html.escape(method_hint)}</b>. "
            "The main table shows public metrics; internal diagnostics are under details. "
            "Click a case name to open its interactive replay."
        )
        header_cells = [
            "<th style='width:270px'>case</th>",
            f"<th>method<br><span class='muted'>({html.escape(method_hint)})</span></th>",
            f"<th>mode<br><span class='muted'>({html.escape(method_hint)})</span></th>",
            f"<th>local SR distance<br><span class='muted'>({html.escape(method_hint)})</span></th>",
            f"<th>local SR time<br><span class='muted'>({html.escape(method_hint)})</span></th>",
            f"<th>APE RMSE<br><span class='muted'>({html.escape(method_hint)})</span></th>",
            f"<th>1s RPE trans<br><span class='muted'>({html.escape(method_hint)})</span></th>",
            f"<th>1s RPE rot<br><span class='muted'>({html.escape(method_hint)})</span></th>",
            f"<th>valid distance<br><span class='muted'>({html.escape(method_hint)})</span></th>",
            "<th style='width:320px'>notes</th>",
            "<th style='width:210px'>details</th>",
        ]
    doc = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>EPA Benchmark Summary</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<main class='page'>",
        "<div class='topbar'>",
        "<div>",
        "<h1>EPA Benchmark Summary</h1>",
        f"<p class='summary'>{summary_text}</p>",
        "</div>",
        "<div class='stats'>",
        f"<div class='stat'><b>{total}</b><span>cases</span></div>",
        f"<div class='stat'><b>{status_counts.get('ok', 0)}</b><span>ok</span></div>",
        f"<div class='stat'><b>{status_counts.get('warning', 0)}</b><span>warn</span></div>",
        f"<div class='stat'><b>{status_counts.get('failed', 0)}</b><span>failed</span></div>",
        "</div>",
        "</div>",
        "<div class='toolbar'>",
        "<input id='filter' placeholder='Filter by case, status, diagnosis, confidence'>",
        "<button id='showMarked' type='button'>Marked</button>",
        "<button id='showAll' type='button'>All</button>",
        "</div>",
        "<div class='tableWrap'>",
        "<table>",
        "<thead><tr>",
        *header_cells,
        "</tr></thead><tbody>",
        *table_rows,
        "</tbody>",
        "</table>",
        "</div>",
        "<p class='hint'>The red marker is stored in this browser via localStorage, so you can flag cases while reviewing.</p>",
        "</main>",
        """<script>
const KEY = 'epaSummaryMarkers:' + location.pathname;
const marked = new Set(JSON.parse(localStorage.getItem(KEY) || '[]'));
function save() { localStorage.setItem(KEY, JSON.stringify([...marked].sort())); }
function paint() {
  document.querySelectorAll('tr[data-case]').forEach(tr => {
    const btn = tr.querySelector('.caseMarker');
    if (btn) btn.classList.toggle('marked', marked.has(tr.dataset.case));
  });
}
function applyFilter(markedOnly=false) {
  const q = (document.getElementById('filter').value || '').toLowerCase();
  document.querySelectorAll('tbody tr[data-case]').forEach(tr => {
    const text = (tr.dataset.search || tr.innerText || '').toLowerCase();
    const textMatch = !q || text.includes(q);
    const markMatch = !markedOnly || marked.has(tr.dataset.case);
    tr.style.display = (textMatch && markMatch) ? '' : 'none';
  });
}
document.querySelectorAll('.caseMarker').forEach(btn => btn.addEventListener('click', event => {
  const tr = event.target.closest('tr[data-case]');
  const key = tr.dataset.case;
  marked.has(key) ? marked.delete(key) : marked.add(key);
  save(); paint();
}));
document.getElementById('filter').addEventListener('input', () => applyFilter(false));
document.getElementById('showMarked').addEventListener('click', () => applyFilter(true));
document.getElementById('showAll').addEventListener('click', () => {
  document.getElementById('filter').value = '';
  applyFilter(false);
});
paint();
</script>""",
        "</body>",
        "</html>",
    ]
    path.write_text("\n".join(doc) + "\n", encoding="utf-8")



