#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_LAMARIA_HARD_CASES = ("R_08_hard", "R_09_hard", "R_10_hard")
DEFAULT_ENTRIES = (
    "epa_step3",
    "openvins_compat",
    "ov_eval_compat",
    "epa_se3",
    "epa_posyaw",
    "epa_sim3",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _unique_run_dir(base: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"{label}_{stamp}"
    idx = 1
    while out.exists():
        out = base / f"{label}_{stamp}_{idx:02d}"
        idx += 1
    out.mkdir(parents=True, exist_ok=True)
    return out


def _case_paths(data_root: Path, case_name: str) -> tuple[Path, Path]:
    gt_candidates = [
        data_root / "lamaria-hard" / "hard" / f"{case_name}.txt",
        data_root / "AlignAnything2" / "AlignAnything2" / "GT" / "lamaria" / "hard" / f"{case_name}.txt",
    ]
    gt = next((path for path in gt_candidates if path.is_file()), None)
    est = (
        data_root
        / "AlignAnything2"
        / "AlignAnything2"
        / "benchmark"
        / "lamaria"
        / "hard"
        / "pose"
        / "svo_mono"
        / case_name
        / "svo_poses.txt"
    )
    if gt is None:
        raise FileNotFoundError(f"GT not found for {case_name}. Tried: {gt_candidates}")
    if not est.is_file():
        raise FileNotFoundError(f"EST not found for {case_name}: {est}")
    return gt.resolve(), est.resolve()


def _prepare_openvins_case(case_root: Path, case_name: str, gt: Path, est: Path) -> Path:
    case_dir = case_root / case_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    os.symlink(gt, case_dir / "stamped_groundtruth.txt")
    os.symlink(est, case_dir / "stamped_traj_estimate.txt")
    return case_dir


def _env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo / "src")
    env["PYTHONPATH"] = src + ((os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
    return env


def _run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
    elapsed_s = float(time.perf_counter() - t0)
    text = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    return int(proc.returncode), elapsed_s, text


def _parse_run_dir(text: str) -> Path | None:
    patterns = [
        r"Saving outputs to:\s*(.+)",
        r"\[case\] saved to:\s*(.+)",
        r"\[case\] output dir:\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return Path(match.group(1).strip()).expanduser().resolve()
    return None


def _latest_metrics_under(root: Path) -> Path | None:
    matches = sorted(root.rglob("metrics.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _get(block: dict[str, Any], *keys: str, default: Any = math.nan) -> Any:
    cur: Any = block
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _extract_metrics_json(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    success = _get(payload, "pose_metrics", "valid_segment", "step3", "success", default={})
    eval_alignment = _get(payload, "metadata", "eval_alignment", default={})
    diagnosis_tags = _get(payload, "case_diagnostics", "diagnosis_tags", default=[])
    if isinstance(diagnosis_tags, list):
        diagnosis_tag_text = ",".join(str(x) for x in diagnosis_tags)
    else:
        diagnosis_tag_text = str(diagnosis_tags)
    return {
        "epa_eval_source": _get(payload, "metadata", "eval_source", default="unknown"),
        "epa_eval_align": _get(payload, "metadata", "eval_align", default=""),
        "epa_ate_rmse_step3_m": _get(payload, "pose_metrics", "ape", "step3", "translation_part", "rmse"),
        "epa_rpe_time_1s_trans_rmse_m": _get(payload, "pose_metrics", "rpe_time_1s", "step3", "translation_part", "rmse"),
        "epa_rpe_time_1s_rot_rmse_deg": _get(payload, "pose_metrics", "rpe_time_1s", "step3", "rotation_angle_deg", "rmse"),
        "epa_sr_distance_raw": _get(success, "raw_success_rate_distance"),
        "epa_sr_distance": _get(success, "success_rate_distance"),
        "epa_sr_distance_gated": _get(success, "success_rate_distance_reliability_gated"),
        "epa_sr_time_raw": _get(success, "raw_success_rate_time"),
        "epa_sr_time": _get(success, "success_rate_time"),
        "epa_sr_time_gated": _get(success, "success_rate_time_reliability_gated"),
        "epa_sr_reliability_status": _get(success, "sr_reliability_status", default=""),
        "epa_sr_warning_explanation": _get(success, "sr_warning_explanation", default=""),
        "epa_sim3_may_mask_failure": str(bool(_get(success, "sim3_may_mask_failure", default=False))),
        "epa_case_status": _get(success, "case_status", default=""),
        "epa_global_gate_failed": str(bool(_get(success, "global_gate_failed", default=False))),
        "epa_global_gate_value_m": _get(success, "global_gate_value_m"),
        "epa_global_gate_m": _get(success, "global_gate_m"),
        "epa_valid_distance_m": _get(success, "valid_distance_m"),
        "epa_total_distance_m": _get(success, "total_distance_m"),
        "epa_step3_alignment_mode": _get(payload, "step3_selection", "step3_alignment_mode", default=""),
        "epa_sim3_scale": _get(eval_alignment, "sim3_scale", default=_get(eval_alignment, "align_scale")),
        "epa_sim3_reliable": str(bool(_get(eval_alignment, "sim3_reliable", default=True)))
        if "sim3" in str(_get(eval_alignment, "align_mode", default="")).lower()
        else "",
        "epa_sim3_warning": _get(eval_alignment, "sim3_warning", default=""),
        "epa_orientation_unstable": str(bool(_get(payload, "orientation_diagnostics", "orientation_unstable", default=False))),
        "epa_orientation_rpe_time_1s_rmse_deg": _get(
            payload, "orientation_diagnostics", "orientation_rpe_time_1s_rmse_deg"
        ),
        "epa_orientation_warning": _get(payload, "orientation_diagnostics", "orientation_warning", default=""),
        "epa_case_diagnosis_primary": _get(payload, "case_diagnostics", "diagnosis_primary", default=""),
        "epa_case_diagnosis_summary": _get(payload, "case_diagnostics", "diagnosis_summary", default=""),
        "epa_case_diagnosis_tags": diagnosis_tag_text,
    }


def _extract_ov_eval_stdout(text: str) -> dict[str, Any]:
    match = re.search(r"EPA_COMPAT_RESULT_JSON\s+({.*})", text)
    payload: dict[str, Any] = json.loads(match.group(1)) if match else {}
    sr_dist = _as_float(payload.get("sr_distance_pct")) / 100.0
    sr_time = _as_float(payload.get("sr_time_pct")) / 100.0
    return {
        "epa_eval_source": payload.get("eval_source", "epa_ov_eval_compat"),
        "epa_eval_align": "ov_eval_compat",
        "epa_ate_rmse_step3_m": payload.get("ape_rmse_m", math.nan),
        "epa_rpe_time_1s_trans_rmse_m": payload.get("rpe_time_1s_rmse_m", math.nan),
        "epa_sr_distance": sr_dist,
        "epa_sr_time": sr_time,
        "epa_sr_distance_raw": sr_dist,
        "epa_sr_time_raw": sr_time,
        "epa_sr_distance_gated": sr_dist,
        "epa_sr_time_gated": sr_time,
        "epa_sr_reliability_status": "ok" if bool(payload.get("eval_reliable", True)) else "failed",
        "epa_sr_warning_explanation": "OV compat stdout summary; inspect direct EPA report for full SR audit.",
        "epa_case_status": "compat_stdout",
    }


def _base_row(
    *,
    case_name: str,
    method: str,
    gt: Path,
    est: Path,
    status: str,
    elapsed_s: float,
    run_dir: Path | None,
    log_path: Path,
    error: str = "",
) -> dict[str, Any]:
    return {
        "case": f"lamaria_{case_name}",
        "dataset": "lamaria_hard_svo_mono",
        "method": method,
        "status": status,
        "gt": str(gt),
        "est": str(est),
        "epa_status": status,
        "evo_status": "not_run",
        "epa_runtime_s": elapsed_s,
        "epa_run_dir": "" if run_dir is None else str(run_dir),
        "epa_stdout_log": str(log_path),
        "epa_stderr_log": str(log_path),
        "error": error,
    }


def _direct_epa_cmd(python_bin: str, gt: Path, est: Path, out_root: Path, run_label: str, eval_align: str) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "epa.cli",
        "--gt-csv",
        str(gt),
        "--gt-format",
        "tum",
        "--est-path",
        str(est),
        "--est-format",
        "tum",
        "--t-max-diff",
        "0.02",
        "--output-root",
        str(out_root),
        "--run-label",
        run_label,
        "--no-plot",
    ]
    if eval_align:
        cmd.extend(["--eval-align", eval_align])
    return cmd


def _run_entry(
    *,
    entry: str,
    case_name: str,
    gt: Path,
    est: Path,
    run_dir: Path,
    repo: Path,
    env: dict[str, str],
    python_bin: str,
    dry_run: bool,
) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    entry_runs = run_dir / "runs" / entry
    method_eval_align = {
        "direct_epa": "epa_step3",
        "mode_se3": "epa_se3",
        "mode_posyaw": "posyaw",
        "mode_sim3": "epa_sim3",
        "epa_step3": "epa_step3",
        "epa_se3": "epa_se3",
        "epa_se3_eval": "epa_se3",
        "epa_posyaw": "posyaw",
        "epa_sim3": "epa_sim3",
    }
    if entry in method_eval_align:
        cmd = _direct_epa_cmd(
            python_bin,
            gt,
            est,
            entry_runs,
            f"{case_name}_{entry}",
            method_eval_align[entry],
        )
    elif entry == "openvins_compat":
        case_dir = _prepare_openvins_case(run_dir / "openvins_cases", case_name, gt, est)
        cmd = [
            python_bin,
            "-m",
            "epa.openvins_runner",
            "--keep-output",
            "--no-plot",
            "--save-root",
            str(entry_runs),
            "--align-mode",
            "none",
            str(case_dir),
        ]
    elif entry == "ov_eval_compat":
        cmd = [
            python_bin,
            "-m",
            "epa.ov_eval_compat",
            "error_singlerun",
            "none",
            str(gt),
            str(est),
            "--max-diff",
            "0.02",
        ]
    else:
        raise ValueError(f"Unknown regression entry: {entry}")

    log_path = logs_dir / case_name / f"{entry}.log"
    if dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
        row = _base_row(
            case_name=case_name,
            method=entry,
            gt=gt,
            est=est,
            status="dry_run",
            elapsed_s=0.0,
            run_dir=None,
            log_path=log_path,
        )
        row["error"] = "dry_run"
        return row

    rc, elapsed_s, text = _run_cmd(cmd, cwd=repo, env=env, log_path=log_path)
    parsed_run_dir = _parse_run_dir(text)
    metrics_path = None
    if parsed_run_dir is not None:
        metrics_path = parsed_run_dir / "metrics.json"
    if metrics_path is None or not metrics_path.exists():
        metrics_path = _latest_metrics_under(entry_runs) if entry != "ov_eval_compat" else None

    status = "ok" if rc == 0 else "failed"
    row = _base_row(
        case_name=case_name,
        method=entry,
        gt=gt,
        est=est,
        status=status,
        elapsed_s=elapsed_s,
        run_dir=metrics_path.parent if metrics_path is not None and metrics_path.exists() else parsed_run_dir,
        log_path=log_path,
        error="" if rc == 0 else f"returncode={rc}",
    )
    if rc == 0 and metrics_path is not None and metrics_path.exists():
        row.update(_extract_metrics_json(metrics_path))
    elif rc == 0 and entry == "ov_eval_compat":
        row.update(_extract_ov_eval_stdout(text))
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(rows: list[dict[str, Any]], out_dir: Path, *, entries: list[str], cases: list[str]) -> None:
    ok = sum(1 for row in rows if row.get("status") == "ok")
    lines = [
        "# EPA Regression Benchmark",
        "",
        f"- Cases: `{', '.join(cases)}`",
        f"- Entries: `{', '.join(entries)}`",
        f"- OK rows: `{ok}/{len(rows)}`",
        "",
        "Outputs:",
        "",
        "- [summary.html](summary.html)",
        "- [summary.csv](summary.csv)",
        "- [summary.md](summary.md)",
        "",
    ]
    out_dir.joinpath("regression_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed EPA regression benchmark matrix.")
    parser.add_argument("--data-root", default=os.environ.get("EPA_DATA_ROOT", str(Path.home() / "epa_data")))
    parser.add_argument("--output-root", default="outputs/regression_benchmark")
    parser.add_argument("--cases", default=",".join(DEFAULT_LAMARIA_HARD_CASES), help="Comma-separated LaMARia hard cases.")
    parser.add_argument("--entries", default=",".join(DEFAULT_ENTRIES), help="Comma-separated regression entries.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = _repo_root()
    sys.path.insert(0, str(repo / "src"))
    from epa.benchmark.benchmark_harness import _write_summary_html, _write_summary_md

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = _unique_run_dir((repo / args.output_root).resolve(), "regression")
    env = _env(repo)
    cases = [item.strip() for item in str(args.cases).split(",") if item.strip()]
    entries = [item.strip() for item in str(args.entries).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []

    config = {"data_root": str(data_root), "cases": cases, "entries": entries, "dry_run": bool(args.dry_run)}
    out_dir.joinpath("regression_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for case_name in cases:
        gt, est = _case_paths(data_root, case_name)
        for entry in entries:
            print(f"[{case_name}] {entry}")
            row = _run_entry(
                entry=entry,
                case_name=case_name,
                gt=gt,
                est=est,
                run_dir=out_dir,
                repo=repo,
                env=env,
                python_bin=str(args.python_bin),
                dry_run=bool(args.dry_run),
            )
            rows.append(row)
            print(f"  -> {row.get('status')} {row.get('epa_run_dir', '')}")

    _write_csv(rows, out_dir / "summary.csv")
    _write_summary_md(rows, out_dir / "summary.md")
    _write_summary_html(rows, out_dir / "summary.html")
    _write_readme(rows, out_dir, entries=entries, cases=cases)
    print(f"Regression summary: {out_dir / 'summary.html'}")
    return 0 if all(row.get("status") in {"ok", "dry_run"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
