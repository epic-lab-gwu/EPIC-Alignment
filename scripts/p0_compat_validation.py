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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = base / stamp
    idx = 1
    while out.exists():
        out = base / f"{stamp}_{idx:02d}"
        idx += 1
    out.mkdir(parents=True, exist_ok=True)
    return out


def _case_paths(data_root: Path, case_name: str) -> tuple[Path, Path]:
    gt_candidates = [
        data_root / "lamaria-hard" / "hard" / f"{case_name}.txt",
        data_root / "AlignAnything2" / "AlignAnything2" / "GT" / "lamaria" / "hard" / f"{case_name}.txt",
    ]
    gt = next((p for p in gt_candidates if p.is_file()), None)
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


def _prepare_case_dir(root: Path, case_name: str, gt: Path, est: Path) -> Path:
    case_dir = root / case_name
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
    elapsed = float(time.perf_counter() - t0)
    text = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    return int(proc.returncode), elapsed, text


def _parse_saving_output(text: str) -> Path | None:
    m = re.search(r"Saving outputs to:\s*(.+)", text)
    if not m:
        return None
    return Path(m.group(1).strip()).expanduser().resolve()


def _latest_metrics_under(root: Path) -> Path | None:
    matches = sorted(root.rglob("metrics.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _get(d: dict[str, Any], *keys: str, default: Any = math.nan) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_epa_metrics(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    stage = "step3"
    return {
        "eval_source": _get(payload, "metadata", "eval_source", default="unknown"),
        "eval_align": _get(payload, "metadata", "eval_align", default=""),
        "step3_mode": _get(payload, "step3_selection", "step3_alignment_mode", default=""),
        "matched": _get(payload, "time_alignment", "evo_matches_equivalent", default=math.nan),
        "ape_rmse_m": _get(payload, "pose_metrics", "ape", stage, "translation_part", "rmse"),
        "rpe_rmse_m": _get(payload, "pose_metrics", "rpe", stage, "translation_part", "rmse"),
        "rpe_time_1s_rmse_m": _get(payload, "pose_metrics", "rpe_time_1s", stage, "translation_part", "rmse"),
        "sr_distance_pct": 100.0
        * float(
            _get(
                payload,
                "pose_metrics",
                "valid_segment",
                stage,
                "success",
                "success_rate_distance",
                default=math.nan,
            )
        ),
        "sr_time_pct": 100.0
        * float(
            _get(
                payload,
                "pose_metrics",
                "valid_segment",
                stage,
                "success",
                "success_rate_time",
                default=math.nan,
            )
        ),
    }


def _extract_ov_eval_stdout(text: str) -> dict[str, Any]:
    json_match = re.search(r"EPA_COMPAT_RESULT_JSON\s+({.*})", text)
    if json_match:
        payload = json.loads(json_match.group(1))
        return {
            "eval_source": payload.get("eval_source", "unknown"),
            "eval_align": "compat_error_singlerun",
            "step3_mode": "",
            "matched": payload.get("matched", math.nan),
            "ape_rmse_m": payload.get("ape_rmse_m", math.nan),
            "rpe_rmse_m": math.nan,
            "rpe_time_1s_rmse_m": payload.get("rpe_time_1s_rmse_m", math.nan),
            "sr_distance_pct": payload.get("sr_distance_pct", math.nan),
            "sr_time_pct": payload.get("sr_time_pct", math.nan),
        }

    def grab(pattern: str, default: Any = math.nan) -> Any:
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    return {
        "eval_source": grab(r"Eval source\s*=\s*(\S+)", "unknown"),
        "eval_align": "compat_error_singlerun",
        "step3_mode": "",
        "matched": float(grab(r"Aligned pairs:\s*([0-9]+)", math.nan)),
        "ape_rmse_m": float(grab(r"rmse_ori\s*=\s*[-+0-9.eE]+ \| rmse_pos\s*=\s*([-+0-9.eE]+)", math.nan)),
        "rpe_rmse_m": math.nan,
        "rpe_time_1s_rmse_m": float(
            grab(r"1s time - rmse_ori\s*=\s*[-+0-9.eE]+ \| rmse_pos\s*=\s*([-+0-9.eE]+)", math.nan)
        ),
        "sr_distance_pct": float(grab(r"SR@[-+0-9.eE]+m - distance\s*=\s*([-+0-9.eE]+)%", math.nan)),
        "sr_time_pct": float(grab(r"SR@[-+0-9.eE]+m - distance\s*=\s*[-+0-9.eE]+% \| time\s*=\s*([-+0-9.eE]+)%", math.nan)),
    }


def _row(
    *,
    case_name: str,
    entry: str,
    status: str,
    elapsed_s: float,
    metrics: dict[str, Any],
    output: Path | str,
    note: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "case": case_name,
        "entry": entry,
        "status": status,
        "elapsed_s": elapsed_s,
        "output": str(output),
        "note": note,
    }
    out.update(metrics)
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    csv_path = out_dir / "compat_validation.csv"
    fieldnames = [
        "case",
        "entry",
        "status",
        "eval_source",
        "eval_align",
        "step3_mode",
        "matched",
        "ape_rmse_m",
        "rpe_rmse_m",
        "rpe_time_1s_rmse_m",
        "sr_distance_pct",
        "sr_time_pct",
        "elapsed_s",
        "output",
        "note",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# P0 Compatibility Validation",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- csv: `{csv_path}`",
        "",
        "| case | entry | status | eval_source | APE RMSE m | RPE 1s RMSE m | SR dist % | time s | note |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case", "")),
                    str(row.get("entry", "")),
                    str(row.get("status", "")),
                    str(row.get("eval_source", "")),
                    _fmt(row.get("ape_rmse_m")),
                    _fmt(row.get("rpe_time_1s_rmse_m")),
                    _fmt(row.get("sr_distance_pct"), 2),
                    _fmt(row.get("elapsed_s"), 3),
                    str(row.get("note", "")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Direct-vs-Compatibility Deltas", ""])
    lines.append("| case | entry | dAPE m | dRPE1s m | dSR dist pct | verdict |")
    lines.append("|---|---|---:|---:|---:|---|")
    for case_name in sorted({str(r["case"]) for r in rows}):
        direct = next((r for r in rows if r["case"] == case_name and r["entry"] == "direct_epa"), None)
        if direct is None or direct.get("status") != "ok":
            continue
        for row in [r for r in rows if r["case"] == case_name and r["entry"] != "direct_epa" and r.get("status") == "ok"]:
            d_ape = abs(float(row.get("ape_rmse_m", math.nan)) - float(direct.get("ape_rmse_m", math.nan)))
            d_rpe1s = abs(
                float(row.get("rpe_time_1s_rmse_m", math.nan)) - float(direct.get("rpe_time_1s_rmse_m", math.nan))
            )
            d_sr = abs(float(row.get("sr_distance_pct", math.nan)) - float(direct.get("sr_distance_pct", math.nan)))
            finite = all(math.isfinite(v) for v in (d_ape, d_rpe1s, d_sr))
            verdict = "ok" if finite and d_ape < 1e-6 and d_rpe1s < 1e-6 and d_sr < 1e-6 else "review"
            lines.append(
                f"| {case_name} | {row['entry']} | {_fmt(d_ape, 8)} | {_fmt(d_rpe1s, 8)} | {_fmt(d_sr, 8)} | {verdict} |"
            )

    (out_dir / "compat_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate EPA P0 compatibility entries and runtime on LaMARia hard SVO Mono.")
    p.add_argument("--data-root", default="/home/yifu/epa_data", help="Root containing AlignAnything2 and lamaria-hard.")
    p.add_argument("--cases", nargs="*", default=list(DEFAULT_LAMARIA_HARD_CASES), help="LaMARia hard case names.")
    p.add_argument("--out-dir", default="", help="Output directory. Default: outputs/p0_compat_validation/run_<stamp>.")
    p.add_argument("--t-max-diff", type=float, default=0.02)
    p.add_argument("--skip-ov-eval", action="store_true", help="Skip ov_eval_compat error_singlerun.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo = _repo_root()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _run_dir(repo / "outputs" / "p0_compat_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    case_root = out_dir / "openvins_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    env = _env(repo)
    data_root = Path(args.data_root).expanduser().resolve()

    rows: list[dict[str, Any]] = []
    for case_name in args.cases:
        gt, est = _case_paths(data_root, case_name)
        case_dir = _prepare_case_dir(case_root, case_name, gt, est)

        direct_root = out_dir / "direct_epa" / case_name
        direct_cmd = [
            sys.executable,
            "-m",
            "epa.cli",
            str(gt),
            str(est),
            "--gt-format",
            "tum",
            "--est-format",
            "tum",
            "--t-max-diff",
            str(float(args.t_max_diff)),
            "--no-plot",
            "--output-root",
            str(direct_root),
            "--run-label",
            case_name,
        ]
        rc, elapsed, text = _run_cmd(direct_cmd, cwd=repo, env=env, log_path=out_dir / "logs" / case_name / "direct_epa.log")
        direct_run = _parse_saving_output(text) or direct_root
        metrics_path = direct_run / "metrics.json"
        if rc == 0 and metrics_path.is_file():
            rows.append(
                _row(
                    case_name=case_name,
                    entry="direct_epa",
                    status="ok",
                    elapsed_s=elapsed,
                    metrics=_extract_epa_metrics(metrics_path),
                    output=direct_run,
                )
            )
        else:
            rows.append(_row(case_name=case_name, entry="direct_epa", status="failed", elapsed_s=elapsed, metrics={}, output=direct_run, note=f"rc={rc}"))

        openvins_root = out_dir / "openvins_runner" / case_name
        openvins_cmd = [
            sys.executable,
            "-m",
            "epa.openvins_runner",
            str(case_dir),
            "--align-mode",
            "se3",
            "--no-plot",
            "--keep-output",
            "--save-root",
            str(openvins_root),
        ]
        rc, elapsed, text = _run_cmd(
            openvins_cmd, cwd=repo, env=env, log_path=out_dir / "logs" / case_name / "openvins_runner.log"
        )
        metrics_path = _latest_metrics_under(openvins_root) if rc == 0 else None
        if rc == 0 and metrics_path is not None:
            rows.append(
                _row(
                    case_name=case_name,
                    entry="openvins_runner",
                    status="ok",
                    elapsed_s=elapsed,
                    metrics=_extract_epa_metrics(metrics_path),
                    output=metrics_path.parent,
                )
            )
        else:
            rows.append(_row(case_name=case_name, entry="openvins_runner", status="failed", elapsed_s=elapsed, metrics={}, output=openvins_root, note=f"rc={rc}"))

        toolchain_root = out_dir / "case_toolchain" / case_name
        toolchain_cmd = [
            sys.executable,
            "-m",
            "epa.case_toolchain",
            "--case-dir",
            str(case_dir),
            "--align-mode",
            "se3",
            "--no-plot",
            "--out-root",
            str(toolchain_root),
        ]
        rc, elapsed, text = _run_cmd(
            toolchain_cmd, cwd=repo, env=env, log_path=out_dir / "logs" / case_name / "case_toolchain.log"
        )
        metrics_path = _latest_metrics_under(toolchain_root / "main_workspace") if rc == 0 else None
        note = ""
        timing_json = toolchain_root / "timing_summary.json"
        if timing_json.is_file():
            timing = json.loads(timing_json.read_text(encoding="utf-8"))
            note = "recorded_s=" + _fmt(timing.get("total_recorded_s", math.nan), 3)
            if "openvins_runner_s" in timing:
                note += "; unexpected_openvins_runner_s=" + _fmt(timing["openvins_runner_s"], 3)
        if rc == 0 and metrics_path is not None:
            rows.append(
                _row(
                    case_name=case_name,
                    entry="case_toolchain",
                    status="ok",
                    elapsed_s=elapsed,
                    metrics=_extract_epa_metrics(metrics_path),
                    output=toolchain_root,
                    note=note,
                )
            )
        else:
            rows.append(_row(case_name=case_name, entry="case_toolchain", status="failed", elapsed_s=elapsed, metrics={}, output=toolchain_root, note=f"rc={rc} {note}"))

        if not bool(args.skip_ov_eval):
            ov_eval_cmd = [
                sys.executable,
                "-m",
                "epa.ov_eval_compat",
                "error_singlerun",
                "se3",
                str(gt),
                str(est),
                "--max-diff",
                str(float(args.t_max_diff)),
            ]
            rc, elapsed, text = _run_cmd(
                ov_eval_cmd, cwd=repo, env=env, log_path=out_dir / "logs" / case_name / "ov_eval_compat.log"
            )
            if rc == 0:
                rows.append(
                    _row(
                        case_name=case_name,
                        entry="ov_eval_compat",
                        status="ok",
                        elapsed_s=elapsed,
                        metrics=_extract_ov_eval_stdout(text),
                        output=out_dir / "logs" / case_name / "ov_eval_compat.log",
                    )
                )
            else:
                rows.append(_row(case_name=case_name, entry="ov_eval_compat", status="failed", elapsed_s=elapsed, metrics={}, output=out_dir / "logs" / case_name / "ov_eval_compat.log", note=f"rc={rc}"))

        _write_outputs(rows, out_dir)

    _write_outputs(rows, out_dir)
    print(f"Wrote: {out_dir / 'compat_validation.csv'}")
    print(f"Wrote: {out_dir / 'compat_validation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
