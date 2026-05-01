from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _latest_run_dir(root: Path) -> Path | None:
    runs = sorted(
        [p for p in root.glob("run_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    return runs[-1] if runs else None


def _safe_root_label(path: Path) -> str:
    name = path.expanduser().resolve().name.strip()
    if not name:
        return "cases"
    label = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._-").lower()
    return label or "cases"


def _default_output_root(cwd: Path, cases_root: Path) -> Path:
    return cwd / "outputs" / f"{_safe_root_label(cases_root)}_benchall"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the EPA multi-case full benchmark workflow in one command."
    )
    p.add_argument(
        "cases_root_pos",
        nargs="?",
        metavar="cases_root",
        help="Root directory containing benchmark/ and GT/ folders.",
    )
    p.add_argument(
        "--cases-root",
        default=os.getenv("EPA_CASES_ROOT", "").strip() or os.getenv("EPA_ALIGNANYTHING_ROOT", "").strip(),
        help="Root directory containing benchmark/ and GT/ folders. Default: $EPA_CASES_ROOT.",
    )
    p.add_argument(
        "--output-root",
        default="",
        help="Root directory for benchmark runs. Default: outputs/<cases_root_name>_benchall",
    )
    p.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to run EPA commands.",
    )
    p.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch subcommands.",
    )
    p.add_argument(
        "--epa-src",
        default="src",
        help="EPA source directory appended into PYTHONPATH by the harness.",
    )
    p.add_argument(
        "--evo-repo",
        default=os.getenv("EVO_REPO", "").strip(),
        help="Local evo repository path. Default: $EVO_REPO.",
    )
    p.add_argument(
        "--t-max-diff",
        type=float,
        default=0.02,
        help="Timestamp association threshold.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k cases to show in summary plots.",
    )
    p.add_argument(
        "--case-pattern",
        action="append",
        default=[],
        help="Regex filter for case IDs. Can be passed multiple times.",
    )
    p.add_argument(
        "--methods",
        default="",
        help="Comma-separated method filter.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of cases to run.",
    )
    p.add_argument(
        "-j",
        "--jobs",
        default="auto",
        help="Number of benchmark cases to run in parallel, or 'auto'. Default: auto.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover cases without executing the benchmark.",
    )
    return p


def run(args: argparse.Namespace) -> int:
    cwd = Path.cwd().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    cases_root_value = str(getattr(args, "cases_root_pos", "") or getattr(args, "cases_root", "")).strip()
    cases_root = (
        Path(cases_root_value).expanduser().resolve()
        if cases_root_value
        else None
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if str(args.output_root).strip()
        else _default_output_root(cwd, cases_root if cases_root is not None else cwd)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    bench_cmd = [
        str(Path(args.python_bin).expanduser()),
        "-m",
        "epa.benchmark.alignanything_harness",
        "--output-root",
        str(output_root),
        "--repo-root",
        str(repo_root),
        "--python-bin",
        str(Path(args.python_bin).expanduser()),
        "--epa-src",
        str(args.epa_src),
        "--t-max-diff",
        str(float(args.t_max_diff)),
    ]
    if str(args.evo_repo).strip():
        bench_cmd.extend(["--evo-repo", str(Path(args.evo_repo).expanduser())])
    if cases_root is not None:
        bench_cmd.extend(["--cases-root", str(cases_root)])
    if str(args.methods).strip():
        bench_cmd.extend(["--methods", str(args.methods)])
    if int(args.limit) > 0:
        bench_cmd.extend(["--limit", str(int(args.limit))])
    bench_cmd.extend(["--jobs", str(args.jobs)])
    if bool(args.dry_run):
        bench_cmd.append("--dry-run")
    for pattern in args.case_pattern:
        if str(pattern).strip():
            bench_cmd.extend(["--case-pattern", str(pattern)])

    _run(bench_cmd, cwd=cwd)

    if bool(args.dry_run):
        return 0

    latest_run = _latest_run_dir(output_root)
    if latest_run is None:
        raise RuntimeError(f"No benchmark run directory found under: {output_root}")

    summary_csv = latest_run / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    plot_cmd = [
        str(Path(args.python_bin).expanduser()),
        "-m",
        "epa.benchmark.plot_summary",
        "--summary-csv",
        str(summary_csv),
        "--top-k",
        str(int(args.top_k)),
    ]
    _run(plot_cmd, cwd=cwd)

    latex_cmd = [
        str(Path(args.python_bin).expanduser()),
        "-m",
        "epa.benchmark.latex_summary",
        "--summary-csv",
        str(summary_csv),
    ]
    _run(latex_cmd, cwd=cwd)

    print()
    print(f"Done. Outputs saved under: {latest_run}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
