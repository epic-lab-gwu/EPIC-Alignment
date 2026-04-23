from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_run_dir(run_dir_hint: Path) -> Path:
    run_dir_hint = run_dir_hint.expanduser().resolve()
    prepared = run_dir_hint / "prepared_tum"
    if prepared.is_dir():
        return run_dir_hint

    if run_dir_hint.is_dir():
        candidates = []
        for p in run_dir_hint.iterdir():
            if p.is_dir() and p.name.startswith("run_") and (p / "prepared_tum").is_dir():
                candidates.append(p)
        if candidates:
            candidates.sort(key=lambda p: p.name, reverse=True)
            return candidates[0]

    raise FileNotFoundError(
        "Cannot resolve harness run directory.\n"
        f"- given: {run_dir_hint}\n"
        "Expected either:\n"
        "1) a concrete run dir containing prepared_tum/, or\n"
        "2) a parent dir containing run_*/prepared_tum/."
    )


def _discover_case_pairs(prepared_dir: Path) -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    for gt_path in sorted(prepared_dir.glob("*__gt.tum")):
        suffix = "__gt.tum"
        if not gt_path.name.endswith(suffix):
            continue
        case_id = gt_path.name[: -len(suffix)]
        est_path = prepared_dir / f"{case_id}__est.tum"
        if est_path.exists():
            pairs[case_id] = (gt_path, est_path)
    return pairs


def _build_epa_cmd(
    *,
    python_bin: str,
    gt_path: Path,
    est_path: Path,
    no_plot: bool,
    no_rerun: bool,
    passthrough: list[str],
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "epa.cli",
        "--gt-csv",
        str(gt_path),
        "--gt-format",
        "tum",
        "--est-path",
        str(est_path),
        "--est-format",
        "tum",
    ]
    if not no_plot:
        cmd.append("--plot")
    if not no_rerun:
        cmd.append("--rerun")
    cmd.extend(passthrough)
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run epa + rerun for a single case from alignanything harness prepared_tum."
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/alignanything_harness",
        help=(
            "Harness run dir (run_*/...) or parent harness dir. "
            "If a parent dir is given, latest run_* is selected."
        ),
    )
    parser.add_argument(
        "--case",
        default="",
        help="Case id, e.g. euroc_mav_MH_01_easy_rovio. Required when multiple cases exist.",
    )
    parser.add_argument("--list-cases", action="store_true", help="List available case IDs and exit.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to invoke `python -m epa.cli`.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable --plot when calling epa.")
    parser.add_argument("--no-rerun", action="store_true", help="Disable --rerun when calling epa.")
    parser.add_argument("--dry-run", action="store_true", help="Print epa command without executing.")
    return parser


def run(args: argparse.Namespace, passthrough: list[str] | None = None) -> int:
    passthrough = list(passthrough or [])
    run_dir = _resolve_run_dir(Path(args.run_dir))
    prepared_dir = run_dir / "prepared_tum"
    pairs = _discover_case_pairs(prepared_dir)
    if not pairs:
        raise FileNotFoundError(f"No case pairs found under: {prepared_dir}")

    if args.list_cases:
        print(f"Resolved run dir: {run_dir}")
        print("Available cases:")
        for case_id in sorted(pairs.keys()):
            print(f"- {case_id}")
        return 0

    selected_case = str(args.case).strip()
    if not selected_case:
        if len(pairs) == 1:
            selected_case = next(iter(pairs.keys()))
        else:
            sample = ", ".join(sorted(pairs.keys())[:8])
            raise ValueError(
                "Multiple cases found. Please pass --case <case_id>.\n"
                f"Found {len(pairs)} cases. Examples: {sample}"
            )

    if selected_case not in pairs:
        sample = ", ".join(sorted(pairs.keys())[:8])
        raise ValueError(
            f"Case not found: {selected_case}\n"
            f"Use --list-cases to inspect available IDs. Examples: {sample}"
        )

    gt_path, est_path = pairs[selected_case]
    cmd = _build_epa_cmd(
        python_bin=str(args.python_bin),
        gt_path=gt_path,
        est_path=est_path,
        no_plot=bool(args.no_plot),
        no_rerun=bool(args.no_rerun),
        passthrough=passthrough,
    )

    print(f"Resolved run dir: {run_dir}")
    print(f"Selected case: {selected_case}")
    print("Command:")
    print(" ".join(cmd))
    if bool(args.dry_run):
        return 0

    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return run(args, passthrough=passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
