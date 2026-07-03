from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


_KNOWN_ALIGN_MODES = {
    "none",
    "epa_step3",
    "se3",
    "epa_se3",
    "epa_se3_eval",
    "sim3",
    "se3single",
    "posyaw",
    "posyawsingle",
}
_LEGACY_ALIGN_MODES = {"se3", "se3single", "posyawsingle"}


def _is_case_dir(path_str: str) -> bool:
    p = Path(path_str).expanduser()
    return p.is_dir() and (p / "stamped_groundtruth.txt").is_file() and (p / "stamped_traj_estimate.txt").is_file()


def _safe_case_label(path_str: str) -> str:
    name = Path(path_str).name.replace(" ", "_").replace("/", "_")
    return name if name else "case"


def _map_align_mode_to_eval_align(mode: str) -> str:
    m = str(mode).lower()
    if m == "none":
        return "none"
    if m == "epa_step3":
        return "epa_step3"
    if m in {"epa_se3", "epa_se3_eval"}:
        return "epa_se3"
    if m == "se3":
        return "epa_step3"
    if m == "sim3":
        return "epa_sim3"
    if m in {"se3single", "posyawsingle"}:
        return "origin"
    if m == "posyaw":
        return "posyaw"
    raise ValueError(f"Unsupported align_mode: {mode}")


def _parse_run_dir(stdout: str, stderr: str) -> Path | None:
    text = f"{stdout}\n{stderr}"
    m = re.search(r"Saving outputs to:\s*(.+)", text)
    if not m:
        return None
    return Path(m.group(1).strip()).expanduser().resolve()


def _unique_dst(base_root: Path, case_label: str) -> Path:
    base = base_root / case_label
    if not base.exists():
        return base
    idx = 1
    while True:
        cand = base_root / f"{case_label}_{idx:02d}"
        if not cand.exists():
            return cand
        idx += 1


def _list_interactive_backend_tokens(matplotlib_mod) -> set[str]:
    try:
        from matplotlib.backends import BackendFilter, backend_registry

        return {
            str(x).strip().lower().split(".")[-1]
            for x in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
    except Exception:
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return {
                    str(x).strip().lower().split(".")[-1]
                    for x in getattr(matplotlib_mod.rcsetup, "interactive_bk", [])
                }
        except Exception:
            return set()


def _show_plots_popup(plots_dir: Path, max_plots: int = 15) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] popup disabled: matplotlib not available ({exc})")
        return

    pngs = sorted(plots_dir.glob("*.png"))
    if not pngs:
        print(f"[warn] no png files in {plots_dir}")
        return

    try:
        current = str(matplotlib.get_backend()).lower()
        if "agg" in current:
            for cand in ("qtagg", "tkagg", "qt5agg", "wxagg"):
                try:
                    plt.switch_backend(cand)
                    break
                except Exception:
                    continue
    except Exception:
        pass

    interactive_tokens = _list_interactive_backend_tokens(matplotlib)
    current_token = str(matplotlib.get_backend()).strip().lower().split(".")[-1]
    if current_token not in interactive_tokens:
        print(f"[info] popup skipped: non-interactive backend '{matplotlib.get_backend()}'")
        return

    for p in pngs[: int(max_plots)]:
        try:
            img = plt.imread(str(p))
        except Exception:
            continue
        fig = plt.figure(figsize=(10.5, 6.2))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.set_title(p.name, fontsize=10)
        ax.axis("off")
        fig.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        print(f"[warn] popup show failed: {exc}")


def _run_timed(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    return proc, float(time.perf_counter() - t0)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run EPA full pipeline on one or multiple OpenVINS case folders."
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Case directories. Backward-compatible: <case_dir> <align_mode> is also accepted.",
    )
    p.add_argument(
        "--align-mode",
        default="",
        choices=sorted(_KNOWN_ALIGN_MODES),
        help="Alignment mode for compatibility with prior workflow.",
    )
    p.add_argument("--no-plot", action="store_true", help="Disable plots.")
    p.add_argument("--keep-output", action="store_true", help="Keep outputs on disk.")
    p.add_argument(
        "--save-root",
        default="",
        help="Destination root for saved run folders (multi-case default: /tmp/epa_batch).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    launch_cwd = Path.cwd().resolve()

    raw_inputs = list(args.inputs)
    align_mode = str(args.align_mode or "").strip().lower()

    # Backward compatibility: epa_openvins <case_dir> <align_mode>
    if not align_mode and len(raw_inputs) >= 2:
        candidate = str(raw_inputs[1]).strip().lower()
        if candidate in _KNOWN_ALIGN_MODES and not _is_case_dir(raw_inputs[1]):
            align_mode = candidate
            raw_inputs = [raw_inputs[0]] + raw_inputs[2:]
    if not align_mode:
        align_mode = "none"

    if align_mode not in _KNOWN_ALIGN_MODES:
        raise ValueError(f"Unsupported align_mode '{align_mode}'.")

    case_dirs = [str(Path(x).expanduser().resolve()) for x in raw_inputs]
    if not case_dirs:
        raise ValueError("No case directories were provided.")
    for d in case_dirs:
        if not _is_case_dir(d):
            raise ValueError(
                f"Invalid case dir: {d}. Expected stamped_groundtruth.txt and stamped_traj_estimate.txt"
            )

    mapped_eval_align = _map_align_mode_to_eval_align(align_mode)
    if align_mode in _LEGACY_ALIGN_MODES:
        messages = {
            "se3": (
                "align_mode 'se3' is a legacy OpenVINS alias for EPA Step3 "
                "(eval-align epa_step3). Use 'epa_se3' for EPA SE3 mode."
            ),
            "se3single": "align_mode 'se3single' is legacy/optional and maps to origin alignment.",
            "posyawsingle": "align_mode 'posyawsingle' is legacy/optional and maps to origin alignment.",
        }
        print(f"[info] {messages[align_mode]}")

    case_count = len(case_dirs)
    plot_enabled = not bool(args.no_plot)
    popup_enabled = bool(plot_enabled and case_count == 1)
    keep_output = bool(args.keep_output)

    save_root = str(args.save_root or "").strip()
    if case_count > 1:
        if not save_root:
            save_root = "/tmp/epa_batch"
        if not args.keep_output:
            keep_output = True

    save_root_path = Path(save_root).expanduser().resolve() if save_root else None
    if save_root_path is not None:
        save_root_path.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
    env["PYTHONPATH"] = str(repo_root / "src") + (
        (os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )

    print(
        f"[demo] running {case_count} case(s), align_mode={align_mode} -> eval_align={mapped_eval_align}"
    )

    for case_dir in case_dirs:
        gt = str(Path(case_dir) / "stamped_groundtruth.txt")
        est = str(Path(case_dir) / "stamped_traj_estimate.txt")
        case_label = _safe_case_label(case_dir)

        cmd = [
            sys.executable,
            "-m",
            "epa.cli",
            "--gt-csv",
            gt,
            "--gt-format",
            "tum",
            "--est-path",
            est,
            "--est-format",
            "tum",
            "--t-max-diff",
            "0.02",
            "--eval-align",
            mapped_eval_align,
        ]
        cmd.append("--plot" if plot_enabled else "--no-plot")

        print(f"[case] {case_label}")
        proc, elapsed_s = _run_timed(cmd, cwd=launch_cwd, env=env)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"EPA run failed for case {case_label} (exit={proc.returncode}).")

        run_dir = _parse_run_dir(proc.stdout, proc.stderr)
        if run_dir is None or not run_dir.exists():
            raise RuntimeError(f"Failed to resolve run dir for case {case_label}.")

        final_dir = run_dir
        if save_root_path is not None:
            dst = _unique_dst(save_root_path, case_label)
            shutil.move(str(run_dir), str(dst))
            final_dir = dst
            print(f"[case] saved to: {final_dir}")
        else:
            print(f"[case] output dir: {final_dir}")

        print(f"[case] elapsed_s: {elapsed_s:.3f}")
        if plot_enabled:
            print(f"[case] plots:   {final_dir / 'plots'}")
        print(f"[case] metrics: {final_dir / 'metrics.json'}")
        print(f"[case] summary: {final_dir / 'metrics_summary.csv'}")
        print(f"[case] report:  {final_dir / 'report_en.md'}")
        print(f"[case] report:  {final_dir / 'report_zh.md'}")

        if popup_enabled and plot_enabled:
            _show_plots_popup(final_dir / "plots")

        if not keep_output:
            shutil.rmtree(final_dir, ignore_errors=True)
            print(f"[case] cleaned output dir (default): {final_dir}")

    print("\n[done] EPA case runner finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
