from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
import matplotlib
from epa.viz.backend_bootstrap import bootstrap_matplotlib_backend

bootstrap_matplotlib_backend(matplotlib)
import matplotlib.pyplot as plt
import numpy as np

from epa.config_cli import parse_args_with_config
from epa.core.evaluation import normalize_pose_relation
from epa.core.math_utils import compute_error_statistics
from epa.viz.plot_runtime import (
    configure_plot_runtime,
    should_enable_interactive_plot,
    show_plots,
)
from epa.viz.metric_plots import aggregate_metric_results

_STAT_KEYS = ("rmse", "mean", "median", "std", "min", "max", "sse")


def _default_out_dir(cwd: Path) -> Path:
    root = cwd / "outputs" / "res"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = root / stamp
    idx = 1
    while out.exists():
        out = root / f"{stamp}_{idx:02d}"
        idx += 1
    out.mkdir(parents=True, exist_ok=False)
    return out


def _load_payload(path: Path) -> dict:
    path = path.resolve()
    if path.is_dir():
        metrics_path = path / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found in directory: {path}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".zip":
        with ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "metrics.json" in names:
                payload = json.loads(zf.read("metrics.json").decode("utf-8"))
            elif "result.json" in names:
                result_obj = json.loads(zf.read("result.json").decode("utf-8"))
                payload = result_obj.get("metrics_payload", result_obj)
            elif "info.json" in names and "stats.json" in names:
                payload = _payload_from_evo_result_zip(zf)
            else:
                raise ValueError(f"No metrics.json or result.json in zip: {path}")
    else:
        raise ValueError(f"Unsupported result input: {path}")

    if "pose_metrics" not in payload:
        legacy = payload.get("evo_metrics")
        if isinstance(legacy, dict):
            payload["pose_metrics"] = legacy
    if "pose_metrics" not in payload:
        raise ValueError(f"Result payload has no pose_metrics: {path}")
    return payload


def _read_numpy_archive_1d(zf: ZipFile, name: str) -> np.ndarray:
    arr = np.load(BytesIO(zf.read(name)), allow_pickle=True)
    if hasattr(arr, "files"):
        files = list(getattr(arr, "files", []))
        if not files:
            return np.array([], dtype=float)
        arr = arr[files[0]]
    return np.asarray(arr, dtype=float).reshape(-1)


def _infer_metric_kind(info: dict) -> str:
    text = f"{info.get('title', '')} {info.get('label', '')}".lower()
    if "rpe" in text:
        return "rpe"
    return "ape"


def _infer_relation_key(info: dict) -> str:
    text = f"{info.get('title', '')} {info.get('label', '')}".lower()
    if "point distance error ratio" in text:
        return "point_distance_error_ratio"
    if "point distance" in text:
        return "point_distance"
    if "rotation part" in text:
        return "rotation_part"
    if "angle" in text:
        if "deg" in text:
            return "rotation_angle_deg"
        if "rad" in text:
            return "rotation_angle_rad"
    if "full transformation" in text:
        return "full_transformation"
    return "translation_part"


def _payload_from_evo_result_zip(zf: ZipFile) -> dict:
    info = json.loads(zf.read("info.json").decode("utf-8"))
    raw_stats = json.loads(zf.read("stats.json").decode("utf-8"))
    errors = (
        _read_numpy_archive_1d(zf, "error_array.npz")
        if "error_array.npz" in set(zf.namelist())
        else np.array([], dtype=float)
    )
    seconds = (
        _read_numpy_archive_1d(zf, "seconds_from_start.npz")
        if "seconds_from_start.npz" in set(zf.namelist())
        else np.array([], dtype=float)
    )

    computed_stats = compute_error_statistics(errors)
    stats: dict[str, float] = dict(computed_stats)
    if isinstance(raw_stats, dict):
        for k in _STAT_KEYS:
            v = raw_stats.get(k, None)
            if isinstance(v, (int, float)):
                stats[k] = float(v)

    metric_kind = _infer_metric_kind(info)
    relation = _infer_relation_key(info)
    stage_block: dict[str, object] = {
        relation: stats,
        "_error_arrays": {relation: errors.tolist()},
        "_x_axis": (
            {"seconds_from_start": seconds.tolist()}
            if int(seconds.size) > 0
            else {"index": np.arange(errors.size, dtype=float).tolist()}
        ),
    }
    if metric_kind == "rpe":
        stage_block["pair_count"] = int(errors.size)

    return {
        "info": info,
        "metadata": {"title": str(info.get("title", "") or "")},
        "pose_metrics": {metric_kind: {"step3": stage_block}},
    }


def _infer_label(path: Path) -> str:
    if path.is_dir():
        return path.name
    if path.suffix.lower() == ".json" and path.name == "metrics.json":
        return path.parent.name
    return path.stem


def _extract_relation_data(
    payload: dict,
    metric_kind: str,
    stage: str,
    relation: str,
) -> tuple[dict, np.ndarray, dict[str, np.ndarray], float]:
    pm = payload.get("pose_metrics", {})
    block = pm.get(metric_kind, {}).get(stage, {})
    if not isinstance(block, dict):
        return {}, np.array([], dtype=float), {}, float("nan")
    rel_block = block.get(relation, {})
    stats = rel_block if isinstance(rel_block, dict) else {}
    arrays = block.get("_error_arrays", {})
    x_axis = block.get("_x_axis", {})
    errors = np.asarray(arrays.get(relation, []), dtype=float).reshape(-1)
    x_axis_arrays = {
        k: np.asarray(v, dtype=float).reshape(-1)
        for k, v in x_axis.items()
        if isinstance(v, (list, tuple, np.ndarray))
    }
    pair_count = float(block.get("pair_count", np.nan)) if metric_kind == "rpe" else float("nan")
    return stats, errors, x_axis_arrays, pair_count


def _extract_value(
    payload: dict,
    metric_kind: str,
    stage: str,
    relation: str,
    stat: str,
) -> tuple[float, float]:
    stats, _, _, pair_count = _extract_relation_data(
        payload=payload,
        metric_kind=metric_kind,
        stage=stage,
        relation=relation,
    )
    value = float(stats.get(stat, np.nan)) if isinstance(stats, dict) else float("nan")
    return value, pair_count


def _pick_x(
    x_axis: dict[str, np.ndarray],
    errors: np.ndarray,
    use_rel_time: bool,
) -> tuple[np.ndarray, str]:
    if not use_rel_time and "timestamps" in x_axis:
        x = x_axis["timestamps"]
        label = "timestamp"
    elif "seconds_from_start" in x_axis:
        x = x_axis["seconds_from_start"]
        label = "t (s)"
    elif "distances_from_start" in x_axis:
        x = x_axis["distances_from_start"]
        label = "d (m)"
    elif "delta_ids" in x_axis:
        x = x_axis["delta_ids"]
        label = "delta_ids"
    else:
        x = np.arange(errors.size, dtype=float)
        label = "index"
    n = min(x.size, errors.size)
    if n == 0:
        return np.array([], dtype=float), label
    return x[:n], label


def _extract_title(payload: dict, metric_kind: str, stage: str, relation: str) -> str:
    info_title = payload.get("info", {}).get("title", "")
    if isinstance(info_title, str) and info_title.strip():
        return info_title.strip()
    meta_title = payload.get("metadata", {}).get("title", "")
    if isinstance(meta_title, str) and meta_title.strip():
        return meta_title.strip()
    return f"{metric_kind.upper()} {relation} ({stage})"


def _plot_raw_series(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
    out_path: Path,
    plot_markers: bool,
    keep_open: bool = False,
) -> Path:
    plt.figure(figsize=(11.0, 4.8))
    xlabel = "index"
    for label, xvals, errs in series:
        n = min(xvals.size, errs.size)
        if n == 0:
            continue
        if n == xvals.size and n > 0:
            if np.all(np.diff(xvals[:n]) >= 0):
                if np.max(xvals[:n]) - np.min(xvals[:n]) > 0:
                    xlabel = "t (s)" if np.max(xvals[:n]) > 1.0 else xlabel
        if plot_markers:
            plt.plot(
                xvals[:n],
                errs[:n],
                "-o",
                markersize=2.5,
                linewidth=1.1,
                label=label,
            )
        else:
            plt.plot(xvals[:n], errs[:n], linewidth=1.4, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("error")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    if not keep_open:
        plt.close()
    return out_path


def _export_plot_artifacts(produced: list[Path], save_plot: str) -> list[Path]:
    base = Path(str(save_plot)).expanduser().resolve()
    exported: list[Path] = []
    plots = [p for p in produced if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} and p.exists()]
    if not plots:
        return exported
    if base.suffix:
        base.parent.mkdir(parents=True, exist_ok=True)
        for src in plots:
            dst = base.with_name(f"{base.stem}_{src.stem}{base.suffix}")
            shutil.copy2(src, dst)
            exported.append(dst)
    else:
        base.mkdir(parents=True, exist_ok=True)
        for src in plots:
            dst = base / src.name
            shutil.copy2(src, dst)
            exported.append(dst)
    return exported


def _build_merged_payload(
    payloads: list[dict],
    metric_kind: str,
    stage: str,
    relation: str,
    stat: str,
    use_rel_time: bool,
) -> tuple[dict, float, float]:
    all_errors: list[np.ndarray] = []
    all_stats_vals: list[float] = []
    total_pairs = 0.0
    for payload in payloads:
        stats, errors, _, pair_count = _extract_relation_data(
            payload=payload,
            metric_kind=metric_kind,
            stage=stage,
            relation=relation,
        )
        finite_err = np.asarray(errors, dtype=float)
        finite_err = finite_err[np.isfinite(finite_err)]
        if finite_err.size > 0:
            all_errors.append(finite_err)
        val = float(stats.get(stat, np.nan)) if isinstance(stats, dict) else np.nan
        if np.isfinite(val):
            all_stats_vals.append(val)
        if metric_kind == "rpe" and np.isfinite(pair_count):
            total_pairs += float(pair_count)

    if all_errors:
        merged_errors = np.concatenate(all_errors)
    else:
        merged_errors = np.asarray(all_stats_vals, dtype=float).reshape(-1)
    merged_stats = compute_error_statistics(merged_errors)
    x_axis: dict[str, np.ndarray] = {"index": np.arange(merged_errors.size, dtype=float)}
    if use_rel_time:
        x_axis["seconds_from_start"] = np.arange(merged_errors.size, dtype=float)

    stage_block: dict[str, object] = {
        relation: merged_stats,
        "_error_arrays": {relation: merged_errors},
        "_x_axis": x_axis,
    }
    if metric_kind == "rpe":
        stage_block["pair_count"] = int(total_pairs) if np.isfinite(total_pairs) else 0

    merged_payload = {
        "pose_metrics": {
            metric_kind: {
                stage: stage_block,
            }
        },
        "metadata": {"title": f"Merged {metric_kind.upper()} {relation} ({stage})"},
    }
    merged_value = float(merged_stats.get(stat, np.nan))
    merged_pairs = total_pairs if metric_kind == "rpe" else float("nan")
    return merged_payload, merged_value, merged_pairs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare one or multiple result bundles / metrics files."
    )
    p.add_argument(
        "-c",
        "--config",
        default="",
        help="Path to JSON config file. If set, config values override CLI flags.",
    )
    p.add_argument(
        "results",
        nargs="+",
        help="Result inputs: .zip bundle, metrics.json, or run directory containing metrics.json.",
    )
    p.add_argument("--labels", nargs="*", default=[], help="Optional labels, same count as results.")
    p.add_argument("--metric", choices=["all", "ape", "rpe"], default="all", help="Metric family to compare.")
    p.add_argument("--ape-relation", default="trans_part", help="APE relation (alias or internal name).")
    p.add_argument("--rpe-relation", default="trans_part", help="RPE relation (alias or internal name).")
    p.add_argument("--stage", choices=["raw", "step2", "step3"], default="step3", help="Stage to compare.")
    p.add_argument("--stat", choices=list(_STAT_KEYS), default="rmse", help="Statistic to compare.")
    p.add_argument("--merge", action="store_true", help="Merge inputs into one aggregated result.")
    p.add_argument("--use-rel-time", "--use_rel_time", action="store_true", help="Use relative time axis when plotting raw values.")
    p.add_argument("--use-filenames", "--use_filenames", action="store_true", help="Use source filenames as labels.")
    p.add_argument("--ignore-title", "--ignore_title", action="store_true", help="Ignore title mismatch checks.")
    p.add_argument("--plot-markers", "--plot_markers", action="store_true", help="Use markers in raw plots.")
    p.add_argument(
        "--plot-interactive",
        "--plot_interactive",
        action="store_true",
        help="Show interactive matplotlib window after generating plots.",
    )
    p.add_argument(
        "--plot-backend",
        "--plot_backend",
        default="",
        help="Optional matplotlib backend override (e.g. qtagg, tkagg).",
    )
    p.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generate aggregate plots (in TTY sessions also opens interactive window).",
    )
    p.add_argument("--out-dir", default="", help="Output directory for generated files.")
    p.add_argument("--save-plot", "--save_plot", default="", help="Path prefix (file or dir) to export plots.")
    p.add_argument("--save-table", "--save_table", default="", help="Path to save comparison table as CSV.")
    p.add_argument("--logfile", default=None, help="Reserved for compatibility.")
    p.add_argument("--no_warnings", action="store_true", help="Skip interactive warning gates.")
    p.add_argument("-v", "--verbose", action="store_true", help="Reserved for compatibility.")
    p.add_argument("--silent", action="store_true", help="Reserved for compatibility.")
    p.add_argument("--debug", action="store_true", help="Reserved for compatibility.")
    return p


def run(args: argparse.Namespace) -> int:
    paths = [Path(x).expanduser().resolve() for x in args.results]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Result input not found: {p}")

    if args.labels:
        if len(args.labels) != len(paths):
            raise ValueError("--labels count must match number of results.")
        labels = [str(x) for x in args.labels]
    elif bool(args.use_filenames):
        labels = [str(p) for p in paths]
    else:
        labels = [_infer_label(p) for p in paths]
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    if duplicates:
        hint = " Try --use-filenames." if not bool(args.use_filenames) else ""
        raise ValueError(f"Duplicate labels are not allowed: {', '.join(duplicates)}.{hint}")

    payloads = [_load_payload(p) for p in paths]
    ape_rel = normalize_pose_relation("ape", args.ape_relation)
    rpe_rel = normalize_pose_relation("rpe", args.rpe_relation)
    metric_kinds = ["ape", "rpe"] if args.metric == "all" else [args.metric]

    rows: list[dict[str, object]] = []
    merged_payloads: dict[str, dict] = {}
    if bool(args.merge):
        for metric_kind in metric_kinds:
            relation = ape_rel if metric_kind == "ape" else rpe_rel
            merged_payload, value, pair_count = _build_merged_payload(
                payloads=payloads,
                metric_kind=metric_kind,
                stage=args.stage,
                relation=relation,
                stat=args.stat,
                use_rel_time=bool(args.use_rel_time),
            )
            merged_payloads[metric_kind] = merged_payload
            rows.append(
                {
                    "label": "merged",
                    "source": ";".join(str(p) for p in paths),
                    "metric": metric_kind,
                    "relation": relation,
                    "stage": args.stage,
                    "stat": args.stat,
                    "value": value,
                    "pair_count": pair_count,
                }
            )
    else:
        for label, path, payload in zip(labels, paths, payloads):
            for metric_kind in metric_kinds:
                relation = ape_rel if metric_kind == "ape" else rpe_rel
                value, pair_count = _extract_value(
                    payload,
                    metric_kind=metric_kind,
                    stage=args.stage,
                    relation=relation,
                    stat=args.stat,
                )
                rows.append(
                    {
                        "label": label,
                        "source": str(path),
                        "metric": metric_kind,
                        "relation": relation,
                        "stage": args.stage,
                        "stat": args.stat,
                        "value": value,
                        "pair_count": pair_count,
                    }
                )

    print("label,metric,relation,stage,stat,value,pair_count")
    for row in rows:
        v = row["value"]
        pc = row["pair_count"]
        v_txt = "nan" if not np.isfinite(v) else f"{float(v):.6f}"
        pc_txt = "nan" if not np.isfinite(pc) else f"{int(pc):d}"
        print(
            f"{row['label']},{row['metric']},{row['relation']},{row['stage']},"
            f"{row['stat']},{v_txt},{pc_txt}"
        )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if bool(args.plot or str(getattr(args, "save_plot", "")).strip()) and out_dir is None:
        out_dir = _default_out_dir(Path.cwd())
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    table_path = Path(args.save_table).expanduser().resolve() if args.save_table else (
        (out_dir / "summary.csv") if out_dir is not None else None
    )
    if table_path is not None:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with table_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["label", "source", "metric", "relation", "stage", "stat", "value", "pair_count"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved table: {table_path}")

    produced: list[Path] = []
    interactive_enabled = False
    if bool(args.plot or str(getattr(args, "save_plot", "")).strip()):
        assert out_dir is not None
        interactive_requested = should_enable_interactive_plot(
            plot=bool(getattr(args, "plot", False)),
            plot_interactive=bool(getattr(args, "plot_interactive", False)),
        )
        interactive_enabled, active_backend = configure_plot_runtime(
            plot_interactive=interactive_requested,
            plot_backend=str(getattr(args, "plot_backend", "") or ""),
        )
        if interactive_requested:
            if interactive_enabled:
                print(f"[plot] Interactive backend: {active_backend}")
            else:
                print("[plot] Interactive mode requested but no interactive backend is available.")
        for metric_kind in metric_kinds:
            relation = ape_rel if metric_kind == "ape" else rpe_rel
            metric_out = out_dir / f"{metric_kind}_{relation}_{args.stage}"
            metric_payloads = [merged_payloads[metric_kind]] if bool(args.merge) else payloads
            metric_labels = ["merged"] if bool(args.merge) else labels

            titles = [
                _extract_title(p, metric_kind=metric_kind, stage=args.stage, relation=relation)
                for p in metric_payloads
            ]
            if not bool(args.ignore_title) and len(set(titles)) > 1:
                if bool(getattr(args, "no_warnings", False)):
                    print(
                        "Warning: mismatching titles detected across result files. "
                        "Proceeding due to --no_warnings."
                    )
                else:
                    raise ValueError(
                        "Mismatching titles detected across result files. "
                        "Use --ignore-title (or --no_warnings) to aggregate anyway."
                    )

            produced.extend(
                aggregate_metric_results(
                    payloads=metric_payloads,
                    labels=metric_labels,
                    out_dir=metric_out,
                    metric_kind=metric_kind,
                    relation=relation,
                    stage=args.stage,
                    keep_open=interactive_enabled,
                )
            )

            series: list[tuple[str, np.ndarray, np.ndarray]] = []
            for payload, label in zip(metric_payloads, metric_labels):
                _, errors, x_axis, _ = _extract_relation_data(
                    payload=payload,
                    metric_kind=metric_kind,
                    stage=args.stage,
                    relation=relation,
                )
                x_vals, _ = _pick_x(x_axis, errors, use_rel_time=bool(args.use_rel_time))
                series.append((label, x_vals, errors))
            raw_path = metric_out / "aggregated_raw.png"
            produced.append(
                _plot_raw_series(
                    series=series,
                    title=f"{metric_kind.upper()} raw values ({relation}, {args.stage})",
                    out_path=raw_path,
                    plot_markers=bool(args.plot_markers),
                    keep_open=interactive_enabled,
                )
            )
        print(f"Generated {len(produced)} plot/artifact file(s) in: {out_dir}")
        if interactive_enabled:
            show_plots()
            plt.close("all")

    if str(getattr(args, "save_plot", "")).strip():
        exported = _export_plot_artifacts(produced, str(args.save_plot))
        if exported:
            print(f"Exported plot files: {', '.join(str(p) for p in exported)}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parse_args_with_config(parser, config_dest="config", tool_name="epa_res")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
