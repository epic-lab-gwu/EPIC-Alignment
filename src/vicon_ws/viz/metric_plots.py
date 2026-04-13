from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vicon_ws")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vicon_ws.core.evaluation import RELATION_UNITS

_STATS_KEYS = ["rmse", "mean", "median", "std", "min", "max"]
_STAGES = ("raw", "step2", "step3")


def _safe_label(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _extract_relation(
    payload: dict,
    metric_kind: str,
    stage: str,
    relation: str,
) -> tuple[dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    pose_metrics = payload.get("pose_metrics", {})
    block = pose_metrics.get(metric_kind, {}).get(stage, {})
    stats = block.get(relation, {}) if isinstance(block, dict) else {}
    arrays = block.get("_error_arrays", {}) if isinstance(block, dict) else {}
    x_axis = block.get("_x_axis", {}) if isinstance(block, dict) else {}
    errors = np.asarray(arrays.get(relation, []), dtype=float).reshape(-1)
    x_axis_arrays = {
        k: np.asarray(v, dtype=float).reshape(-1)
        for k, v in x_axis.items()
        if isinstance(v, (list, tuple, np.ndarray))
    }
    return stats, errors, x_axis_arrays


def _pick_x(x_axis: dict[str, np.ndarray], errors: np.ndarray, x_dimension: str) -> tuple[np.ndarray, str]:
    if x_dimension == "seconds" and "seconds_from_start" in x_axis:
        x = x_axis["seconds_from_start"]
        label = "t (s)"
    elif x_dimension == "distances" and "distances_from_start" in x_axis:
        x = x_axis["distances_from_start"]
        label = "d (m)"
    elif "delta_ids" in x_axis:
        x = x_axis["delta_ids"]
        label = "delta_ids"
    else:
        x = x_axis.get("index", np.arange(errors.size, dtype=float))
        label = "index"
    n = min(x.size, errors.size)
    if n == 0:
        return np.array([], dtype=float), label
    return x[:n], label


def _plot_raw(
    traces: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
    ylabel: str,
    xlabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(11, 4.8))
    for label, xvals, errs in traces:
        n = min(xvals.size, errs.size)
        if n == 0:
            continue
        plt.plot(xvals[:n], errs[:n], label=label, linewidth=1.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_stats(
    stage_stats: list[tuple[str, dict[str, float]]],
    title: str,
    out_path: Path,
) -> None:
    labels = [name for name, _ in stage_stats]
    values = np.array(
        [[float(stats.get(k, np.nan)) for k in _STATS_KEYS] for _, stats in stage_stats],
        dtype=float,
    )
    x = np.arange(len(_STATS_KEYS), dtype=float)
    width = 0.24

    plt.figure(figsize=(11, 5.3))
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2.0) * width
        plt.bar(x + offset, values[idx], width=width, label=label)
    plt.xticks(x, _STATS_KEYS)
    plt.title(title)
    plt.ylabel("value")
    plt.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_hist(
    stage_errors: list[tuple[str, np.ndarray]],
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(11, 4.8))
    for label, errs in stage_errors:
        vals = np.asarray(errs, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        plt.hist(vals, bins=36, alpha=0.4, density=True, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_box(stage_errors: list[tuple[str, np.ndarray]], title: str, ylabel: str, out_path: Path) -> None:
    labels = []
    values = []
    for label, errs in stage_errors:
        vals = np.asarray(errs, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        labels.append(label)
        values.append(vals)
    if not values:
        return
    plt.figure(figsize=(9.8, 4.8))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_violin(stage_errors: list[tuple[str, np.ndarray]], title: str, ylabel: str, out_path: Path) -> None:
    labels = []
    values = []
    for label, errs in stage_errors:
        vals = np.asarray(errs, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        labels.append(label)
        values.append(vals)
    if not values:
        return
    plt.figure(figsize=(9.8, 4.8))
    plt.violinplot(values, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_metric_plots(
    metrics_payload: dict,
    out_dir: Path,
    ape_relation: str = "translation_part",
    rpe_relation: str = "translation_part",
    x_dimension: str = "seconds",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for metric_kind, relation in (("ape", ape_relation), ("rpe", rpe_relation)):
        slug = _safe_label(relation)
        unit = RELATION_UNITS.get(relation, "")
        ylabel = f"error ({unit})" if unit else "error"
        traces = []
        stage_stats = []
        stage_errors = []
        x_label = "index"
        for stage in _STAGES:
            stats, errors, x_axis = _extract_relation(metrics_payload, metric_kind, stage, relation)
            x_vals, x_label = _pick_x(x_axis, errors, x_dimension=x_dimension)
            traces.append((stage, x_vals, errors))
            stage_stats.append((stage, stats if isinstance(stats, dict) else {}))
            stage_errors.append((stage, errors))

        raw_path = out_dir / f"{metric_kind}_{slug}_raw.png"
        _plot_raw(
            traces,
            title=f"{metric_kind.upper()} raw values ({relation})",
            ylabel=ylabel,
            xlabel=x_label,
            out_path=raw_path,
        )
        produced.append(raw_path)

        stats_path = out_dir / f"{metric_kind}_{slug}_stats.png"
        _plot_stats(
            stage_stats,
            title=f"{metric_kind.upper()} stats ({relation})",
            out_path=stats_path,
        )
        produced.append(stats_path)

        hist_path = out_dir / f"{metric_kind}_{slug}_hist.png"
        _plot_hist(
            stage_errors,
            title=f"{metric_kind.upper()} distribution ({relation})",
            xlabel=ylabel,
            out_path=hist_path,
        )
        produced.append(hist_path)

        box_path = out_dir / f"{metric_kind}_{slug}_box.png"
        _plot_box(
            stage_errors,
            title=f"{metric_kind.upper()} box ({relation})",
            ylabel=ylabel,
            out_path=box_path,
        )
        if box_path.exists():
            produced.append(box_path)

        violin_path = out_dir / f"{metric_kind}_{slug}_violin.png"
        _plot_violin(
            stage_errors,
            title=f"{metric_kind.upper()} violin ({relation})",
            ylabel=ylabel,
            out_path=violin_path,
        )
        if violin_path.exists():
            produced.append(violin_path)

    md_path = out_dir / "plots.md"
    lines = [
        "# Metric Plots",
        "",
        f"- APE relation: `{ape_relation}`",
        f"- RPE relation: `{rpe_relation}`",
        f"- x dimension: `{x_dimension}`",
        "",
    ]
    for p in produced:
        lines.append(f"## {p.name}")
        lines.append("")
        lines.append(f"![{p.name}]({p.name})")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    produced.append(md_path)
    return produced


def aggregate_metric_results(
    payloads: list[dict],
    labels: list[str],
    out_dir: Path,
    metric_kind: str,
    relation: str,
    stage: str,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    series: list[tuple[str, np.ndarray]] = []
    for payload, label in zip(payloads, labels):
        stats, errors, _ = _extract_relation(payload, metric_kind, stage, relation)
        row = {"label": label}
        for key in _STATS_KEYS:
            row[key] = float(stats.get(key, np.nan)) if isinstance(stats, dict) else np.nan
        stats_rows.append(row)
        series.append((label, errors))

    csv_path = out_dir / "aggregated_stats.csv"
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", *_STATS_KEYS])
        writer.writeheader()
        writer.writerows(stats_rows)

    unit = RELATION_UNITS.get(relation, "")
    ylabel = f"error ({unit})" if unit else "error"

    rmse_path = out_dir / "aggregated_rmse.png"
    xs = np.arange(len(stats_rows), dtype=float)
    rmse_vals = [float(row["rmse"]) for row in stats_rows]
    plt.figure(figsize=(max(8.0, 0.7 * len(stats_rows)), 4.6))
    plt.bar(xs, rmse_vals)
    plt.xticks(xs, [row["label"] for row in stats_rows], rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{metric_kind.upper()} {relation} RMSE ({stage})")
    plt.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(rmse_path, dpi=180, bbox_inches="tight")
    plt.close()

    hist_path = out_dir / "aggregated_hist.png"
    _plot_hist(series, title=f"{metric_kind.upper()} {relation} distribution ({stage})", xlabel=ylabel, out_path=hist_path)

    box_path = out_dir / "aggregated_box.png"
    _plot_box(series, title=f"{metric_kind.upper()} {relation} box ({stage})", ylabel=ylabel, out_path=box_path)

    violin_path = out_dir / "aggregated_violin.png"
    _plot_violin(series, title=f"{metric_kind.upper()} {relation} violin ({stage})", ylabel=ylabel, out_path=violin_path)

    produced = [csv_path, rmse_path, hist_path]
    if box_path.exists():
        produced.append(box_path)
    if violin_path.exists():
        produced.append(violin_path)

    md_path = out_dir / "aggregate_plots.md"
    lines = [
        "# Aggregate Results",
        "",
        f"- metric: `{metric_kind}`",
        f"- relation: `{relation}`",
        f"- stage: `{stage}`",
        "",
        f"- stats table: `{csv_path.name}`",
        "",
    ]
    for p in produced[1:]:
        lines.append(f"## {p.name}")
        lines.append("")
        lines.append(f"![{p.name}]({p.name})")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    produced.append(md_path)
    return produced
