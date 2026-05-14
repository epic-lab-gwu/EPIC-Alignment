from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
import matplotlib

from epa.viz.backend_bootstrap import bootstrap_matplotlib_backend

bootstrap_matplotlib_backend(matplotlib)
import matplotlib.pyplot as plt
import numpy as np

from epa.core.evaluation import RELATION_UNITS

_STATS_KEYS = ["rmse", "mean", "median", "std", "min", "max"]
_DEFAULT_STAGES = ("raw", "step2", "step3")


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


def _discover_stages(payload: dict) -> list[str]:
    pose_metrics = payload.get("pose_metrics", {}) if isinstance(payload, dict) else {}
    ape_block = pose_metrics.get("ape", {}) if isinstance(pose_metrics, dict) else {}
    rpe_block = pose_metrics.get("rpe", {}) if isinstance(pose_metrics, dict) else {}
    seen = set()
    ordered = []
    for stage in _DEFAULT_STAGES:
        if (isinstance(ape_block, dict) and stage in ape_block) or (isinstance(rpe_block, dict) and stage in rpe_block):
            ordered.append(stage)
            seen.add(stage)
    for blk in (ape_block, rpe_block):
        if not isinstance(blk, dict):
            continue
        for stage in blk.keys():
            if stage.startswith("_"):
                continue
            if stage not in seen:
                ordered.append(stage)
                seen.add(stage)
    return ordered if ordered else list(_DEFAULT_STAGES[:3])


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


def _fail_spans_from_success(success: dict, xlabel: str) -> list[tuple[float, float]]:
    if not isinstance(success, dict):
        return []
    fail_segments = success.get("fail_segments", [])
    if not isinstance(fail_segments, list):
        return []
    if str(xlabel).strip() == "d (m)":
        start_key, end_key = "start_distance_m", "end_distance_m"
    elif str(xlabel).strip() == "t (s)":
        start_key, end_key = "start_time_s", "end_time_s"
    else:
        start_key, end_key = "start_index", "end_index"

    spans = []
    for segment in fail_segments:
        if not isinstance(segment, dict):
            continue
        start = float(segment.get(start_key, np.nan))
        end = float(segment.get(end_key, np.nan))
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        lo, hi = sorted((start, end))
        spans.append((lo, hi))
    return spans


def _step3_fail_spans(metrics_payload: dict, xlabel: str) -> list[tuple[float, float]]:
    pose_metrics = metrics_payload.get("pose_metrics", {}) if isinstance(metrics_payload, dict) else {}
    valid = pose_metrics.get("valid_segment", {}) if isinstance(pose_metrics, dict) else {}
    step3 = valid.get("step3", {}) if isinstance(valid, dict) else {}
    success = step3.get("success", {}) if isinstance(step3, dict) else {}
    return _fail_spans_from_success(success, xlabel)


def _pose_relation_title_label(relation: str) -> str:
    mapping = {
        "translation_part": "translation part",
        "rotation_part": "rotation part",
        "rotation_angle_deg": "rotation angle (deg)",
        "rotation_angle_rad": "rotation angle (rad)",
        "point_distance": "point distance",
        "full_transformation": "full transformation",
    }
    return mapping.get(str(relation), str(relation).replace("_", " "))


def _stage_caption(stage: str) -> str:
    stage_name = str(stage).lower()
    if stage_name == "step3":
        return "EPA Step-3 world alignment (SE(3) SVD)"
    if stage_name == "step2":
        return "after extrinsic calibration"
    return "without alignment"


def _plot_raw_with_stats(
    *,
    x_vals: np.ndarray,
    errors: np.ndarray,
    stats: dict[str, float],
    title: str,
    ylabel: str,
    xlabel: str,
    line_label: str,
    out_path: Path,
    fail_spans: list[tuple[float, float]] | None = None,
    keep_open: bool = False,
) -> None:
    n = min(x_vals.size, errors.size)
    if n == 0:
        return

    x = np.asarray(x_vals[:n], dtype=float)
    y = np.asarray(errors[:n], dtype=float)
    fig = plt.figure(figsize=(11, 6.0))
    ax = fig.add_subplot(111)
    if fail_spans:
        for idx, (start, end) in enumerate(fail_spans):
            ax.axvspan(
                start,
                end,
                color="#e74c3c",
                alpha=0.14,
                linewidth=0.0,
                label="fail segment" if idx == 0 else None,
            )
    ax.plot(x, y, linewidth=1.4, color="gray", label=line_label)

    mean_v = float(stats.get("mean", np.nan))
    std_v = float(stats.get("std", np.nan))
    rmse_v = float(stats.get("rmse", np.nan))
    median_v = float(stats.get("median", np.nan))

    finite_x = x[np.isfinite(x)]
    if finite_x.size > 0 and np.isfinite(mean_v) and np.isfinite(std_v) and std_v >= 0.0:
        x0 = float(np.min(finite_x))
        x1 = float(np.max(finite_x))
        ax.fill_between(
            [x0, x1],
            [mean_v - std_v, mean_v - std_v],
            [mean_v + std_v, mean_v + std_v],
            color="#7f6db0",
            alpha=0.35,
            label="std",
        )
    if np.isfinite(rmse_v):
        ax.axhline(rmse_v, color="#3b6db1", linewidth=1.5, label="rmse")
    if np.isfinite(median_v):
        ax.axhline(median_v, color="#4ca45a", linewidth=1.5, label="median")
    if np.isfinite(mean_v):
        ax.axhline(mean_v, color="#c44747", linewidth=1.5, label="mean")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    for name in ("fail segment", line_label, "rmse", "median", "mean", "std"):
        if name in labels:
            idx = labels.index(name)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    if ordered_handles:
        ax.legend(ordered_handles, ordered_labels, loc="best")
    else:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def _plot_raw(
    traces: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
    ylabel: str,
    xlabel: str,
    out_path: Path,
    fail_spans: list[tuple[float, float]] | None = None,
    keep_open: bool = False,
) -> None:
    fig = plt.figure(figsize=(11, 4.8))
    ax = fig.add_subplot(111)
    if fail_spans:
        for idx, (start, end) in enumerate(fail_spans):
            ax.axvspan(
                start,
                end,
                color="#e74c3c",
                alpha=0.14,
                linewidth=0.0,
                label="fail segment" if idx == 0 else None,
            )
    for label, xvals, errs in traces:
        n = min(xvals.size, errs.size)
        if n == 0:
            continue
        ax.plot(xvals[:n], errs[:n], label=label, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if not keep_open:
        plt.close()


def _plot_stats(
    stage_stats: list[tuple[str, dict[str, float]]],
    title: str,
    out_path: Path,
    keep_open: bool = False,
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
    if not keep_open:
        plt.close()


def _plot_hist(
    stage_errors: list[tuple[str, np.ndarray]],
    title: str,
    xlabel: str,
    out_path: Path,
    keep_open: bool = False,
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
    if not keep_open:
        plt.close()


def _plot_box(
    stage_errors: list[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
    out_path: Path,
    keep_open: bool = False,
) -> None:
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
    if not keep_open:
        plt.close()


def _plot_violin(
    stage_errors: list[tuple[str, np.ndarray]],
    title: str,
    ylabel: str,
    out_path: Path,
    keep_open: bool = False,
) -> None:
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
    if not keep_open:
        plt.close()


def generate_metric_plots(
    metrics_payload: dict,
    out_dir: Path,
    ape_relation: str = "translation_part",
    rpe_relation: str = "translation_part",
    x_dimension: str = "seconds",
    keep_open: bool = False,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    stages = _discover_stages(metrics_payload)
    for metric_kind, relation in (("ape", ape_relation), ("rpe", rpe_relation)):
        slug = _safe_label(relation)
        unit = RELATION_UNITS.get(relation, "")
        ylabel = f"error ({unit})" if unit else "error"
        traces = []
        stage_stats = []
        stage_errors = []
        x_label = "index"
        for stage in stages:
            stats, errors, x_axis = _extract_relation(metrics_payload, metric_kind, stage, relation)
            x_vals, x_label = _pick_x(x_axis, errors, x_dimension=x_dimension)
            traces.append((stage, x_vals, errors))
            stage_stats.append((stage, stats if isinstance(stats, dict) else {}))
            stage_errors.append((stage, errors))

        raw_path = out_dir / f"{metric_kind}_{slug}_raw.png"
        fail_spans = _step3_fail_spans(metrics_payload, x_label) if metric_kind == "ape" else None
        _plot_raw(
            traces,
            title=f"{metric_kind.upper()} raw values ({relation})",
            ylabel=ylabel,
            xlabel=x_label,
            out_path=raw_path,
            fail_spans=fail_spans,
            keep_open=keep_open,
        )
        produced.append(raw_path)

        stats_path = out_dir / f"{metric_kind}_{slug}_stats.png"
        _plot_stats(
            stage_stats,
            title=f"{metric_kind.upper()} stats ({relation})",
            out_path=stats_path,
            keep_open=keep_open,
        )
        produced.append(stats_path)

        hist_path = out_dir / f"{metric_kind}_{slug}_hist.png"
        _plot_hist(
            stage_errors,
            title=f"{metric_kind.upper()} distribution ({relation})",
            xlabel=ylabel,
            out_path=hist_path,
            keep_open=keep_open,
        )
        produced.append(hist_path)

        box_path = out_dir / f"{metric_kind}_{slug}_box.png"
        _plot_box(
            stage_errors,
            title=f"{metric_kind.upper()} box ({relation})",
            ylabel=ylabel,
            out_path=box_path,
            keep_open=keep_open,
        )
        if box_path.exists():
            produced.append(box_path)

        violin_path = out_dir / f"{metric_kind}_{slug}_violin.png"
        _plot_violin(
            stage_errors,
            title=f"{metric_kind.upper()} violin ({relation})",
            ylabel=ylabel,
            out_path=violin_path,
            keep_open=keep_open,
        )
        if violin_path.exists():
            produced.append(violin_path)

    return produced


def generate_time_rpe_metric_plots(
    metrics_payload: dict,
    out_dir: Path,
    relation: str = "translation_part",
    x_dimension: str = "seconds",
    keep_open: bool = False,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    stages = _discover_stages(metrics_payload)
    slug = _safe_label(relation)
    unit = RELATION_UNITS.get(relation, "")
    ylabel = f"1s drift ({unit})" if unit else "1s drift"
    traces = []
    stage_stats = []
    stage_errors = []
    x_label = "index"
    for stage in stages:
        stats, errors, x_axis = _extract_relation(metrics_payload, "rpe_time_1s", stage, relation)
        x_vals, x_label = _pick_x(x_axis, errors, x_dimension=x_dimension)
        traces.append((stage, x_vals, errors))
        stage_stats.append((stage, stats if isinstance(stats, dict) else {}))
        stage_errors.append((stage, errors))

    raw_path = out_dir / f"rpe_time_1s_{slug}_raw.png"
    _plot_raw(
        traces,
        title=f"1-second time RPE raw values ({relation})",
        ylabel=ylabel,
        xlabel=x_label,
        out_path=raw_path,
        fail_spans=_step3_fail_spans(metrics_payload, x_label),
        keep_open=keep_open,
    )
    produced.append(raw_path)

    stats_path = out_dir / f"rpe_time_1s_{slug}_stats.png"
    _plot_stats(
        stage_stats,
        title=f"1-second time RPE stats ({relation})",
        out_path=stats_path,
        keep_open=keep_open,
    )
    produced.append(stats_path)

    box_path = out_dir / f"rpe_time_1s_{slug}_box.png"
    _plot_box(
        stage_errors,
        title=f"1-second time RPE box ({relation})",
        ylabel=ylabel,
        out_path=box_path,
        keep_open=keep_open,
    )
    if box_path.exists():
        produced.append(box_path)

    return produced


def generate_ape_stage_raw_plot(
    metrics_payload: dict,
    out_dir: Path,
    ape_relation: str = "translation_part",
    stage: str = "step3",
    x_dimension: str = "seconds",
    keep_open: bool = False,
    file_name: str = "",
) -> Path | None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, errors, x_axis = _extract_relation(metrics_payload, "ape", stage, ape_relation)
    if errors.size == 0:
        return None
    x_vals, x_label = _pick_x(x_axis, errors, x_dimension=x_dimension)
    if x_vals.size == 0:
        return None

    unit = RELATION_UNITS.get(ape_relation, "")
    ylabel = f"APE ({unit})" if unit else "APE"
    relation_title = _pose_relation_title_label(ape_relation)
    relation_with_unit = f"{relation_title} ({unit})" if unit else relation_title
    title = f"APE w.r.t. {relation_with_unit}\n({_stage_caption(stage)})"
    line_label = f"APE ({unit})" if unit else "APE"

    if str(file_name).strip():
        out_path = out_dir / str(file_name).strip()
    else:
        out_path = out_dir / f"ape_{_safe_label(ape_relation)}_{_safe_label(stage)}_raw.png"

    _plot_raw_with_stats(
        x_vals=x_vals,
        errors=errors,
        stats=stats if isinstance(stats, dict) else {},
        title=title,
        ylabel=ylabel,
        xlabel=x_label,
        line_label=line_label,
        out_path=out_path,
        fail_spans=_step3_fail_spans(metrics_payload, x_label),
        keep_open=keep_open,
    )
    return out_path if out_path.exists() else None


def aggregate_metric_results(
    payloads: list[dict],
    labels: list[str],
    out_dir: Path,
    metric_kind: str,
    relation: str,
    stage: str,
    keep_open: bool = False,
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
    if not keep_open:
        plt.close()

    hist_path = out_dir / "aggregated_hist.png"
    _plot_hist(
        series,
        title=f"{metric_kind.upper()} {relation} distribution ({stage})",
        xlabel=ylabel,
        out_path=hist_path,
        keep_open=keep_open,
    )

    box_path = out_dir / "aggregated_box.png"
    _plot_box(
        series,
        title=f"{metric_kind.upper()} {relation} box ({stage})",
        ylabel=ylabel,
        out_path=box_path,
        keep_open=keep_open,
    )

    violin_path = out_dir / "aggregated_violin.png"
    _plot_violin(
        series,
        title=f"{metric_kind.upper()} {relation} violin ({stage})",
        ylabel=ylabel,
        out_path=violin_path,
        keep_open=keep_open,
    )

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
