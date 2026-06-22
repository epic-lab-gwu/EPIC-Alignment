from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from epa.core.evaluation import RELATION_UNITS

DEFAULT_MAX_POINTS = 10000


def _finite_or_none(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _to_float_list(values: Any, *, max_points: int = DEFAULT_MAX_POINTS) -> list[float | None]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return []
    if arr.size > max_points:
        idx = np.linspace(0, arr.size - 1, max_points).astype(int)
        arr = arr[idx]
    return [_finite_or_none(v) for v in arr]


def _xyz_trace(pos: Any, *, max_points: int = DEFAULT_MAX_POINTS) -> dict[str, list[float]]:
    pts = np.asarray(pos, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
        return {"x": [], "y": [], "z": []}
    pts = pts[:, :3]
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts = pts[idx]
    return {
        "x": pts[:, 0].astype(float).tolist(),
        "y": pts[:, 1].astype(float).tolist(),
        "z": pts[:, 2].astype(float).tolist(),
    }


def _error_xyz(pos_ref: Any, pos_est: Any, *, max_points: int = DEFAULT_MAX_POINTS) -> list[float]:
    ref = np.asarray(pos_ref, dtype=float)
    est = np.asarray(pos_est, dtype=float)
    if ref.shape != est.shape or ref.ndim != 2 or ref.shape[1] < 3:
        return []
    err = np.linalg.norm(est[:, :3] - ref[:, :3], axis=1)
    return _to_float_list(err, max_points=max_points)


def _speed_trace(timestamps: Any, pos: Any, *, label: str, max_points: int = DEFAULT_MAX_POINTS) -> dict[str, Any]:
    pts = np.asarray(pos, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] < 2:
        return {"label": label, "x": [], "y": [], "xLabel": "index"}
    pts = pts[:, :3]
    t = np.asarray(timestamps, dtype=float).reshape(-1)
    if t.size != pts.shape[0]:
        t = np.arange(pts.shape[0], dtype=float)
        x_label = "index"
    else:
        t = t - t[0] if t.size else t
        x_label = "time (s)"
    dt = np.diff(t)
    dist = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    valid = np.isfinite(dt) & np.isfinite(dist) & (dt > 0.0)
    if not np.any(valid):
        return {"label": label, "x": [], "y": [], "xLabel": x_label}
    x_vals = t[1:][valid]
    speed = dist[valid] / dt[valid]
    return {
        "label": label,
        "x": _to_float_list(x_vals, max_points=max_points),
        "y": _to_float_list(speed, max_points=max_points),
        "xLabel": x_label,
    }


def _speed_plot_data(
    timestamps: Any,
    *,
    pos_gt: Any,
    pr_sync: Any,
    pr_corrected: Any,
    pr_final: Any,
    debug: bool,
    max_points: int,
) -> dict[str, Any]:
    traces = [
        _speed_trace(timestamps, pos_gt, label="ground truth", max_points=max_points),
        _speed_trace(timestamps, pr_final, label="step3", max_points=max_points),
    ]
    if debug:
        traces.extend(
            [
                _speed_trace(timestamps, pr_sync, label="raw", max_points=max_points),
                _speed_trace(timestamps, pr_corrected, label="step2", max_points=max_points),
            ]
        )
    x_label = next((trace["xLabel"] for trace in traces if trace.get("x")), "index")
    return {
        "title": "Linear velocity",
        "xLabel": x_label,
        "yLabel": "speed (m/s)",
        "traces": traces,
    }


def _pick_x_axis(x_axis: dict[str, Any], errors: np.ndarray, x_dimension: str) -> tuple[np.ndarray, str]:
    if x_dimension == "seconds" and "seconds_from_start" in x_axis:
        return np.asarray(x_axis["seconds_from_start"], dtype=float).reshape(-1), "seconds"
    if x_dimension == "distances" and "distances_from_start" in x_axis:
        return np.asarray(x_axis["distances_from_start"], dtype=float).reshape(-1), "distance (m)"
    if "index" in x_axis:
        return np.asarray(x_axis["index"], dtype=float).reshape(-1), "index"
    return np.arange(errors.size, dtype=float), "index"


def _metric_plot_data(
    metrics_payload: dict,
    *,
    metric_kind: str,
    relation: str,
    x_dimension: str,
    max_points: int,
    stages: tuple[str, ...] = ("step3",),
) -> dict[str, Any]:
    pose_metrics = metrics_payload.get("pose_metrics", {}) if isinstance(metrics_payload, dict) else {}
    metric_block = pose_metrics.get(metric_kind, {}) if isinstance(pose_metrics, dict) else {}
    traces = []
    boxes = []
    stats_rows = []
    x_label = "index"
    for stage in stages:
        stage_block = metric_block.get(stage, {}) if isinstance(metric_block, dict) else {}
        arrays = stage_block.get("_error_arrays", {}) if isinstance(stage_block, dict) else {}
        errors = np.asarray(arrays.get(relation, []), dtype=float).reshape(-1)
        if errors.size == 0:
            continue
        x_axis = stage_block.get("_x_axis", {}) if isinstance(stage_block, dict) else {}
        x_vals, x_label = _pick_x_axis(x_axis, errors, x_dimension)
        n = min(x_vals.size, errors.size)
        x_vals = x_vals[:n]
        errors = errors[:n]
        traces.append(
            {
                "stage": stage,
                "x": _to_float_list(x_vals, max_points=max_points),
                "y": _to_float_list(errors, max_points=max_points),
            }
        )
        boxes.append({"stage": stage, "y": _to_float_list(errors, max_points=max_points)})
        stats = stage_block.get(relation, {}) if isinstance(stage_block, dict) else {}
        if isinstance(stats, dict):
            stats_rows.append(
                {
                    "stage": stage,
                    "rmse": _finite_or_none(stats.get("rmse")),
                    "mean": _finite_or_none(stats.get("mean")),
                    "median": _finite_or_none(stats.get("median")),
                    "p95": _finite_or_none(stats.get("p95")),
                }
            )
    unit = RELATION_UNITS.get(relation, "")
    return {
        "title": f"{metric_kind.upper()} {relation}",
        "xLabel": x_label,
        "yLabel": f"error ({unit})" if unit else "error",
        "traces": traces,
        "boxes": boxes,
        "stats": stats_rows,
    }


def _scalar_text(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, str):
        return value
    finite = _finite_or_none(value)
    if finite is None:
        return "N/A"
    if abs(finite) >= 1000.0 or (0.0 < abs(finite) < 0.001):
        return f"{finite:.3e}"
    return f"{finite:.6g}"


def _lookup(payload: dict, path: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _summary_card(label: str, value: Any, unit: str = "", *, percent: bool = False) -> dict[str, str]:
    finite = _finite_or_none(value)
    if percent and finite is not None:
        text = f"{finite * 100.0:.2f}%"
    else:
        text = _scalar_text(value)
        if unit and text != "N/A":
            text = f"{text} {unit}"
    return {"label": label, "value": text}


def _metric_summary_cards(metrics_payload: dict, *, ape_relation: str, rpe_relation: str) -> list[dict[str, str]]:
    return [
        _summary_card("diagnosis", _lookup(metrics_payload, ("case_diagnostics", "diagnosis_primary"))),
        _summary_card("diagnosis tags", _lookup(metrics_payload, ("case_diagnostics", "diagnosis_summary"))),
        _summary_card(
            "case status",
            _lookup(metrics_payload, ("pose_metrics", "valid_segment", "step3", "success", "case_status")),
        ),
        _summary_card(
            "SR distance",
            _lookup(metrics_payload, ("pose_metrics", "valid_segment", "step3", "success", "success_rate_distance")),
            percent=True,
        ),
        _summary_card(
            "SR time",
            _lookup(metrics_payload, ("pose_metrics", "valid_segment", "step3", "success", "success_rate_time")),
            percent=True,
        ),
        _summary_card(
            "APE step3 RMSE",
            _lookup(metrics_payload, ("pose_metrics", "ape", "step3", ape_relation, "rmse")),
            "m",
        ),
        _summary_card(
            "RPE step3 RMSE",
            _lookup(metrics_payload, ("pose_metrics", "rpe", "step3", rpe_relation, "rmse")),
            "m",
        ),
        _summary_card(
            "1s RPE RMSE",
            _lookup(metrics_payload, ("pose_metrics", "rpe_time_1s", "step3", "translation_part", "rmse")),
            "m",
        ),
        _summary_card("Step3 mode", _lookup(metrics_payload, ("step3_selection", "step3_alignment_mode"))),
        _summary_card(
            "Step3 rejected",
            _lookup(metrics_payload, ("step3_selection", "step3_rejection_ratio")),
            percent=True,
        ),
        _summary_card(
            "global gate",
            _lookup(metrics_payload, ("pose_metrics", "valid_segment", "step3", "success", "global_gate_failed")),
        ),
        _summary_card(
            "gate value",
            _lookup(metrics_payload, ("pose_metrics", "valid_segment", "step3", "success", "global_gate_value_m")),
            "m",
        ),
        _summary_card(
            "orientation",
            _lookup(metrics_payload, ("orientation_diagnostics", "orientation_unstable")),
        ),
        _summary_card("quality", _lookup(metrics_payload, ("metadata", "alignment_quality_label"))),
    ]


def _flatten_scalars(value: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 4) -> list[dict[str, str]]:
    if depth > max_depth:
        return []
    if isinstance(value, dict):
        rows: list[dict[str, str]] = []
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_scalars(item, prefix=child_prefix, depth=depth + 1, max_depth=max_depth))
        return rows
    if isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)):
        return [{"metric": prefix, "value": _scalar_text(value)}]
    return []


def _metric_detail_sections(metrics_payload: dict) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for name in (
        "time_alignment",
        "step2_residuals",
        "trajectory",
        "step3_selection",
        "alignment_quality",
        "rigid_alignability",
        "piecewise_diagnostics",
        "case_diagnostics",
        "orientation_diagnostics",
        "user_alert",
    ):
        block = metrics_payload.get(name, {}) if isinstance(metrics_payload, dict) else {}
        rows = _flatten_scalars(block, max_depth=3)
        if rows:
            sections.append({"name": name, "rows": rows})

    success = _lookup(metrics_payload, ("pose_metrics", "valid_segment", "step3", "success"))
    success_rows = _flatten_scalars(success, max_depth=3)
    if success_rows:
        sections.append({"name": "valid_segment.step3.success", "rows": success_rows})
    return sections


def _figure_links(metrics_payload: dict, base_dir: Path) -> list[dict[str, str]]:
    plot_files = _lookup(metrics_payload, ("metadata", "plot", "files"))
    if not isinstance(plot_files, list):
        return []
    wanted_tokens = (
        "_hist.png",
        "_hist_p95.png",
        "_hist_p99.png",
        "_series.png",
        "_series_p95.png",
        "_series_p99.png",
        "_raw.png",
        "_raw_p95.png",
        "_raw_p99.png",
        "_stats_core.png",
        "_box_p95.png",
        "_box_p99.png",
        "_violin_p95.png",
        "_violin_p99.png",
        "rpe_time_1s_translation_part",
    )
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in plot_files:
        path = Path(str(item))
        name = path.name
        if not any(token in name for token in wanted_tokens):
            continue
        try:
            href = os.path.relpath(path, start=base_dir)
        except ValueError:
            href = str(path)
        if href in seen:
            continue
        seen.add(href)
        links.append({"label": name, "href": href})
    priority = {
        "rpe_time_1s_translation_part_series.png": 0,
        "rpe_time_1s_translation_part_stats.png": 1,
        "rpe_time_1s_translation_part_box.png": 2,
        "ape_translation_part_hist.png": 3,
        "ape_translation_part_hist_p95.png": 4,
        "ape_translation_part_hist_p99.png": 5,
        "rpe_translation_part_hist.png": 6,
        "rpe_translation_part_hist_p95.png": 7,
        "rpe_translation_part_hist_p99.png": 8,
        "ape_translation_part_series_p95.png": 9,
        "ape_translation_part_series_p99.png": 10,
        "ape_translation_part_stats_core.png": 11,
        "ape_translation_part_box_p95.png": 12,
        "ape_translation_part_box_p99.png": 13,
        "ape_translation_part_violin_p95.png": 14,
        "ape_translation_part_violin_p99.png": 15,
        "rpe_translation_part_series_p95.png": 16,
        "rpe_translation_part_series_p99.png": 17,
        "rpe_translation_part_stats_core.png": 18,
        "rpe_translation_part_box_p95.png": 19,
        "rpe_translation_part_box_p99.png": 20,
        "rpe_translation_part_violin_p95.png": 21,
        "rpe_translation_part_violin_p99.png": 22,
        "rpe_time_1s_translation_part_series_p95.png": 23,
        "rpe_time_1s_translation_part_series_p99.png": 24,
        "rpe_time_1s_translation_part_stats_core.png": 25,
        "rpe_time_1s_translation_part_box_p95.png": 26,
        "rpe_time_1s_translation_part_box_p99.png": 27,
        "rpe_time_1s_translation_part_raw.png": 50,
        "ape_translation_part_raw_p95.png": 59,
        "ape_translation_part_raw_p99.png": 60,
        "rpe_translation_part_raw_p95.png": 66,
        "rpe_translation_part_raw_p99.png": 67,
        "rpe_time_1s_translation_part_raw_p95.png": 73,
        "rpe_time_1s_translation_part_raw_p99.png": 74,
    }
    links.sort(key=lambda item: (priority.get(item["label"], 100), item["label"]))
    return links


def write_interactive_run_html(
    out_path: Path,
    *,
    title: str,
    metrics_payload: dict,
    pos_gt: Any,
    pr_sync: Any,
    pr_corrected: Any,
    pr_final: Any,
    timestamps_s: Any | None = None,
    time_alignment: dict[str, Any] | None = None,
    x_dimension: str = "seconds",
    max_points: int = DEFAULT_MAX_POINTS,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_meta = metrics_payload.get("metadata", {}).get("plot", {}) if isinstance(metrics_payload, dict) else {}
    metadata = metrics_payload.get("metadata", {}) if isinstance(metrics_payload, dict) else {}
    ape_relation = str(plot_meta.get("ape_relation", "translation_part"))
    rpe_relation = str(plot_meta.get("rpe_relation", "translation_part"))
    x_dimension = str(plot_meta.get("x_dimension", x_dimension))
    debug_outputs = bool(metadata.get("debug", False))
    metric_stages = ("raw", "step2", "step3") if debug_outputs else ("step3",)
    step3_selection = metrics_payload.get("step3_selection", {}) if isinstance(metrics_payload, dict) else {}
    trajectory = {
        "gt": _xyz_trace(pos_gt, max_points=max_points),
        "step3": _xyz_trace(pr_final, max_points=max_points),
        "step3Err": _error_xyz(pos_gt, pr_final, max_points=max_points),
    }
    if debug_outputs:
        trajectory.update(
            {
                "raw": _xyz_trace(pr_sync, max_points=max_points),
                "step2": _xyz_trace(pr_corrected, max_points=max_points),
                "rawErr": _error_xyz(pos_gt, pr_sync, max_points=max_points),
                "step2Err": _error_xyz(pos_gt, pr_corrected, max_points=max_points),
            }
        )
    time_block = dict(time_alignment or {})
    lag_times = np.asarray(time_block.get("lags", []), dtype=float).reshape(-1) * float(time_block.get("dt_resample", 1.0))
    data = {
        "title": str(title or "EPA Interactive Report"),
        "debug": debug_outputs,
        "trajectory": trajectory,
        "step3": {
            "mode": str(step3_selection.get("step3_alignment_mode", "")),
            "inliers": int(_finite_or_none(step3_selection.get("step3_inlier_count")) or 0),
            "rejected": int(_finite_or_none(step3_selection.get("step3_rejected_count")) or 0),
        },
        "time": {
            "t": _to_float_list(time_block.get("t_uniform", []), max_points=max_points),
            "gtOmega": _to_float_list(time_block.get("sig_gt", []), max_points=max_points),
            "estOmega": _to_float_list(time_block.get("sig_est", []), max_points=max_points),
            "lag": _to_float_list(lag_times, max_points=max_points),
            "corr": _to_float_list(time_block.get("corr", []), max_points=max_points),
            "offset": _finite_or_none(time_block.get("calculated_offset")) or 0.0,
        },
        "speed": _speed_plot_data(
            timestamps_s,
            pos_gt=pos_gt,
            pr_sync=pr_sync,
            pr_corrected=pr_corrected,
            pr_final=pr_final,
            debug=debug_outputs,
            max_points=max_points,
        ),
        "metrics": [
            _metric_plot_data(
                metrics_payload,
                metric_kind="ape",
                relation=ape_relation,
                x_dimension=x_dimension,
                max_points=max_points,
                stages=metric_stages,
            ),
            _metric_plot_data(
                metrics_payload,
                metric_kind="rpe",
                relation=rpe_relation,
                x_dimension=x_dimension,
                max_points=max_points,
                stages=metric_stages,
            ),
            _metric_plot_data(
                metrics_payload,
                metric_kind="rpe_time_1s",
                relation="translation_part",
                x_dimension=x_dimension,
                max_points=max_points,
                stages=metric_stages,
            ),
        ],
        "metricSummary": _metric_summary_cards(
            metrics_payload,
            ape_relation=ape_relation,
            rpe_relation=rpe_relation,
        ),
        "metricDetails": _metric_detail_sections(metrics_payload),
        "figureLinks": _figure_links(metrics_payload, out_path.parent),
    }
    payload_json = json.dumps(data, ensure_ascii=False, allow_nan=False)
    debug_sections = (
        """
<section class="panel" data-panel="timeSignals">
  <div class="panelHeader" draggable="true"><h2>Step 1 Signals</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="timeSignals">x</button></div>
  <div id="timeSignals" class="plot small"></div>
</section>
<section class="panel" data-panel="timeCorrelation">
  <div class="panelHeader" draggable="true"><h2>Step 1 Correlation</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="timeCorrelation">x</button></div>
  <div id="timeCorrelation" class="plot small"></div>
</section>
"""
        if debug_outputs
        else ""
    )
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{data["title"]}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f7f8fb;color:#17202a}}
header{{padding:18px 22px;background:#fff;border-bottom:1px solid #dce3ed;position:sticky;top:0;z-index:2}}
h1{{font-size:22px;margin:0 0 6px;overflow-wrap:anywhere;line-height:1.2}} .meta{{display:flex;gap:14px;color:#52606d;font-size:13px;flex-wrap:wrap}}
main{{padding:18px 22px}}
.toolbar{{display:flex;justify-content:flex-end;margin:0 0 12px}}
.toolbar button,.panelClose{{border:1px solid #ccd5e1;background:#fff;color:#334155;border-radius:4px;cursor:pointer}}
.toolbar button{{padding:7px 10px;font-size:13px}}
.dashboard{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;align-items:start}}
.panel{{background:#fff;border:1px solid #dce3ed;border-radius:6px;padding:12px;min-width:0}}
.panel.dragging{{opacity:.55}}
.panel.span2{{grid-column:span 2}}
.panelHeader{{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 0 10px;cursor:grab;user-select:none}}
.panelHeader:active{{cursor:grabbing}}
.panel h2{{font-size:16px;margin:0;line-height:1.25;overflow-wrap:anywhere}}
.panelClose{{width:26px;height:26px;line-height:20px;font-size:14px;flex:0 0 auto}}
.plot{{height:560px}} .plot.small{{height:390px}}
.trajectoryControls{{display:flex;align-items:center;gap:12px;margin:0 0 10px;color:#52606d;font-size:13px}}
.trajectoryControls input{{flex:1;min-width:180px}}
.trajectoryControls output{{min-width:48px;text-align:right;font-variant-numeric:tabular-nums}}
.metricCards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}
.metricCard{{border:1px solid #dce3ed;border-radius:6px;padding:10px;background:#fbfcfe;min-width:0;overflow:hidden}}
.metricCard.span2{{grid-column:span 2}}
.metricLabel{{font-size:12px;color:#667085;margin-bottom:4px}}
.metricValue{{font-size:16px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.22;overflow-wrap:anywhere;word-break:break-word;white-space:normal;max-width:100%}}
.metricValue.compact{{font-size:13px;font-weight:600;line-height:1.28}}
details{{margin-top:12px}} summary{{cursor:pointer;color:#334155;font-weight:600}}
.detailsGrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-top:10px}}
.metricTable{{width:100%;border-collapse:collapse;font-size:12px}}
.metricTable th,.metricTable td{{padding:5px 6px;border-bottom:1px solid #e6ebf2;text-align:left;vertical-align:top}}
.metricTable th{{color:#52606d;background:#f5f7fb}}
.metricTable td:last-child{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.figureLinks{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}}
.figureLinks a{{display:inline-block;border:1px solid #dce3ed;border-radius:4px;padding:6px 8px;color:#1d4ed8;text-decoration:none;background:#fbfcfe;font-size:13px}}
.fallback{{padding:12px;background:#fff3cd;border:1px solid #ffec99;border-radius:6px;display:none}}
@media (max-width: 980px){{.dashboard{{grid-template-columns:1fr}}.panel.span2{{grid-column:span 1}}.plot{{height:460px}}}}
</style>
</head>
<body>
<header>
<h1>{data["title"]}</h1>
<div class="meta"><span>Step3 mode: {data["step3"]["mode"]}</span><span>inliers: {data["step3"]["inliers"]}</span><span>rejected: {data["step3"]["rejected"]}</span></div>
</header>
<main>
<div id="plotlyFallback" class="fallback">Plotly did not load. Check network access or use a browser with access to cdn.plot.ly.</div>
<div class="toolbar"><button id="resetLayout" type="button">Reset layout</button></div>
<div id="dashboard" class="dashboard">
<section class="panel span2" data-panel="summary">
  <div class="panelHeader" draggable="true"><h2>Metrics Summary</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="summary">x</button></div>
  <div id="metricCards" class="metricCards"></div><div id="figureLinks" class="figureLinks"></div><details><summary>Metrics details</summary><div id="metricDetails" class="detailsGrid"></div></details>
</section>
<section class="panel span2" data-panel="trajectory">
  <div class="panelHeader" draggable="true"><h2>3D Trajectory</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="trajectory">x</button></div>
  <div class="trajectoryControls"><span>progress</span><input id="trajectoryProgress" type="range" min="2" max="100" value="100"><output id="trajectoryProgressLabel">100%</output></div><div id="trajectory3d" class="plot"></div>
</section>
<section class="panel span2" data-panel="speed">
  <div class="panelHeader" draggable="true"><h2>Linear Velocity</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="speed">x</button></div>
  <div id="linearVelocity" class="plot small"></div>
</section>
<section class="panel" data-panel="ape">
  <div class="panelHeader" draggable="true"><h2>APE</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="ape">x</button></div>
  <div id="metric0" class="plot small"></div>
</section>
<section class="panel" data-panel="apeDistribution">
  <div class="panelHeader" draggable="true"><h2>APE Distribution</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="apeDistribution">x</button></div>
  <div id="metricBox0" class="plot small"></div>
</section>
<section class="panel" data-panel="rpe1s">
  <div class="panelHeader" draggable="true"><h2>1s RPE</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="rpe1s">x</button></div>
  <div id="metric2" class="plot small"></div>
</section>
<section class="panel" data-panel="rpe1sDistribution">
  <div class="panelHeader" draggable="true"><h2>1s RPE Distribution</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="rpe1sDistribution">x</button></div>
  <div id="metricBox2" class="plot small"></div>
</section>
<section class="panel" data-panel="rpe">
  <div class="panelHeader" draggable="true"><h2>RPE</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="rpe">x</button></div>
  <div id="metric1" class="plot small"></div>
</section>
<section class="panel" data-panel="rpeDistribution">
  <div class="panelHeader" draggable="true"><h2>RPE Distribution</h2><button class="panelClose" type="button" title="Close panel" data-close-panel="rpeDistribution">x</button></div>
  <div id="metricBox1" class="plot small"></div>
</section>
{debug_sections}</div>
</main>
<script id="epaInteractiveData" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById('epaInteractiveData').textContent);
const config = {{responsive:true, scrollZoom:false, displaylogo:false}};
const defaultPanelOrder = ['summary','trajectory','speed','ape','apeDistribution','rpe1s','rpe1sDistribution','rpe','rpeDistribution'].concat(payload.debug ? ['timeSignals','timeCorrelation'] : []);
function layout(title, xTitle, yTitle) {{
  return {{title, margin:{{l:58,r:20,t:42,b:52}}, hovermode:'closest', dragmode:'pan', xaxis:{{title:xTitle}}, yaxis:{{title:yTitle}}, legend:{{orientation:'h'}}}};
}}
function line3d(name, obj, color, visible) {{
  return {{type:'scatter3d', mode:'lines', name, x:obj.x, y:obj.y, z:obj.z, visible, line:{{color, width:4}}}};
}}
function sliceTrace(obj, pct) {{
  const n = obj.x.length;
  const keep = Math.max(2, Math.ceil(n * pct / 100));
  return {{x: obj.x.slice(0, keep), y: obj.y.slice(0, keep), z: obj.z.slice(0, keep)}};
}}
function sliceArray(values, pct) {{
  const keep = Math.max(2, Math.ceil(values.length * pct / 100));
  return values.slice(0, keep);
}}
function trajectoryTraces(pct) {{
  const tr = payload.trajectory;
  const gt = sliceTrace(tr.gt, pct);
  const step3 = sliceTrace(tr.step3, pct);
  const traces = [
    line3d('ground truth', gt, '#20242a', true),
    {{type:'scatter3d', mode:'lines', name:'step3', x:step3.x, y:step3.y, z:step3.z,
      line:{{color:sliceArray(tr.step3Err, pct), colorscale:'Jet', width:6, colorbar:{{title:'err m'}}}}}}
  ];
  if (tr.raw && tr.step2) {{
    traces.splice(1, 0,
      line3d('raw', sliceTrace(tr.raw, pct), '#c44e52', 'legendonly'),
      line3d('step2', sliceTrace(tr.step2, pct), '#dd9c32', 'legendonly')
    );
  }}
  return traces;
}}
function currentTrajectoryCamera() {{
  const plot = document.getElementById('trajectory3d');
  const scene = (plot.layout && plot.layout.scene) || (plot._fullLayout && plot._fullLayout.scene);
  if (!scene || !scene.camera) return null;
  return JSON.parse(JSON.stringify(scene.camera));
}}
function trajectoryLayout(camera) {{
  const scene = {{aspectmode:'data'}};
  if (camera) scene.camera = camera;
  return {{
    title:'Trajectory alignment',
    margin:{{l:0,r:0,t:36,b:0}},
    scene,
    legend:{{orientation:'h'}},
    dragmode:'orbit',
    uirevision:'trajectory-camera'
  }};
}}
function panelById(panelId) {{
  return document.querySelector(`[data-panel="${{panelId}}"]`);
}}
function resizePlots() {{
  document.querySelectorAll('.plot').forEach(el => {{
    if (el.offsetParent !== null && window.Plotly) Plotly.Plots.resize(el);
  }});
}}
function setupDashboard() {{
  const dashboard = document.getElementById('dashboard');
  let draggedPanel = null;
  document.querySelectorAll('.panelHeader').forEach(handle => {{
    handle.addEventListener('dragstart', event => {{
      draggedPanel = handle.closest('.panel');
      draggedPanel.classList.add('dragging');
      event.dataTransfer.effectAllowed = 'move';
      event.dataTransfer.setData('text/plain', draggedPanel.dataset.panel);
    }});
    handle.addEventListener('dragend', () => {{
      if (draggedPanel) draggedPanel.classList.remove('dragging');
      draggedPanel = null;
      setTimeout(resizePlots, 80);
    }});
  }});
  document.querySelectorAll('.panel').forEach(panel => {{
    panel.addEventListener('dragover', event => event.preventDefault());
    panel.addEventListener('drop', event => {{
      event.preventDefault();
      if (!draggedPanel || draggedPanel === panel) return;
      dashboard.insertBefore(draggedPanel, panel);
    }});
  }});
  dashboard.addEventListener('dragover', event => event.preventDefault());
  dashboard.addEventListener('drop', event => {{
    if (!draggedPanel || event.target !== dashboard) return;
    event.preventDefault();
    dashboard.appendChild(draggedPanel);
  }});
  document.querySelectorAll('[data-close-panel]').forEach(button => {{
    button.addEventListener('click', () => {{
      const panel = panelById(button.dataset.closePanel);
      if (panel) panel.hidden = true;
    }});
  }});
  const reset = document.getElementById('resetLayout');
  reset.addEventListener('click', () => {{
    defaultPanelOrder.forEach(panelId => {{
      const panel = panelById(panelId);
      if (!panel) return;
      panel.hidden = false;
      dashboard.appendChild(panel);
    }});
    setTimeout(resizePlots, 80);
  }});
}}
function renderMetrics() {{
  const cards = document.getElementById('metricCards');
  cards.innerHTML = payload.metricSummary.map(item => `
    <div class="metricCard${{String(item.value).length > 28 ? ' span2' : ''}}"><div class="metricLabel">${{item.label}}</div><div class="metricValue${{String(item.value).length > 28 ? ' compact' : ''}}">${{item.value}}</div></div>
  `).join('');
  const figureLinks = document.getElementById('figureLinks');
  figureLinks.innerHTML = payload.figureLinks.map(item => `<a href="${{item.href}}">${{item.label}}</a>`).join('');
  const details = document.getElementById('metricDetails');
  details.innerHTML = payload.metricDetails.map(section => `
    <div>
      <h3>${{section.name}}</h3>
      <table class="metricTable"><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>
      ${{section.rows.map(row => `<tr><td>${{row.metric}}</td><td>${{row.value}}</td></tr>`).join('')}}
      </tbody></table>
    </div>
  `).join('');
}}
function renderPlot(id, traces, plotLayout) {{
  const el = document.getElementById(id);
  if (!el || !traces || traces.length === 0) return;
  Plotly.newPlot(id, traces, plotLayout, config);
}}
function render() {{
  if (!window.Plotly) {{
    document.getElementById('plotlyFallback').style.display = 'block';
    return;
  }}
  setupDashboard();
  renderMetrics();
  renderPlot('trajectory3d', trajectoryTraces(100), trajectoryLayout(null));
  const progress = document.getElementById('trajectoryProgress');
  const progressLabel = document.getElementById('trajectoryProgressLabel');
  progress.addEventListener('input', () => {{
    const pct = Number(progress.value);
    const camera = currentTrajectoryCamera();
    progressLabel.textContent = `${{pct}}%`;
    Plotly.react('trajectory3d', trajectoryTraces(pct), trajectoryLayout(camera), config);
  }});

  const speedTraces = payload.speed.traces.map(t => ({{type:'scattergl', mode:'lines', name:t.label, x:t.x, y:t.y}}));
  renderPlot('linearVelocity', speedTraces, layout(payload.speed.title, payload.speed.xLabel, payload.speed.yLabel));

  renderPlot('timeSignals', [
    {{type:'scattergl', mode:'lines', name:'GT omega', x:payload.time.t, y:payload.time.gtOmega}},
    {{type:'scattergl', mode:'lines', name:'EST omega', x:payload.time.t, y:payload.time.estOmega}}
  ], layout('Angular velocity norm', 'time (s)', 'omega norm'));
  const corrTraces = [{{type:'scattergl', mode:'lines', name:'correlation', x:payload.time.lag, y:payload.time.corr}}];
  if (payload.time.corr.length > 0) {{
    corrTraces.push({{type:'scatter', mode:'lines', name:'selected offset', x:[payload.time.offset, payload.time.offset], y:[Math.min(...payload.time.corr), Math.max(...payload.time.corr)], line:{{dash:'dash', color:'#d62728'}}}});
  }}
  renderPlot('timeCorrelation', corrTraces, layout('Cross-correlation vs lag', 'lag (s)', 'correlation'));

  payload.metrics.forEach((metric, idx) => {{
    const traces = metric.traces.map(t => ({{type:'scattergl', mode:'lines', name:t.stage, x:t.x, y:t.y}}));
    renderPlot(`metric${{idx}}`, traces, layout(metric.title, metric.xLabel, metric.yLabel));
    const boxes = metric.boxes.map(t => ({{type:'box', name:t.stage, y:t.y, boxpoints:false}}));
    renderPlot(`metricBox${{idx}}`, boxes, layout(`${{metric.title}} distribution`, 'stage', metric.yLabel));
  }});
}}
window.addEventListener('load', render);
</script>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")
    return out_path
