from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-epa")
import matplotlib

matplotlib.use("Agg")
try:
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401
except Exception:
    _Axes3D = None
import matplotlib.pyplot as plt

from epa.metric_cli_common import load_ros_map_spec, resolve_map_tile_contextily

PLOT_BUNDLE_SCHEMA = "epa.plot_bundle"
PLOT_BUNDLE_VERSION = 1


def _to_array(values: Any, width: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if width is None:
        return arr.reshape(-1)
    return arr.reshape(-1, width)


def _to_path_text(path_like: str | Path | None) -> str:
    text = "" if path_like is None else str(path_like).strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def _compute_color_bounds(
    values: np.ndarray,
    cmin: float | None,
    cmax: float | None,
    cmax_percentile: float | None,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite)) if cmin is None else float(cmin)
    if cmax_percentile is not None:
        vmax = float(np.percentile(finite, float(cmax_percentile)))
    else:
        vmax = float(np.max(finite)) if cmax is None else float(cmax)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax


def make_raw_line_spec(
    *,
    name: str,
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
) -> dict[str, Any]:
    return {
        "type": "raw_line",
        "name": str(name),
        "title": str(title),
        "x_label": str(x_label),
        "y_label": str(y_label),
        "x": _to_array(x).tolist(),
        "y": _to_array(y).tolist(),
    }


def make_trajectory_error_map_spec(
    *,
    name: str,
    title: str,
    plot_mode: str,
    ref_positions: np.ndarray,
    est_positions: np.ndarray,
    scatter_positions: np.ndarray,
    errors: np.ndarray,
    cmin: float | None,
    cmax: float | None,
    cmax_percentile: float | None,
    ros_map_yaml: str | None,
    map_tile: str | None,
) -> dict[str, Any]:
    return {
        "type": "trajectory_error_map",
        "name": str(name),
        "title": str(title),
        "plot_mode": str(plot_mode),
        "ref_positions": _to_array(ref_positions, 3).tolist(),
        "est_positions": _to_array(est_positions, 3).tolist(),
        "scatter_positions": _to_array(scatter_positions, 3).tolist(),
        "errors": _to_array(errors).tolist(),
        "cmin": None if cmin is None else float(cmin),
        "cmax": None if cmax is None else float(cmax),
        "cmax_percentile": None if cmax_percentile is None else float(cmax_percentile),
        "ros_map_yaml": _to_path_text(ros_map_yaml),
        "map_tile": "" if map_tile is None else str(map_tile),
    }


def make_plot_bundle(
    *,
    tool: str,
    figures: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": PLOT_BUNDLE_SCHEMA,
        "version": PLOT_BUNDLE_VERSION,
        "tool": str(tool),
        "figures": figures,
        "metadata": dict(metadata or {}),
    }


def save_plot_bundle(bundle: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path).expanduser().resolve()
    if out.suffix.lower() != ".json":
        out = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return out


def load_plot_bundle(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Serialized plot bundle not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid serialized plot bundle: {p}")
    schema = data.get("schema", "")
    if schema != PLOT_BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported plot bundle schema: {schema}")
    if not isinstance(data.get("figures"), list) or not data["figures"]:
        raise ValueError("Serialized plot bundle has no figures.")
    return data


def iter_figure_specs(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    figures = bundle.get("figures", [])
    if not isinstance(figures, list):
        raise ValueError("plot bundle 'figures' must be a list.")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(figures):
        if not isinstance(item, dict):
            raise ValueError(f"figure spec #{idx} must be an object.")
        out.append(item)
    return out


def _render_raw_line(spec: dict[str, Any], out_path: Path, dpi: int) -> None:
    x = _to_array(spec.get("x", []))
    y = _to_array(spec.get("y", []))
    n = min(x.size, y.size)
    if n <= 0:
        raise ValueError("raw_line figure has no data.")
    plt.figure(figsize=(10.8, 4.8))
    plt.plot(x[:n], y[:n], linewidth=1.4)
    plt.title(str(spec.get("title", "Raw values")))
    plt.xlabel(str(spec.get("x_label", "x")))
    plt.ylabel(str(spec.get("y_label", "y")))
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close()


def _render_trajectory_error_map(spec: dict[str, Any], out_path: Path, dpi: int) -> None:
    pos_ref = _to_array(spec.get("ref_positions", []), 3)
    pos_est = _to_array(spec.get("est_positions", []), 3)
    pos_scatter = _to_array(spec.get("scatter_positions", []), 3)
    errors = _to_array(spec.get("errors", []))
    n_scatter = min(pos_scatter.shape[0], errors.size)
    pos_scatter = pos_scatter[:n_scatter]
    errors = errors[:n_scatter]

    mode = str(spec.get("plot_mode", "xyz")).lower()
    cmin = spec.get("cmin", None)
    cmax = spec.get("cmax", None)
    cmax_percentile = spec.get("cmax_percentile", None)
    vmin, vmax = _compute_color_bounds(errors, cmin, cmax, cmax_percentile)

    scatter = None
    if mode == "xyz":
        fig = plt.figure(figsize=(9.2, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        if pos_ref.size:
            ax.plot(
                pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2],
                color="black", alpha=0.35, linewidth=1.2, label="ref"
            )
        if pos_est.size:
            ax.plot(
                pos_est[:, 0], pos_est[:, 1], pos_est[:, 2],
                color="gray", alpha=0.2, linewidth=0.9, label="est"
            )
        if pos_scatter.size and errors.size:
            scatter = ax.scatter(
                pos_scatter[:, 0],
                pos_scatter[:, 1],
                pos_scatter[:, 2],
                c=errors,
                s=8,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    else:
        axis_id = {"x": 0, "y": 1, "z": 2}
        if len(mode) != 2 or mode[0] not in axis_id or mode[1] not in axis_id:
            raise ValueError(f"Unsupported plot mode: {mode}")
        i0, i1 = axis_id[mode[0]], axis_id[mode[1]]
        fig = plt.figure(figsize=(9.2, 7.0))
        ax = fig.add_subplot(111)
        if pos_ref.size:
            ax.plot(
                pos_ref[:, i0], pos_ref[:, i1],
                color="black", alpha=0.35, linewidth=1.2, label="ref"
            )
        if pos_est.size:
            ax.plot(
                pos_est[:, i0], pos_est[:, i1],
                color="gray", alpha=0.2, linewidth=0.9, label="est"
            )
        if pos_scatter.size and errors.size:
            scatter = ax.scatter(
                pos_scatter[:, i0],
                pos_scatter[:, i1],
                c=errors,
                s=8,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
        ax.set_xlabel(f"{mode[0]} (m)")
        ax.set_ylabel(f"{mode[1]} (m)")
        ax.axis("equal")

        ros_map_yaml = str(spec.get("ros_map_yaml", "") or "")
        if ros_map_yaml:
            if mode != "xy":
                raise ValueError("--ros_map_yaml currently requires --plot_mode xy")
            ros_map_spec = load_ros_map_spec(ros_map_yaml)
            img = plt.imread(str(ros_map_spec.image_path))
            h, w = img.shape[:2]
            extent = [
                ros_map_spec.origin_x,
                ros_map_spec.origin_x + float(w) * ros_map_spec.resolution,
                ros_map_spec.origin_y,
                ros_map_spec.origin_y + float(h) * ros_map_spec.resolution,
            ]
            ax.imshow(
                img,
                extent=extent,
                origin="lower",
                cmap="gray",
                alpha=0.45,
                zorder=-10,
            )

        map_tile = str(spec.get("map_tile", "") or "")
        if map_tile:
            if mode != "xy":
                raise ValueError("--map_tile currently requires --plot_mode xy")
            try:
                import contextily as cx  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Map tile overlay requires contextily. Install with: pip install contextily"
                ) from exc
            cx.add_basemap(ax, **resolve_map_tile_contextily(cx, map_tile))

    ax.set_title(str(spec.get("title", "Trajectory map")))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def render_figure_spec(spec: dict[str, Any], out_path: str | Path, dpi: int = 180) -> Path:
    p = Path(out_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    figure_type = str(spec.get("type", "")).strip()
    if figure_type == "raw_line":
        _render_raw_line(spec, p, dpi=int(dpi))
    elif figure_type == "trajectory_error_map":
        _render_trajectory_error_map(spec, p, dpi=int(dpi))
    else:
        raise ValueError(f"Unsupported figure type: {figure_type}")
    return p
