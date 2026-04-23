from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-epa"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NUMERIC_KEYS = {
    "epa_offset_est_s",
    "evo_offset_s",
    "epa_ate_rmse_raw_m",
    "epa_ate_rmse_step3_m",
    "evo_ape_raw_rmse_m",
    "evo_ape_se3_rmse_m",
    "epa_improve_pct",
    "evo_improve_pct",
    "epa_xcorr_peak",
    "epa_xcorr_psr",
    "epa_omega_improve_pct",
    "epa_matches_equivalent",
    "evo_matches",
    "evo_sweep_evals",
}


def _to_float(text: str) -> float:
    value = text.strip()
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _load_summary_rows(summary_csv: Path) -> list[dict[str, object]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, object]] = []
        for row in reader:
            casted: dict[str, object] = dict(row)
            for key in NUMERIC_KEYS:
                casted[key] = _to_float(row.get(key, ""))
            rows.append(casted)
    return rows


def _valid(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_status_counts(rows: list[dict[str, object]], out_dir: Path) -> Path:
    both_ok = sum(1 for row in rows if row.get("status") == "ok")
    both_fail = sum(1 for row in rows if row.get("status") != "ok")
    epa_fail = sum(1 for row in rows if row.get("epa_status") != "ok")
    evo_fail = sum(1 for row in rows if row.get("evo_status") != "ok")

    labels = ["both_ok", "both_fail", "epa_fail", "evo_fail"]
    vals = [both_ok, both_fail, epa_fail, evo_fail]
    colors = ["#2e7d32", "#c62828", "#ef6c00", "#1565c0"]

    plt.figure(figsize=(8, 4.2))
    bars = plt.bar(labels, vals, color=colors)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val + 0.5, str(val), ha="center")
    plt.ylabel("Case Count")
    plt.title("Benchmark Status Counts")
    out = out_dir / "status_counts.png"
    _save_fig(out)
    return out


def _plot_log_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> Path | None:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(mask.sum()) == 0:
        return None

    xs = x[mask]
    ys = y[mask]
    lo = float(min(np.min(xs), np.min(ys)))
    hi = float(max(np.max(xs), np.max(ys)))
    lo = max(lo, 1e-6)
    hi = max(hi, lo * 1.05)

    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(xs, ys, s=18, alpha=0.75, c="#00695c")
    plt.plot([lo, hi], [lo, hi], "--", color="#616161", linewidth=1.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    _save_fig(out_path)
    return out_path


def _plot_linear_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> Path | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) == 0:
        return None

    xs = x[mask]
    ys = y[mask]
    lo = float(min(np.min(xs), np.min(ys)))
    hi = float(max(np.max(xs), np.max(ys)))
    pad = max((hi - lo) * 0.06, 1e-9)

    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(xs, ys, s=18, alpha=0.75, c="#3949ab")
    plt.plot([lo, hi], [lo, hi], "--", color="#616161", linewidth=1.2)
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    _save_fig(out_path)
    return out_path


def _plot_offset_hist(rows: list[dict[str, object]], out_dir: Path) -> Path | None:
    epa = _valid([float(row["epa_offset_est_s"]) for row in rows])
    evo = _valid([float(row["evo_offset_s"]) for row in rows])
    if epa.size == 0 and evo.size == 0:
        return None

    plt.figure(figsize=(8.5, 4.8))
    bins = 40
    if epa.size > 0:
        plt.hist(epa, bins=bins, alpha=0.55, label="epa_offset_est_s", color="#00897b")
    if evo.size > 0:
        plt.hist(evo, bins=bins, alpha=0.55, label="evo_offset_s", color="#5e35b1")
    plt.xlabel("Offset (s)")
    plt.ylabel("Case Count")
    plt.title("Offset Distribution")
    plt.legend()
    plt.grid(True, alpha=0.25)
    out = out_dir / "offset_distribution.png"
    _save_fig(out)
    return out


def _plot_dataset_box(rows: list[dict[str, object]], out_dir: Path) -> Path | None:
    data_by_dataset: dict[str, tuple[list[float], list[float]]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        if dataset not in data_by_dataset:
            data_by_dataset[dataset] = ([], [])
        data_by_dataset[dataset][0].append(float(row["epa_ate_rmse_step3_m"]))
        data_by_dataset[dataset][1].append(float(row["evo_ape_se3_rmse_m"]))

    if not data_by_dataset:
        return None

    labels = sorted(data_by_dataset.keys())
    v_data = []
    e_data = []
    for label in labels:
        v_vals = _valid(data_by_dataset[label][0])
        e_vals = _valid(data_by_dataset[label][1])
        if v_vals.size == 0:
            v_vals = np.array([np.nan])
        if e_vals.size == 0:
            e_vals = np.array([np.nan])
        v_data.append(v_vals)
        e_data.append(e_vals)

    pos = np.arange(len(labels), dtype=float)
    width = 0.34
    plt.figure(figsize=(11.2, 5.6))
    bp1 = plt.boxplot(v_data, positions=pos - width / 2.0, widths=width, patch_artist=True)
    bp2 = plt.boxplot(e_data, positions=pos + width / 2.0, widths=width, patch_artist=True)
    for patch in bp1["boxes"]:
        patch.set(facecolor="#26a69a", alpha=0.75)
    for patch in bp2["boxes"]:
        patch.set(facecolor="#7e57c2", alpha=0.75)
    plt.yscale("log")
    plt.xticks(pos, labels, rotation=25, ha="right")
    plt.ylabel("Aligned RMSE (m, log scale)")
    plt.title("Aligned RMSE by Dataset (epa vs evo)")
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ["epa", "evo"], loc="upper right")
    plt.grid(True, axis="y", alpha=0.25)
    out = out_dir / "aligned_rmse_by_dataset_box.png"
    _save_fig(out)
    return out


def _plot_top_cases(rows: list[dict[str, object]], out_dir: Path, top_k: int = 20) -> Path | None:
    ranked = []
    for row in rows:
        v = float(row["epa_ate_rmse_step3_m"])
        e = float(row["evo_ape_se3_rmse_m"])
        score = max(v if math.isfinite(v) else float("-inf"), e if math.isfinite(e) else float("-inf"))
        if math.isfinite(score):
            ranked.append((str(row.get("case", "")), v, e, score))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[3], reverse=True)
    selected = ranked[:top_k]
    labels = [item[0] for item in selected]
    v_vals = [item[1] for item in selected]
    e_vals = [item[2] for item in selected]

    y = np.arange(len(labels), dtype=float)
    h = 0.38
    plt.figure(figsize=(12.2, max(6.0, 0.32 * len(labels))))
    plt.barh(y - h / 2.0, v_vals, height=h, color="#00897b", alpha=0.9, label="epa")
    plt.barh(y + h / 2.0, e_vals, height=h, color="#5e35b1", alpha=0.9, label="evo")
    plt.xscale("log")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Aligned RMSE (m, log scale)")
    plt.title(f"Top {len(labels)} Cases by Aligned RMSE")
    plt.legend()
    plt.grid(True, axis="x", alpha=0.25)
    plt.gca().invert_yaxis()
    out = out_dir / "top_cases_aligned_rmse.png"
    _save_fig(out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate visualization charts from alignanything harness summary.csv."
    )
    parser.add_argument("--summary-csv", required=True, help="Path to summary.csv.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for images (default: <summary_dir>/plots).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many cases to include in top-case bar chart.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    summary_csv = Path(args.summary_csv).resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary csv not found: {summary_csv}")
    rows = _load_summary_rows(summary_csv)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else summary_csv.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    produced.append(_plot_status_counts(rows, out_dir))

    v_aligned = np.asarray([float(row["epa_ate_rmse_step3_m"]) for row in rows], dtype=float)
    e_aligned = np.asarray([float(row["evo_ape_se3_rmse_m"]) for row in rows], dtype=float)
    v_improve = np.asarray([float(row["epa_improve_pct"]) for row in rows], dtype=float)
    e_improve = np.asarray([float(row["evo_improve_pct"]) for row in rows], dtype=float)
    v_matches = np.asarray([float(row["epa_matches_equivalent"]) for row in rows], dtype=float)
    e_matches = np.asarray([float(row["evo_matches"]) for row in rows], dtype=float)

    maybe = _plot_log_scatter(
        v_aligned,
        e_aligned,
        "epa aligned RMSE (m)",
        "evo aligned RMSE (m)",
        "Aligned RMSE: epa vs evo",
        out_dir / "aligned_rmse_scatter_log.png",
    )
    if maybe is not None:
        produced.append(maybe)

    maybe = _plot_linear_scatter(
        v_improve,
        e_improve,
        "epa improve (%)",
        "evo improve (%)",
        "Improve Percentage: epa vs evo",
        out_dir / "improve_pct_scatter.png",
    )
    if maybe is not None:
        produced.append(maybe)

    maybe = _plot_log_scatter(
        v_matches,
        e_matches,
        "epa matches",
        "evo matches",
        "Match Count: epa vs evo",
        out_dir / "matches_scatter_log.png",
    )
    if maybe is not None:
        produced.append(maybe)

    maybe = _plot_offset_hist(rows, out_dir)
    if maybe is not None:
        produced.append(maybe)

    maybe = _plot_dataset_box(rows, out_dir)
    if maybe is not None:
        produced.append(maybe)

    maybe = _plot_top_cases(rows, out_dir, top_k=max(1, int(args.top_k)))
    if maybe is not None:
        produced.append(maybe)

    print(f"Generated {len(produced)} plot(s) in: {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
