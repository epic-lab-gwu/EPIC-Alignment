import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from .math_utils import normalize_quat_array, normalize_time_to_seconds


def to_builtin(value):
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_metrics(output_dir, metrics_payload):
    output_dir = Path(output_dir)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics_summary.csv"

    json_path.write_text(
        json.dumps(to_builtin(metrics_payload), indent=2),
        encoding="utf-8",
    )

    rows = []
    for section, block in metrics_payload.items():
        if not isinstance(block, dict):
            continue
        for metric, value in block.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                rows.append(
                    {
                        "section": section,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    if rows:
        if pd is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        else:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
                writer.writeheader()
                writer.writerows(rows)


METRIC_ZH_EXPLAIN = {
    "offset_est_s": "估计得到的时间偏移（秒），表示 estimation 相对 GT 的时间差。",
    "evo_t_offset_used_s": "按 evo 时间关联语义使用的时间偏移（秒），等于 -offset_est_s（加到 estimation 时间戳上）。",
    "evo_match_max_diff_s": "按 evo 关联规则统计匹配时使用的最大时间差阈值（秒）。",
    "evo_matches_equivalent": "按 evo 最近邻时间关联规则得到的有效匹配对数量。",
    "offset_s": "最终采用的时间偏移参数（秒），用于把 estimation 时间轴对齐到 GT 时间轴。",
    "offset_err_ms": "时间偏移误差（毫秒），仅在 synthetic 有真值时有效。",
    "xcorr_peak_normalized": "互相关峰值归一化强度，越接近 1 表示时间匹配越强。",
    "xcorr_psr": "互相关峰值与旁瓣比（PSR），越大表示峰值越清晰唯一。",
    "omega_rmse_before": "时间对齐前，两路角速度模长差异的 RMSE（rad/s）。",
    "omega_rmse_after": "时间对齐后，两路角速度模长差异的 RMSE（rad/s）。",
    "omega_rmse_improve_pct": "时间对齐后 RMSE 改善百分比，越大越好。",
    "rot_res_mean_deg": "Step2 相对旋转约束残差均值（度），越小越好。",
    "rot_res_median_deg": "Step2 相对旋转约束残差中位数（度），越小越好。",
    "rot_res_p95_deg": "Step2 相对旋转约束残差 95 分位（度），越小越好。",
    "trans_eq_rmse_m": "Step2 平移方程残差 RMSE（米），越小越好。",
    "trans_eq_p95_m": "Step2 平移方程残差 95 分位（米），越小越好。",
    "translation_system_cond": "Step2 平移线性系统条件数，越接近 1 越稳定。",
    "translation_constraints": "Step2 有效平移约束数量，越多通常越稳。",
    "ate_rmse_raw_m": "未完成空间对齐前（Step1 后）的绝对轨迹误差 RMSE（米）。",
    "ate_rmse_step2_m": "完成 Step2 外参修正后的绝对轨迹误差 RMSE（米）。",
    "ate_rmse_step3_m": "完成 Step3 世界系对齐后的绝对轨迹误差 RMSE（米）。",
    "ate_p95_raw_m": "raw 绝对轨迹误差 95 分位（米）。",
    "ate_p95_step2_m": "step2 绝对轨迹误差 95 分位（米）。",
    "ate_p95_step3_m": "step3 绝对轨迹误差 95 分位（米）。",
    "ate_rmse_improve_raw_to_step3_pct": "从 raw 到 step3 的 RMSE 改善百分比。",
    "ate_rmse_improve_step2_to_step3_pct": "从 step2 到 step3 的 RMSE 改善百分比。",
    "extrinsic_rotation_error_deg": "外参旋转误差（度），仅 synthetic 有真值时有效。",
    "extrinsic_translation_error_m": "外参平移误差（米），仅 synthetic 有真值时有效。",
    "world_rotation_error_deg": "世界对齐旋转误差（度），仅 synthetic 有真值时有效。",
    "world_translation_error_m": "世界对齐平移误差（米），仅 synthetic 有真值时有效。",
}


def _as_float(block, key):
    if key not in block:
        return np.nan
    val = block[key]
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    return np.nan


def _analyze_metric_value(section, metric, value, block):
    if np.isnan(value):
        return "当前无真值或该指标不适用（nan）。"

    if metric == "offset_est_s":
        peak = _as_float(block, "xcorr_peak_normalized")
        psr = _as_float(block, "xcorr_psr")
        if np.isnan(peak) or np.isnan(psr):
            return f"估计时间偏移为 {value:.4f}s。"
        confidence = "高" if (peak >= 0.9 and psr >= 10) else ("中" if (peak >= 0.75 and psr >= 6) else "低")
        return (
            f"估计时间偏移为 {value:.4f}s；互相关峰值 {peak:.4f}、PSR {psr:.2f}，"
            f"峰值可信度为{confidence}。"
        )

    if metric == "evo_t_offset_used_s":
        est = _as_float(block, "offset_est_s")
        if np.isnan(est):
            return f"按 evo 语义采用 t_offset={value:.4f}s。"
        return (
            f"按 evo 语义采用 t_offset={value:.4f}s，"
            f"与本流程 offset_est_s={est:.4f}s 互为相反数。"
        )

    if metric == "evo_matches_equivalent":
        max_diff = _as_float(block, "evo_match_max_diff_s")
        if np.isnan(max_diff):
            return f"按 evo 关联规则得到 {value:.0f} 对匹配。"
        return f"按 evo 关联规则（max_diff={max_diff:.3f}s）得到 {value:.0f} 对匹配。"

    if metric == "xcorr_peak_normalized":
        if value >= 0.9:
            return f"归一化峰值为 {value:.4f}（>0.9），时间相关性很强。"
        if value >= 0.75:
            return f"归一化峰值为 {value:.4f}（0.75~0.9），时间相关性中等。"
        return f"归一化峰值为 {value:.4f}（<0.75），时间相关性偏弱。"

    if metric == "xcorr_psr":
        if value >= 10:
            return f"PSR 为 {value:.2f}（>=10），峰值清晰且唯一性较好。"
        if value >= 6:
            return f"PSR 为 {value:.2f}（6~10），峰值可用但需要留意次峰。"
        return f"PSR 为 {value:.2f}（<6），峰值分辨度偏低。"

    if metric in {"omega_rmse_before", "omega_rmse_after"}:
        before = _as_float(block, "omega_rmse_before")
        after = _as_float(block, "omega_rmse_after")
        if not np.isnan(before) and not np.isnan(after):
            delta = before - after
            ratio = delta / (before + 1e-12) * 100.0
            return (
                f"角速度 RMSE 从 {before:.4f} 降到 {after:.4f} rad/s，"
                f"绝对下降 {delta:.4f} rad/s（{ratio:.2f}%）。"
            )
        return f"该值为 {value:.6f}，越小越好。"

    if metric == "omega_rmse_improve_pct":
        if value >= 50:
            return f"时间对齐改善 {value:.2f}%（>50%），收益明显。"
        if value >= 20:
            return f"时间对齐改善 {value:.2f}%（20%~50%），有中等收益。"
        return f"时间对齐仅改善 {value:.2f}%（<20%），建议复核 offset。"

    if metric.startswith("rot_res_"):
        if value < 0.1:
            return f"旋转残差 {value:.4f} deg（<0.1），一致性很好。"
        if value < 1.0:
            return f"旋转残差 {value:.4f} deg（0.1~1），整体可接受。"
        return f"旋转残差 {value:.4f} deg（>=1），偏大。"

    if metric in {"trans_eq_rmse_m", "trans_eq_p95_m"}:
        cond = _as_float(block, "translation_system_cond")
        if np.isnan(cond):
            return f"平移残差为 {value:.6f} m，越小越好。"
        return (
            f"平移残差为 {value:.6f} m；对应系统条件数 {cond:.3f}，"
            f"{'解算稳定' if cond <= 5 else '解算稳定性一般'}。"
        )

    if metric == "translation_system_cond":
        if value <= 5:
            return f"条件数为 {value:.3f}（接近 1），平移解算稳定。"
        if value <= 20:
            return f"条件数为 {value:.3f}，可解但有一定病态风险。"
        return f"条件数为 {value:.3f}，病态风险较高。"

    if metric == "translation_constraints":
        if value >= 10000:
            return f"有效约束 {value:.0f} 条，约束非常充足。"
        if value >= 1000:
            return f"有效约束 {value:.0f} 条，数量充足。"
        return f"有效约束仅 {value:.0f} 条，可能受噪声影响。"

    if metric in {"ate_rmse_raw_m", "ate_rmse_step2_m", "ate_rmse_step3_m", "ate_p95_raw_m", "ate_p95_step2_m", "ate_p95_step3_m"}:
        raw = _as_float(block, "ate_rmse_raw_m")
        step2 = _as_float(block, "ate_rmse_step2_m")
        step3 = _as_float(block, "ate_rmse_step3_m")
        if metric == "ate_rmse_step3_m":
            if not np.isnan(raw):
                drop = raw - value
                ratio = drop / (raw + 1e-12) * 100.0
                quality = (
                    "达到 10cm 内" if value < 0.1 else
                    "达到分米级" if value < 0.3 else
                    "仍偏大"
                )
                return (
                    f"Step3 RMSE={value:.4f} m（{quality}）；"
                    f"相对 raw 降低 {drop:.4f} m（{ratio:.2f}%）。"
                )
        if metric == "ate_rmse_step2_m" and not np.isnan(step3):
            delta = value - step3
            return f"Step2 RMSE={value:.4f} m；比 Step3 高 {delta:.4f} m。"
        if metric == "ate_rmse_raw_m" and not np.isnan(step3):
            delta = value - step3
            return f"Raw RMSE={value:.4f} m；比 Step3 高 {delta:.4f} m。"
        if metric == "ate_p95_step3_m":
            return f"Step3 P95={value:.4f} m，表示 95% 时刻误差不超过该值。"
        if metric == "ate_p95_step2_m" and not np.isnan(step2):
            return f"Step2 P95={value:.4f} m。"
        if metric == "ate_p95_raw_m" and not np.isnan(raw):
            return f"Raw P95={value:.4f} m。"
        return f"该值为 {value:.6f}，越小越好。"

    if metric == "ate_rmse_improve_raw_to_step3_pct":
        raw = _as_float(block, "ate_rmse_raw_m")
        step3 = _as_float(block, "ate_rmse_step3_m")
        if not np.isnan(raw) and not np.isnan(step3):
            return (
                f"raw->step3 改善 {value:.2f}%（{raw:.4f} m -> {step3:.4f} m）。"
            )
        return f"改善比例为 {value:.2f}%。"

    if metric == "ate_rmse_improve_step2_to_step3_pct":
        step2 = _as_float(block, "ate_rmse_step2_m")
        step3 = _as_float(block, "ate_rmse_step3_m")
        if not np.isnan(step2) and not np.isnan(step3):
            return (
                f"step2->step3 改善 {value:.2f}%（{step2:.4f} m -> {step3:.4f} m）。"
            )
        return f"改善比例为 {value:.2f}%。"

    if metric in {
        "extrinsic_rotation_error_deg",
        "extrinsic_translation_error_m",
        "world_rotation_error_deg",
        "world_translation_error_m",
    }:
        return f"该值为 {value:.6f}，越接近 0 越好（仅 synthetic 有意义）。"

    if section == "estimated_params" and metric == "offset_s":
        return f"估计的全局时间偏移为 {value:.4f}s，用于将 estimation 时间轴对齐到 GT。"

    return f"当前值为 {value:.6f}，建议结合同组指标综合判断。"


def write_metrics_zh_report(output_dir, metrics_payload):
    output_dir = Path(output_dir)
    report_path = output_dir / "metrics_zh.md"

    lines = []
    lines.append("# 指标解读（中文）")
    lines.append("")
    lines.append("本文件自动生成：对每个英文指标给出中文释义，并结合当前数值做简要分析。")
    lines.append("")

    for section, block in metrics_payload.items():
        if not isinstance(block, dict):
            continue
        has_scalar = any(isinstance(v, (int, float, np.integer, np.floating)) for v in block.values())
        if not has_scalar:
            continue
        lines.append(f"## {section}")
        lines.append("")
        for metric, value in block.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            value_float = float(value)
            value_text = "nan" if np.isnan(value_float) else f"{value_float:.6f}"
            zh_explain = METRIC_ZH_EXPLAIN.get(metric, "该指标暂无预置中文解释。")
            analysis = _analyze_metric_value(section, metric, value_float, block)
            lines.append(f"- `{metric}`")
            lines.append(f"  - 中文解释：{zh_explain}")
            lines.append(f"  - 当前数值：`{value_text}`")
            lines.append(f"  - 数值分析：{analysis}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def find_col(columns, candidates):
    for cand in candidates:
        if cand in columns:
            return cand
    stripped = {c.strip(): c for c in columns}
    for cand in candidates:
        if cand in stripped:
            return stripped[cand]
    return None


def load_csv_numeric_columns(path):
    if pd is not None:
        df = pd.read_csv(path)
        columns = list(df.columns)
        data = {c: np.asarray(df[c].values, dtype=float) for c in columns}
        return columns, data

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns = list(reader.fieldnames)
        data = {c: [] for c in columns}
        for row_id, row in enumerate(reader, start=2):
            for c in columns:
                raw = row.get(c, "")
                if raw is None or str(raw).strip() == "":
                    raise ValueError(f"Missing numeric value at row {row_id}, column '{c}' in {path}")
                try:
                    data[c].append(float(raw))
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value at row {row_id}, column '{c}' in {path}: {raw}"
                    ) from exc
    data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    return columns, data


def load_vicon_csv(path):
    columns, data = load_csv_numeric_columns(path)
    t_col = find_col(columns, ["#timestamp", "timestamp"])
    px_col = find_col(columns, ["p_RS_R_x [m]", "p_x", "px", "tx"])
    py_col = find_col(columns, ["p_RS_R_y [m]", "p_y", "py", "ty"])
    pz_col = find_col(columns, ["p_RS_R_z [m]", "p_z", "pz", "tz"])
    qx_col = find_col(columns, ["q_RS_x []", "q_x", "qx"])
    qy_col = find_col(columns, ["q_RS_y []", "q_y", "qy"])
    qz_col = find_col(columns, ["q_RS_z []", "q_z", "qz"])
    qw_col = find_col(columns, ["q_RS_w []", "q_w", "qw"])

    needed = [t_col, px_col, py_col, pz_col, qx_col, qy_col, qz_col, qw_col]
    if any(c is None for c in needed):
        raise ValueError(f"GT CSV missing required columns: {path}")

    t = normalize_time_to_seconds(data[t_col])
    pos = np.column_stack([data[px_col], data[py_col], data[pz_col]])
    quat = normalize_quat_array(np.column_stack([data[qx_col], data[qy_col], data[qz_col], data[qw_col]]))
    return t, pos, quat


def load_estimation_csv(path):
    columns, data = load_csv_numeric_columns(path)
    t_col = find_col(columns, ["#timestamp", "timestamp", "time", "t"])
    px_col = find_col(columns, ["p_RS_R_x [m]", "p_x", "px", "tx", "x"])
    py_col = find_col(columns, ["p_RS_R_y [m]", "p_y", "py", "ty", "y"])
    pz_col = find_col(columns, ["p_RS_R_z [m]", "p_z", "pz", "tz", "z"])
    qx_col = find_col(columns, ["q_RS_x []", "q_x", "qx"])
    qy_col = find_col(columns, ["q_RS_y []", "q_y", "qy"])
    qz_col = find_col(columns, ["q_RS_z []", "q_z", "qz"])
    qw_col = find_col(columns, ["q_RS_w []", "q_w", "qw"])

    needed = [t_col, px_col, py_col, pz_col, qx_col, qy_col, qz_col, qw_col]
    if any(c is None for c in needed):
        raise ValueError(f"Estimation CSV missing required columns: {path}")

    t = normalize_time_to_seconds(data[t_col])
    pos = np.column_stack([data[px_col], data[py_col], data[pz_col]])
    quat = normalize_quat_array(np.column_stack([data[qx_col], data[qy_col], data[qz_col], data[qw_col]]))
    return t, pos, quat


def load_estimation_tum(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 8:
        raise ValueError("Trajectory file must have at least 8 columns: t tx ty tz qx qy qz qw")

    t = normalize_time_to_seconds(arr[:, 0])
    pos = arr[:, 1:4]
    quat = normalize_quat_array(arr[:, 4:8])
    return t, pos, quat


def load_estimation_trajectory(path, est_format):
    if est_format == "csv":
        return load_estimation_csv(path)
    if est_format == "tum":
        return load_estimation_tum(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_estimation_csv(path)
    return load_estimation_tum(path)


def make_output_dir(script_dir):
    output_root = script_dir / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_stamp
    idx = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_stamp}_{idx:02d}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
