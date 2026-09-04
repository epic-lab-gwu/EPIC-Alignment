from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import pandas as pd
except Exception:
    pd = None

from .math_utils import normalize_quat_array, normalize_time_to_seconds

SUPPORTED_ROS_MSGS = {
    "geometry_msgs/msg/PointStamped",
    "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/msg/PoseWithCovarianceStamped",
    "geometry_msgs/msg/TransformStamped",
    "nav_msgs/msg/Odometry",
    "tf2_msgs/msg/TFMessage",
}


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


COMPACT_METRICS_DROP_KEYS = {"_error_arrays", "_x_axis", "_pair_ids"}


def compact_metrics_payload(metrics_payload):
    def _compact(value):
        if isinstance(value, dict):
            return {
                key: _compact(item)
                for key, item in value.items()
                if key not in COMPACT_METRICS_DROP_KEYS
            }
        if isinstance(value, list):
            return [_compact(item) for item in value]
        return value

    return _compact(metrics_payload)


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


def write_result_bundle(output_dir, metrics_payload, out_zip):
    output_dir = Path(output_dir).resolve()
    out_zip = Path(out_zip).expanduser().resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    payload_builtin = to_builtin(metrics_payload)
    manifest = {
        "format": "epa_result_bundle",
        "version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "entries": [
            "metrics.json",
            "metrics_summary.csv",
            "report_zh.md",
            "report_en.md",
            "manifest.json",
        ],
    }

    metrics_json = json.dumps(payload_builtin, indent=2, ensure_ascii=False)
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)

    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", manifest_json)
        zf.writestr("metrics.json", metrics_json)
        metrics_summary = output_dir / "metrics_summary.csv"
        report_zh = output_dir / "report_zh.md"
        report_en = output_dir / "report_en.md"
        if metrics_summary.exists():
            zf.write(metrics_summary, arcname="metrics_summary.csv")
        if report_zh.exists():
            zf.write(report_zh, arcname="report_zh.md")
        if report_en.exists():
            zf.write(report_en, arcname="report_en.md")
    return out_zip


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
    "ate_rmse_step3_m": "完成最终世界系对齐后的绝对轨迹误差 RMSE（米）。",
    "ate_p95_raw_m": "raw 绝对轨迹误差 95 分位（米）。",
    "ate_p95_step2_m": "step2 绝对轨迹误差 95 分位（米）。",
    "ate_p95_step3_m": "最终对齐后绝对轨迹误差 95 分位（米）。",
    "ate_rmse_improve_raw_to_step3_pct": "从 raw 到最终对齐结果的 RMSE 改善百分比。",
    "ate_rmse_improve_step2_to_step3_pct": "从 step2 到最终对齐结果的 RMSE 改善百分比。",
    "step3_rmse_selected_m": "最终世界系对齐后的拟合 RMSE（米）。",
    "alert_level_code": "异常提示等级编码：0=ok，1=warning，2=critical。",
    "alert_count": "触发的异常提示条目数量。",
    "quality_label_code": "对齐质量标签编码：2=good_align，1=partial_align，0=poor_align。",
    "rigid_alignability_code": "刚体可对齐性标签编码：1=rigidly_alignable，0=not_rigidly_alignable。",
    "step3_rmse_m": "用于质量门控的最终全局 RMSE（米）。",
    "raw_to_step3_improve_pct": "用于质量门控的 raw 到 step3 RMSE 改善百分比。",
    "segment_count": "质量评估中使用的分段数量。",
    "segment_rmse_mean_m": "分段 RMSE 均值（米）。",
    "segment_rmse_std_m": "分段 RMSE 标准差（米）。",
    "segment_rmse_cv": "分段 RMSE 变异系数（std/mean），越小越稳定。",
    "heading_median_deg": "局部运动方向夹角中位数（度），越小越一致。",
    "heading_p90_deg": "局部运动方向夹角 90 分位（度），越小越一致。",
    "path_length_ratio_sym": "Step2 轨迹与 GT 轨迹总路程的对称比值（>=1，越接近 1 越好）。",
    "bbox_diag_ratio_sym": "Step2 轨迹与 GT 轨迹包围盒对角线的对称比值（>=1，越接近 1 越好）。",
    "segment_local_se3_rmse_median_m": "按时间分段、每段独立 SE3 最优对齐后的 RMSE 中位数（米）。",
    "segment_global_se3_rmse_median_m": "按时间分段、使用全局 SE3 对齐后的 RMSE 中位数（米）。",
    "segment_global_local_rmse_ratio": "分段全局/局部 RMSE 比值中位数，越接近 1 表示越符合单一刚体假设。",
    "sim3_rmse_m": "同一配对点集下 Sim3（含尺度）对齐 RMSE（米），仅用于诊断。",
    "sim3_scale": "Sim3 估计尺度因子，越接近 1 越符合纯刚体假设。",
    "sim3_gain_ratio": "Sim3 相对 SE3 的 RMSE 改善比例，过大说明可能存在尺度问题。",
    "rigid_check_fail_count": "刚体可对齐性规则触发失败项数量。",
    "piecewise_segment_count": "用于分段诊断的有效时间段数量。",
    "piecewise_global_rmse_median_m": "分段上使用单一全局 SE3 时的 RMSE 中位数（米）。",
    "piecewise_local_rmse_median_m": "分段上每段独立 SE3 最优拟合的 RMSE 中位数（米）。",
    "piecewise_global_local_ratio_median": "分段全局/局部 RMSE 比值中位数，越接近 1 越说明全局刚体假设成立。",
    "piecewise_gap_median_m": "分段全局RMSE-局部RMSE 的中位差值（米）。",
    "piecewise_gap_p90_m": "分段全局RMSE-局部RMSE 的 90 分位差值（米）。",
    "piecewise_gap_max_m": "分段全局RMSE-局部RMSE 的最大差值（米）。",
    "piecewise_peak_recovery_pct": "从最差分段到后续最佳分段的误差恢复百分比（仅诊断）。",
    "piecewise_early_late_delta_m": "后四分之一时段与前四分之一时段全局分段RMSE均值差（米）。",
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

    if metric == "quality_label_code":
        if value >= 1.5:
            return "质量标签为 good_align（整体与局部一致性均较好）。"
        if value >= 0.5:
            return "质量标签为 partial_align（整体可对齐但局部仍有残差）。"
        return "质量标签为 poor_align（当前结果不可靠，建议复查数据或参数）。"

    if metric == "alert_level_code":
        if value >= 1.5:
            return "异常提示等级为 critical，建议优先排查数据与时间对齐。"
        if value >= 0.5:
            return "异常提示等级为 warning，建议结合图像与指标复核。"
        return "异常提示等级为 ok，当前结果整体可用。"

    if metric == "rigid_alignability_code":
        if value >= 0.5:
            return "判定为 rigidly_alignable（可由单一刚体变换解释）。"
        return "判定为 not_rigidly_alignable（单一刚体假设可能不成立）。"

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
                    f"最终对齐 RMSE={value:.4f} m（{quality}）；"
                    f"相对 raw 降低 {drop:.4f} m（{ratio:.2f}%）。"
                )
        if metric == "ate_rmse_step2_m" and not np.isnan(step3):
            delta = value - step3
            return f"Step2 RMSE={value:.4f} m；比最终对齐结果高 {delta:.4f} m。"
        if metric == "ate_rmse_raw_m" and not np.isnan(step3):
            delta = value - step3
            return f"Raw RMSE={value:.4f} m；比最终对齐结果高 {delta:.4f} m。"
        if metric == "ate_p95_step3_m":
            return f"最终对齐 P95={value:.4f} m，表示 95% 时刻误差不超过该值。"
        if metric == "ate_p95_step2_m" and not np.isnan(step2):
            return f"Step2 P95={value:.4f} m。"
        if metric == "ate_p95_raw_m" and not np.isnan(raw):
            return f"Raw P95={value:.4f} m。"
        return f"该值为 {value:.6f}，越小越好。"

    if metric in {"segment_rmse_cv", "heading_p90_deg", "heading_median_deg"}:
        if metric == "segment_rmse_cv":
            if value <= 0.35:
                return f"分段 RMSE CV={value:.3f}，局部误差分布较稳定。"
            if value <= 0.8:
                return f"分段 RMSE CV={value:.3f}，存在一定局部波动。"
            return f"分段 RMSE CV={value:.3f}，局部误差波动较大。"
        if metric == "heading_p90_deg":
            if value <= 35:
                return f"方向夹角 p90={value:.2f}°，局部方向一致性较好。"
            if value <= 70:
                return f"方向夹角 p90={value:.2f}°，局部方向存在偏差。"
            return f"方向夹角 p90={value:.2f}°，局部方向偏差明显。"
        if value <= 15:
            return f"方向夹角中位数={value:.2f}°，整体方向较一致。"
        return f"方向夹角中位数={value:.2f}°，存在可见方向偏差。"

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


def _fmt_report_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _collect_plot_images(output_dir: Path) -> list[Path]:
    plots_dir = output_dir / "plots"
    if not plots_dir.exists():
        return []
    images = []
    for p in sorted(plots_dir.iterdir(), key=lambda x: x.name):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            images.append(p)
    return images


def _filter_report_images(images: list[Path], *, include_debug: bool = False) -> list[Path]:
    priority = {
        "step3_alignment_map.png": 0,
        "rpe_time_1s_translation_part_series_p95.png": 1,
        "rpe_time_1s_translation_part_series_p99.png": 2,
        "rpe_time_1s_translation_part_stats_core.png": 3,
        "rpe_time_1s_translation_part_box_p95.png": 4,
        "rpe_time_1s_translation_part_box_p99.png": 5,
        "rpe_time_1s_translation_part_series.png": 6,
        "rpe_time_1s_translation_part_stats.png": 7,
        "rpe_time_1s_translation_part_box.png": 8,
        "ape_translation_part_series_p95.png": 9,
        "ape_translation_part_series_p99.png": 10,
        "ape_translation_part_stats_core.png": 11,
        "ape_translation_part_hist_p95.png": 12,
        "ape_translation_part_hist_p99.png": 13,
        "ape_translation_part_box_p95.png": 14,
        "ape_translation_part_box_p99.png": 15,
        "ape_translation_part_violin_p95.png": 16,
        "ape_translation_part_violin_p99.png": 17,
        "rpe_translation_part_series_p95.png": 18,
        "rpe_translation_part_series_p99.png": 19,
        "rpe_translation_part_stats_core.png": 20,
        "rpe_translation_part_hist_p95.png": 21,
        "rpe_translation_part_hist_p99.png": 22,
        "rpe_translation_part_box_p95.png": 23,
        "rpe_translation_part_box_p99.png": 24,
        "rpe_translation_part_violin_p95.png": 25,
        "rpe_translation_part_violin_p99.png": 26,
        "rpe_time_1s_translation_part_raw_p95.png": 51,
        "rpe_time_1s_translation_part_raw_p99.png": 52,
        "rpe_time_1s_translation_part_raw.png": 56,
        "ape_translation_part_raw_p95.png": 59,
        "ape_translation_part_raw_p99.png": 60,
        "rpe_translation_part_raw_p95.png": 68,
        "rpe_translation_part_raw_p99.png": 69,
    }
    debug_only_names = {
        "debug_alignment_pipeline_3d.png",
        "debug_step123_trajectory_alignment_3d.png",
        "alignment_pipeline_trajectory_3d.png",
    }
    report_images = [
        img
        for img in images
        if include_debug or (img.name not in debug_only_names and not img.name.startswith("debug_"))
    ]
    report_images.sort(key=lambda img: (priority.get(img.name, 100), img.name))
    return report_images


def _append_image_gallery(lines: list[str], *, title: str, images: list[Path], empty_text: str) -> None:
    lines.append(title)
    lines.append("")
    if not images:
        lines.append(empty_text)
        lines.append("")
        return
    for img in images:
        rel = f"plots/{img.name}"
        lines.append(f"### {img.name}")
        lines.append("")
        lines.append(f"![{img.name}]({rel})")
        lines.append("")


def _infer_case_name(metadata: dict) -> str:
    gt_path = Path(str(metadata.get("gt_path", "") or "")).expanduser()
    est_path = Path(str(metadata.get("est_path", "") or "")).expanduser()

    gt_name = gt_path.name
    est_name = est_path.name

    if gt_name.endswith("__gt.tum"):
        return gt_name[: -len("__gt.tum")]
    if est_name.endswith("__est.tum"):
        return est_name[: -len("__est.tum")]

    if gt_name:
        return gt_path.stem
    if est_name:
        return est_path.stem
    return "unknown_case"


def _append_report_metrics_zh(lines: list[str], metrics_payload: dict) -> None:
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
            value_text = _fmt_report_value(value_float)
            zh_explain = METRIC_ZH_EXPLAIN.get(metric, "该指标暂无预置中文解释。")
            analysis = _analyze_metric_value(section, metric, value_float, block)
            lines.append(f"- `{metric}`")
            lines.append(f"  - 中文解释：{zh_explain}")
            lines.append(f"  - 当前数值：`{value_text}`")
            lines.append(f"  - 数值分析：{analysis}")
        lines.append("")


def _append_report_metrics_en(lines: list[str], metrics_payload: dict) -> None:
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
            value_text = _fmt_report_value(value_float)
            lines.append(f"- `{metric}`: `{value_text}`")
        lines.append("")


def _append_sim3_alignment_report(lines: list[str], *, metadata: dict, language: str) -> None:
    if not isinstance(metadata, dict):
        return
    info = metadata.get("eval_alignment", {})
    if not isinstance(info, dict) or str(info.get("align_mode", "")).lower() != "sim3":
        return
    scale = _fmt_report_value(float(info.get("sim3_scale", info.get("align_scale", np.nan))))
    reliable = bool(info.get("sim3_reliable", True))
    warning = str(info.get("sim3_warning", "") or "").strip()
    if language == "zh":
        lines.extend(
            [
                "## Sim3 对齐提示",
                "",
                f"- Sim3 scale：`{scale}`",
                f"- 结果可靠：`{str(reliable).lower()}`",
            ]
        )
        if warning:
            lines.append(f"- 警告：{warning}")
    else:
        lines.extend(
            [
                "## Sim3 Alignment",
                "",
                f"- Sim3 scale: `{scale}`",
                f"- Reliable: `{str(reliable).lower()}`",
            ]
        )
        if warning:
            lines.append(f"- Warning: {warning}")
    lines.append("")


def _append_sr_reliability_report(lines: list[str], *, metrics_payload: dict, language: str) -> None:
    success = _nested_dict_value(metrics_payload, ("pose_metrics", "valid_segment", "step3", "success"), {})
    if not isinstance(success, dict):
        return
    raw_dist = _fmt_report_value(float(success.get("raw_success_rate_distance", np.nan)) * 100.0)
    local_dist = _fmt_report_value(float(success.get("local_success_rate_distance", success.get("success_rate_distance", np.nan))) * 100.0)
    complete_dist = _fmt_report_value(float(success.get("complete_success_rate_distance", success.get("success_rate_distance", np.nan))) * 100.0)
    raw_time = _fmt_report_value(float(success.get("raw_success_rate_time", np.nan)) * 100.0)
    local_time = _fmt_report_value(float(success.get("local_success_rate_time", success.get("success_rate_time", np.nan))) * 100.0)
    complete_time = _fmt_report_value(float(success.get("complete_success_rate_time", success.get("success_rate_time", np.nan))) * 100.0)
    status = str(success.get("sr_reliability_status", "ok") or "ok")
    explanation = str(success.get("sr_warning_explanation", "") or "No SR reliability issue was detected.")
    hard = success.get("sr_reliability_hard_reasons", [])
    soft = success.get("sr_reliability_soft_reasons", [])
    hard_text = ", ".join(str(item) for item in hard) if isinstance(hard, list) else str(hard)
    soft_text = ", ".join(str(item) for item in soft) if isinstance(soft, list) else str(soft)
    may_mask = bool(success.get("sim3_may_mask_failure", False))

    if language == "zh":
        lines.extend(
            [
                "## Successful Rate 可靠性",
                "",
                f"- 状态：`{status}`",
                f"- SR distance raw / local：`{raw_dist}%` / `{local_dist}%`",
                f"- SR time raw / local：`{raw_time}%` / `{local_time}%`",
                f"- 完整轨迹 SR distance / time：`{complete_dist}%` / `{complete_time}%`",
                f"- Sim3 是否可能掩盖失败：`{str(may_mask).lower()}`",
                f"- 说明：{explanation}",
            ]
        )
        if hard_text:
            lines.append(f"- Hard reasons：`{hard_text}`")
        if soft_text:
            lines.append(f"- Soft reasons：`{soft_text}`")
    else:
        lines.extend(
            [
                "## Successful Rate Reliability",
                "",
                f"- Status: `{status}`",
                f"- SR distance raw / local: `{raw_dist}%` / `{local_dist}%`",
                f"- SR time raw / local: `{raw_time}%` / `{local_time}%`",
                f"- Complete-reference SR distance / time: `{complete_dist}%` / `{complete_time}%`",
                f"- Sim3 may mask failure: `{str(may_mask).lower()}`",
                f"- Explanation: {explanation}",
            ]
        )
        if hard_text:
            lines.append(f"- Hard reasons: `{hard_text}`")
        if soft_text:
            lines.append(f"- Soft reasons: `{soft_text}`")
    lines.append("")


def _append_orientation_report(lines: list[str], *, metadata: dict, language: str) -> None:
    if not isinstance(metadata, dict) or not bool(metadata.get("orientation_unstable", False)):
        return
    ape = _fmt_report_value(float(metadata.get("orientation_ape_rmse_deg", np.nan)))
    rpe = _fmt_report_value(float(metadata.get("orientation_rpe_rmse_deg", np.nan)))
    rpe_time = _fmt_report_value(float(metadata.get("orientation_rpe_time_1s_rmse_deg", np.nan)))
    warning = str(metadata.get("orientation_warning", "") or "").strip()
    if language == "zh":
        lines.extend(
            [
                "## Orientation Warning",
                "",
                "- 状态：`orientation_unstable`",
                f"- APE rotation RMSE：`{ape}` deg",
                f"- RPE rotation RMSE：`{rpe}` deg",
                f"- RPE 1s rotation RMSE：`{rpe_time}` deg",
            ]
        )
        if warning:
            lines.append(f"- 说明：{warning}")
    else:
        lines.extend(
            [
                "## Orientation Warning",
                "",
                "- Status: `orientation_unstable`",
                f"- APE rotation RMSE: `{ape}` deg",
                f"- RPE rotation RMSE: `{rpe}` deg",
                f"- RPE 1s rotation RMSE: `{rpe_time}` deg",
            ]
        )
        if warning:
            lines.append(f"- Note: {warning}")
        lines.append("")


def _nested_dict_value(block: dict, path: tuple[str, ...], default=None):
    cur = block
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _append_time_rpe_report(lines: list[str], *, metrics_payload: dict, language: str) -> None:
    trans = _nested_dict_value(
        metrics_payload,
        ("pose_metrics", "rpe_time_1s", "step3", "translation_part"),
        {},
    )
    rot = _nested_dict_value(
        metrics_payload,
        ("pose_metrics", "rpe_time_1s", "step3", "rotation_angle_deg"),
        {},
    )
    pair_count = _nested_dict_value(metrics_payload, ("pose_metrics", "rpe_time_1s", "step3", "pair_count"), np.nan)
    success = _nested_dict_value(metrics_payload, ("pose_metrics", "valid_segment", "step3", "success"), {})
    if not isinstance(trans, dict) and not isinstance(rot, dict):
        return
    trans_rmse = _fmt_report_value(float(trans.get("rmse", np.nan))) if isinstance(trans, dict) else "nan"
    trans_p95 = _fmt_report_value(float(trans.get("p95", np.nan))) if isinstance(trans, dict) else "nan"
    trans_max = _fmt_report_value(float(trans.get("max", np.nan))) if isinstance(trans, dict) else "nan"
    rot_rmse = _fmt_report_value(float(rot.get("rmse", np.nan))) if isinstance(rot, dict) else "nan"
    drift_threshold = (
        _fmt_report_value(float(success.get("drift_rpe_1s_m", np.nan))) if isinstance(success, dict) else "nan"
    )
    pair_text = str(int(pair_count)) if np.isfinite(float(pair_count)) else "nan"
    if language == "zh":
        lines.extend(
            [
                "## 1-second RPE（局部跳变）",
                "",
                f"- Final aligned translation RMSE：`{trans_rmse}` m",
                f"- Final aligned translation p95 / max：`{trans_p95}` / `{trans_max}` m",
                f"- Final aligned rotation RMSE：`{rot_rmse}` deg",
                f"- Pair count：`{pair_text}`",
                f"- Local drift threshold：`{drift_threshold}` m",
                "- 图：[`series p95`](plots/rpe_time_1s_translation_part_series_p95.png) / [`stats core`](plots/rpe_time_1s_translation_part_stats_core.png) / [`box p95`](plots/rpe_time_1s_translation_part_box_p95.png)",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 1-second RPE (Local Jumps)",
                "",
                f"- Final aligned translation RMSE: `{trans_rmse}` m",
                f"- Final aligned translation p95 / max: `{trans_p95}` / `{trans_max}` m",
                f"- Final aligned rotation RMSE: `{rot_rmse}` deg",
                f"- Pair count: `{pair_text}`",
                f"- Local drift threshold: `{drift_threshold}` m",
                "- Figures: [`series p95`](plots/rpe_time_1s_translation_part_series_p95.png) / [`stats core`](plots/rpe_time_1s_translation_part_stats_core.png) / [`box p95`](plots/rpe_time_1s_translation_part_box_p95.png)",
                "",
            ]
        )


def _append_pose_state_export_report(lines: list[str], *, metadata: dict, language: str) -> None:
    csv_path = str(metadata.get("pose_state_csv", "") or "").strip()
    if not csv_path:
        return
    if language == "zh":
        lines.extend(
            [
                "## Pose State CSV",
                "",
                "- [pose_states.csv](pose_states.csv)",
                "- 每个 timestamp 输出 GT/raw/step2/step3 的 position、orientation、linear velocity 和 angular velocity。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Pose State CSV",
                "",
                "- [pose_states.csv](pose_states.csv)",
                "- Per timestamp export for GT/raw/step2/step3 position, orientation, linear velocity, and angular velocity.",
                "",
            ]
        )


def _append_case_diagnostics_report(lines: list[str], *, metrics_payload: dict, language: str) -> None:
    diag = metrics_payload.get("case_diagnostics", {}) if isinstance(metrics_payload, dict) else {}
    if not isinstance(diag, dict):
        return
    primary = str(diag.get("diagnosis_primary", "") or "").strip()
    summary = str(diag.get("diagnosis_summary", "") or "").strip()
    tags = diag.get("diagnosis_tags", [])
    reasons = diag.get("diagnosis_reasons", {})
    if not primary and not summary and not tags:
        return
    tag_text = ", ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
    if language == "zh":
        lines.extend(["## Case Diagnostics", "", f"- Primary：`{primary or 'none'}`", f"- Tags：`{tag_text or 'none'}`"])
        if isinstance(reasons, dict) and reasons:
            for tag, reason in reasons.items():
                lines.append(f"- `{tag}`：{reason}")
        lines.append("")
    else:
        lines.extend(["## Case Diagnostics", "", f"- Primary: `{primary or 'none'}`", f"- Tags: `{tag_text or 'none'}`"])
        if isinstance(reasons, dict) and reasons:
            for tag, reason in reasons.items():
                lines.append(f"- `{tag}`: {reason}")
        lines.append("")


def write_run_reports(output_dir, metrics_payload):
    output_dir = Path(output_dir)
    metadata = metrics_payload.get("metadata", {}) if isinstance(metrics_payload, dict) else {}
    images = _collect_plot_images(output_dir)
    include_debug = bool(metadata.get("debug", False)) if isinstance(metadata, dict) else False
    report_images = _filter_report_images(images, include_debug=include_debug)
    case_name = _infer_case_name(metadata if isinstance(metadata, dict) else {})

    zh_lines = [
        f"# EPA 运行报告（中文）: {case_name}",
        "",
        "本文件自动生成，汇总本次运行的关键指标与全部图片输出。",
        "",
        f"- 时间：`{metadata.get('timestamp', '')}`",
        f"- GT：`{metadata.get('gt_path', '')}`",
        f"- EST：`{metadata.get('est_path', '')}`",
        "",
    ]
    if bool(metadata.get("step1_failed", False)):
        zh_lines.extend(
            [
                "## 时间对齐失败",
                "",
                "- 状态：`step1_failed`",
                f"- 原因：{metadata.get('step1_failure_reason', 'unknown')}",
                "",
            ]
        )
    elif bool(metadata.get("step1_forced_candidate", False)):
        zh_lines.extend(
            [
                "## 时间对齐回退",
                "",
                "- 状态：`step1_forced_candidate`",
                f"- 原因：{metadata.get('step1_force_reason', 'unknown')}",
                "",
            ]
        )
    alert_level = str(metadata.get("user_alert_level", "") or "").strip()
    alert_msg = str(metadata.get("user_alert_message", metadata.get("user_alert_message_zh", "")) or "").strip()
    alert_reasons = str(metadata.get("user_alert_reasons", metadata.get("user_alert_reasons_zh", "")) or "").strip()
    if alert_level or alert_msg or alert_reasons:
        zh_lines.extend(
            [
                "## 异常提示",
                "",
                f"- 等级：`{alert_level or 'ok'}`",
                f"- 提示：{alert_msg or '无。'}",
            ]
        )
        if alert_reasons:
            zh_lines.append(f"- 原因：{alert_reasons}")
        zh_lines.append("")
    interactive_report = str(
        metadata.get("interactive_report", metadata.get("interactive_html", "")) or ""
    ).strip()
    if interactive_report:
        zh_lines.extend(["## 交互式报告", "", "- [interactive_report.html](interactive_report.html)", ""])
    _append_sim3_alignment_report(zh_lines, metadata=metadata, language="zh")
    _append_sr_reliability_report(zh_lines, metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {}, language="zh")
    _append_orientation_report(zh_lines, metadata=metadata, language="zh")
    _append_case_diagnostics_report(zh_lines, metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {}, language="zh")
    _append_time_rpe_report(zh_lines, metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {}, language="zh")
    _append_pose_state_export_report(zh_lines, metadata=metadata, language="zh")
    _append_report_metrics_zh(zh_lines, metrics_payload if isinstance(metrics_payload, dict) else {})
    _append_image_gallery(
        zh_lines,
        title="## 图片总览",
        images=report_images,
        empty_text="- 无图片输出。",
    )

    en_lines = [
        f"# EPA Run Report (English): {case_name}",
        "",
        "Auto-generated report with key metrics and all generated figures.",
        "",
        f"- Time: `{metadata.get('timestamp', '')}`",
        f"- GT: `{metadata.get('gt_path', '')}`",
        f"- EST: `{metadata.get('est_path', '')}`",
        "",
    ]
    if bool(metadata.get("step1_failed", False)):
        en_lines.extend(
            [
                "## Time Alignment Failure",
                "",
                "- Status: `step1_failed`",
                f"- Reason: {metadata.get('step1_failure_reason', 'unknown')}",
                "",
            ]
        )
    elif bool(metadata.get("step1_forced_candidate", False)):
        en_lines.extend(
            [
                "## Time Alignment Fallback",
                "",
                "- Status: `step1_forced_candidate`",
                f"- Reason: {metadata.get('step1_force_reason', 'unknown')}",
                "",
            ]
        )
    if alert_level or alert_msg or alert_reasons:
        en_lines.extend(
            [
                "## Alert",
                "",
                f"- Level: `{alert_level or 'ok'}`",
            ]
        )
        if alert_msg:
            en_lines.append(f"- Message: {alert_msg}")
        if alert_reasons:
            en_lines.append(f"- Reasons: {alert_reasons}")
        en_lines.append("")
    if interactive_report:
        en_lines.extend(["## Interactive Report", "", "- [interactive_report.html](interactive_report.html)", ""])
    _append_sim3_alignment_report(en_lines, metadata=metadata, language="en")
    _append_sr_reliability_report(en_lines, metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {}, language="en")
    _append_orientation_report(en_lines, metadata=metadata, language="en")
    _append_case_diagnostics_report(en_lines, metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {}, language="en")
    _append_time_rpe_report(en_lines, metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {}, language="en")
    _append_pose_state_export_report(en_lines, metadata=metadata, language="en")
    _append_report_metrics_en(en_lines, metrics_payload if isinstance(metrics_payload, dict) else {})
    _append_image_gallery(
        en_lines,
        title="## Figure Gallery",
        images=report_images,
        empty_text="- No figures were generated.",
    )

    report_zh_path = output_dir / "report_zh.md"
    report_en_path = output_dir / "report_en.md"
    report_zh_path.write_text("\n".join(zh_lines), encoding="utf-8")
    report_en_path.write_text("\n".join(en_lines), encoding="utf-8")
    return report_zh_path, report_en_path


def write_metrics_zh_report(output_dir, metrics_payload):
    # Backward-compatible wrapper; report_zh.md replaces metrics_zh.md.
    write_run_reports(output_dir, metrics_payload)


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
    path = Path(path)
    csv_text = _read_csv_text_without_comments(path)
    if pd is not None:
        df = pd.read_csv(io.StringIO(csv_text))
        columns = list(df.columns)
        data = {c: np.asarray(df[c].values, dtype=float) for c in columns}
        return columns, data

    reader = csv.DictReader(io.StringIO(csv_text))
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


def _read_csv_text_without_comments(path):
    lines = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for raw in f:
            stripped = raw.lstrip()
            if stripped.startswith("#") and not stripped.lower().startswith("#timestamp"):
                continue
            lines.append(raw)
    if not lines:
        raise ValueError(f"CSV has no header: {path}")
    return "".join(lines)


def _parse_numeric_text_row(line, path, line_no, *, allow_header=False):
    parts = line.replace(",", " ").split()
    if not parts:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        if allow_header:
            return None
        raise ValueError(f"Non-numeric value in trajectory text file {path} at line {line_no}: {line}") from exc


def load_text_numeric_table(path, *, min_cols=1, allow_header=True):
    path = Path(path)
    rows = []
    width = None
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values = _parse_numeric_text_row(line, path, line_no, allow_header=allow_header and not rows)
            if values is None:
                continue
            if len(values) < min_cols:
                raise ValueError(f"Trajectory file must have at least {min_cols} columns: {path}")
            row = values[:min_cols]
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(
                    f"Inconsistent column count in trajectory text file {path} at line {line_no}: "
                    f"expected {width}, got {len(row)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No numeric rows found in trajectory text file: {path}")
    return np.asarray(rows, dtype=float)


def load_vicon_csv(path):
    columns, data = load_csv_numeric_columns(path)
    return _trajectory_from_csv_columns(columns, data, path, label="GT CSV")


def _trajectory_from_csv_columns(columns, data, path, label):
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
        raise ValueError(f"{label} missing required columns: {path}")

    t = data[t_col]
    pos = np.column_stack([data[px_col], data[py_col], data[pz_col]])
    quat = normalize_quat_array(np.column_stack([data[qx_col], data[qy_col], data[qz_col], data[qw_col]]))
    timestamp_unit = "ns" if str(t_col).strip() == "#timestamp" else "s"
    return _sanitize_timed_trajectory(t, pos, quat, path, timestamp_unit=timestamp_unit)


def load_estimation_csv(path):
    columns, data = load_csv_numeric_columns(path)
    return _trajectory_from_csv_columns(columns, data, path, label="Estimation CSV")


def _sanitize_timed_trajectory(t, pos, quat, path, *, timestamp_unit="auto"):
    t = np.asarray(t, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float)
    quat = normalize_quat_array(np.asarray(quat, dtype=float))
    if t.size != pos.shape[0] or t.size != quat.shape[0]:
        raise ValueError(f"Trajectory arrays have inconsistent lengths: {path}")

    finite = np.isfinite(t) & np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1)
    q_norm = np.linalg.norm(quat, axis=1)
    finite &= q_norm > 1e-12
    t = t[finite]
    pos = pos[finite]
    quat = normalize_quat_array(quat[finite])
    if t.size < 2:
        raise ValueError(f"Trajectory has fewer than 2 valid rows: {path}")

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    pos = pos[order]
    quat = quat[order]
    keep = np.ones(t.shape[0], dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    t = t[keep]
    pos = pos[keep]
    quat = quat[keep]
    if t.size < 2:
        raise ValueError(f"Trajectory has fewer than 2 unique timestamps: {path}")
    t = normalize_time_to_seconds(t, zero_start=False, unit=timestamp_unit)
    return t, pos, quat


def load_estimation_tum(path):
    arr = load_text_numeric_table(path, min_cols=8)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 8:
        raise ValueError("Trajectory file must have at least 8 columns: t tx ty tz qx qy qz qw")

    t = arr[:, 0]
    pos = arr[:, 1:4]
    quat = normalize_quat_array(arr[:, 4:8])
    # TUM timestamps are defined in seconds, including low-rate trajectories.
    return _sanitize_timed_trajectory(t, pos, quat, path, timestamp_unit="s")


def load_estimation_kitti(path):
    arr = load_text_numeric_table(path, min_cols=12)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 12:
        raise ValueError("KITTI pose file must have at least 12 columns per row.")

    n = arr.shape[0]
    mats = np.repeat(np.eye(3)[None, :, :], n, axis=0)
    mats[:, 0, 0] = arr[:, 0]
    mats[:, 0, 1] = arr[:, 1]
    mats[:, 0, 2] = arr[:, 2]
    mats[:, 1, 0] = arr[:, 4]
    mats[:, 1, 1] = arr[:, 5]
    mats[:, 1, 2] = arr[:, 6]
    mats[:, 2, 0] = arr[:, 8]
    mats[:, 2, 1] = arr[:, 9]
    mats[:, 2, 2] = arr[:, 10]

    pos = np.column_stack([arr[:, 3], arr[:, 7], arr[:, 11]])
    quat = normalize_quat_array(R.from_matrix(mats).as_quat())
    t = normalize_time_to_seconds(np.arange(n, dtype=float), unit="s")
    return t, pos, quat


def _normalize_msg_type(msgtype):
    msgtype = str(msgtype)
    if "/msg/" in msgtype:
        return msgtype
    parts = msgtype.split("/")
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return msgtype


def _stamp_to_sec(stamp):
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    return float(sec) + float(nsec) * 1e-9


def _extract_msg_stamp_sec(msg, fallback_stamp_ns):
    header = getattr(msg, "header", None)
    if header is not None and hasattr(header, "stamp"):
        return _stamp_to_sec(header.stamp)
    return float(fallback_stamp_ns) * 1e-9


def _extract_xyz_quat(msg, msgtype):
    canonical = _normalize_msg_type(msgtype)
    if canonical == "geometry_msgs/msg/TransformStamped":
        tr = msg.transform
        xyz = [tr.translation.x, tr.translation.y, tr.translation.z]
        quat = [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]
        return xyz, quat

    if canonical == "geometry_msgs/msg/PointStamped":
        pt = msg.point
        xyz = [pt.x, pt.y, pt.z]
        return xyz, [0.0, 0.0, 0.0, 1.0]

    pose = msg.pose
    while not hasattr(pose, "position") and hasattr(pose, "pose"):
        pose = pose.pose
    xyz = [pose.position.x, pose.position.y, pose.position.z]
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return xyz, quat


def _extract_xyz_quat_from_transform(transform_stamped):
    tr = transform_stamped.transform
    xyz = [tr.translation.x, tr.translation.y, tr.translation.z]
    quat = [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]
    return xyz, quat


def _parse_tf_topic(topic):
    topic = str(topic)
    if ":" not in topic:
        return topic, None, None
    base_topic, tf_id = topic.split(":", 1)
    if "." not in tf_id:
        raise ValueError(
            "TF topic id must use '/tf:parent.child' format, "
            f"got: {topic}"
        )
    parent, child = tf_id.split(".", 1)
    if not base_topic or not parent or not child:
        raise ValueError(
            "TF topic id must use '/tf:parent.child' format, "
            f"got: {topic}"
        )
    return base_topic, parent, child


def _infer_text_trajectory_format(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values = _parse_numeric_text_row(line, path, line_no, allow_header=True)
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            if values.size >= 12:
                mat = np.array(
                    [
                        [values[0], values[1], values[2]],
                        [values[4], values[5], values[6]],
                        [values[8], values[9], values[10]],
                    ],
                    dtype=float,
                )
                det = float(np.linalg.det(mat))
                if det > 0.5 and np.allclose(mat @ mat.T, np.eye(3), atol=1e-2):
                    return "kitti"
            if values.size >= 8:
                return "tum"
            break
    raise ValueError(f"Cannot infer trajectory format from text file: {path}")


def _load_auto_text_trajectory(path):
    inferred = _infer_text_trajectory_format(path)
    if inferred == "kitti":
        return load_estimation_kitti(path)
    return load_estimation_tum(path)


def _infer_bag_kind(path):
    path = Path(path)
    if path.is_dir():
        return "bag2"
    suffix = path.suffix.lower()
    if suffix == ".bag":
        return "bag"
    if suffix == ".mcap":
        return "mcap"
    raise ValueError(
        f"Cannot infer bag type from path: {path}. "
        "Use --est-format bag|bag2|mcap and provide --est-topic."
    )


def load_bag_trajectory(path, topic, bag_format="auto"):
    if topic is None or str(topic).strip() == "":
        raise ValueError("Bag trajectory loading requires a non-empty topic (e.g. --est-topic /pose).")
    bag_topic, tf_parent, tf_child = _parse_tf_topic(topic)
    use_tf_selector = tf_parent is not None and tf_child is not None

    try:
        from rosbags.rosbag1 import Reader as Rosbag1Reader
        from rosbags.rosbag2 import Reader as Rosbag2Reader
        from rosbags.typesys import Stores, get_typestore
    except Exception as exc:
        raise ImportError("Bag input requires rosbags. Install with: pip install 'epica[ros]'") from exc

    bag_kind = _infer_bag_kind(path) if bag_format == "auto" else bag_format
    if bag_kind not in {"bag", "bag2", "mcap"}:
        raise ValueError(f"Unsupported bag format: {bag_kind}")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bag path not found: {path}")

    reader = Rosbag1Reader(str(path)) if bag_kind == "bag" else Rosbag2Reader(str(path))
    stamps = []
    xyz = []
    quat = []
    try:
        reader.open()
        connections = [c for c in reader.connections if c.topic == bag_topic]
        if not connections:
            available = sorted({c.topic for c in reader.connections})
            hint = ", ".join(available[:10]) + (" ..." if len(available) > 10 else "")
            raise ValueError(f"Topic not found in bag: {bag_topic}. Available topics: {hint}")

        is_ros1 = isinstance(reader, Rosbag1Reader)
        typestore = get_typestore(Stores.ROS1_NOETIC if is_ros1 else Stores.LATEST)
        for connection, stamp_ns, raw in reader.messages(connections=connections):  # type: ignore
            msgtype = _normalize_msg_type(connection.msgtype)
            if msgtype not in SUPPORTED_ROS_MSGS:
                raise ValueError(f"Unsupported bag message type: {connection.msgtype}")
            msg = (
                typestore.deserialize_ros1(raw, connection.msgtype)
                if is_ros1
                else typestore.deserialize_cdr(raw, connection.msgtype)
            )
            if use_tf_selector:
                if msgtype != "tf2_msgs/msg/TFMessage":
                    raise ValueError(
                        f"Topic '{bag_topic}' is not a TF message stream, got {connection.msgtype}"
                    )
                for tf in msg.transforms:
                    if tf.header.frame_id == tf_parent and tf.child_frame_id == tf_child:
                        stamps.append(_stamp_to_sec(tf.header.stamp))
                        p_xyz, q_xyzw = _extract_xyz_quat_from_transform(tf)
                        xyz.append(p_xyz)
                        quat.append(q_xyzw)
            else:
                t_sec = _extract_msg_stamp_sec(msg, stamp_ns)
                p_xyz, q_xyzw = _extract_xyz_quat(msg, connection.msgtype)
                stamps.append(t_sec)
                xyz.append(p_xyz)
                quat.append(q_xyzw)
    finally:
        reader.close()

    if not stamps:
        if use_tf_selector:
            raise ValueError(
                f"No TF trajectory found for {bag_topic}:{tf_parent}.{tf_child}"
            )
        raise ValueError(f"No trajectory messages found for topic: {bag_topic}")

    t = normalize_time_to_seconds(np.asarray(stamps, dtype=float), zero_start=False, unit="s")
    pos = np.asarray(xyz, dtype=float)
    q = normalize_quat_array(np.asarray(quat, dtype=float))
    return t, pos, q


def load_reference_trajectory(path, gt_format="csv", gt_topic=""):
    gt_format = str(gt_format).lower()
    path = Path(path)

    if gt_format == "csv":
        return load_vicon_csv(path)
    if gt_format == "euroc":
        return load_estimation_csv(path)
    if gt_format == "tum":
        return load_estimation_tum(path)
    if gt_format == "kitti":
        return load_estimation_kitti(path)
    if gt_format in {"bag", "bag2", "mcap"}:
        return load_bag_trajectory(path, gt_topic, bag_format=gt_format)
    if gt_format == "auto":
        if path.is_dir() or path.suffix.lower() in {".bag", ".mcap"}:
            return load_bag_trajectory(path, gt_topic, bag_format="auto")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                return load_vicon_csv(path)
            except Exception:
                return _load_auto_text_trajectory(path)
        return _load_auto_text_trajectory(path)
    raise ValueError(f"Unsupported gt format: {gt_format}")


def load_estimation_trajectory(path, est_format, est_topic=""):
    est_format = str(est_format).lower()
    path = Path(path)

    if est_format == "csv":
        return load_estimation_csv(path)
    if est_format == "euroc":
        return load_estimation_csv(path)
    if est_format == "tum":
        return load_estimation_tum(path)
    if est_format == "kitti":
        return load_estimation_kitti(path)
    if est_format in {"bag", "bag2", "mcap"}:
        return load_bag_trajectory(path, est_topic, bag_format=est_format)

    if est_format == "auto":
        if path.is_dir() or path.suffix.lower() in {".bag", ".mcap"}:
            return load_bag_trajectory(path, est_topic, bag_format="auto")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                return load_estimation_csv(path)
            except Exception:
                return _load_auto_text_trajectory(path)
        return _load_auto_text_trajectory(path)

    raise ValueError(f"Unsupported estimation format: {est_format}")


def _safe_run_label(label) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-")


def make_output_dir(script_dir, output_root=None, run_label=""):
    output_root = Path(output_root) if output_root else script_dir / "outputs"
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    safe_label = _safe_run_label(run_label)
    run_name = f"{run_stamp}_{safe_label}" if safe_label else run_stamp
    for idx in range(10000):
        suffix = "" if idx == 0 else f"_{idx:02d}"
        run_dir = output_root / f"{run_name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a unique output directory under: {output_root}")
