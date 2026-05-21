#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from epa.benchmark.benchmark_harness import discover_cases  # noqa: E402


IDEA_ROWS = [
    {
        "idea_id": "robot_arm_handeye_static",
        "domain": "机械臂",
        "scenario": "静态抓取前手眼标定后轨迹一致性验证",
        "signals": "机械臂末端位姿 + 相机/marker 位姿",
        "target_dataset": "自采集 (UR/Franka + AprilTag)",
        "priority": "P0",
    },
    {
        "idea_id": "robot_arm_handeye_dynamic",
        "domain": "机械臂",
        "scenario": "动态抓取过程中在线外参漂移监控",
        "signals": "末端位姿 + 视觉里程计",
        "target_dataset": "自采集动态抓取序列",
        "priority": "P0",
    },
    {
        "idea_id": "dual_arm_coordination",
        "domain": "机械臂",
        "scenario": "双臂协作任务中的时空对齐",
        "signals": "双臂关节/末端轨迹",
        "target_dataset": "双臂装配任务日志",
        "priority": "P1",
    },
    {
        "idea_id": "mobile_manipulation",
        "domain": "移动操作",
        "scenario": "底盘+机械臂联合轨迹对齐",
        "signals": "底盘里程计 + 末端位姿",
        "target_dataset": "移动抓取自采集",
        "priority": "P1",
    },
    {
        "idea_id": "drone_vio_indoor",
        "domain": "无人机",
        "scenario": "室内 VIO 与真值轨迹对齐",
        "signals": "IMU + camera + mocap",
        "target_dataset": "EuRoC / UZH-FPV",
        "priority": "P0",
    },
    {
        "idea_id": "drone_vio_outdoor",
        "domain": "无人机",
        "scenario": "室外快速飞行场景对齐鲁棒性",
        "signals": "IMU + camera + GNSS",
        "target_dataset": "UZH-FPV / 自采集",
        "priority": "P1",
    },
    {
        "idea_id": "drone_gps_dropout",
        "domain": "无人机",
        "scenario": "GPS 丢失时短时对齐恢复",
        "signals": "GNSS + VIO",
        "target_dataset": "自采集断链数据",
        "priority": "P1",
    },
    {
        "idea_id": "ground_robot_warehouse",
        "domain": "AMR",
        "scenario": "仓储机器人长走廊重复纹理对齐",
        "signals": "wheel odom + lidar odom + vision",
        "target_dataset": "自采集仓储数据",
        "priority": "P1",
    },
    {
        "idea_id": "ground_robot_loop",
        "domain": "AMR",
        "scenario": "闭环场景下局部漂移对齐评估",
        "signals": "SLAM 轨迹 + GT",
        "target_dataset": "KITTI / TUM",
        "priority": "P1",
    },
    {
        "idea_id": "autonomous_driving_localization",
        "domain": "自动驾驶",
        "scenario": "定位模块与高精地图基准对齐",
        "signals": "GNSS/INS + lidar odom",
        "target_dataset": "KITTI / NCLT",
        "priority": "P2",
    },
    {
        "idea_id": "autonomous_driving_multi_sensor",
        "domain": "自动驾驶",
        "scenario": "多传感器时间偏差统一估计",
        "signals": "camera + lidar + radar + imu",
        "target_dataset": "nuScenes / self-driving logs",
        "priority": "P2",
    },
    {
        "idea_id": "ar_vr_headset_tracking",
        "domain": "AR/VR",
        "scenario": "头显 inside-out 与外部真值对齐",
        "signals": "HMD pose + mocap",
        "target_dataset": "自采集头显数据",
        "priority": "P2",
    },
    {
        "idea_id": "human_motion_capture_fusion",
        "domain": "人体动作",
        "scenario": "IMU suit 与光学 mocap 的轨迹对齐",
        "signals": "body IMU + mocap",
        "target_dataset": "AMASS / HumanEva",
        "priority": "P2",
    },
    {
        "idea_id": "legged_robot_indoor",
        "domain": "足式机器人",
        "scenario": "四足机器人里程计与视觉轨迹对齐",
        "signals": "leg odom + VIO",
        "target_dataset": "自采集四足数据",
        "priority": "P2",
    },
    {
        "idea_id": "legged_robot_rough_terrain",
        "domain": "足式机器人",
        "scenario": "起伏地形下的对齐稳定性",
        "signals": "IMU + LiDAR + 足端接触",
        "target_dataset": "自采集野外数据",
        "priority": "P2",
    },
    {
        "idea_id": "underwater_auv_localization",
        "domain": "水下机器人",
        "scenario": "声呐里程计与惯导对齐",
        "signals": "DVL + IMU + sonar",
        "target_dataset": "AUV 自采集",
        "priority": "P3",
    },
    {
        "idea_id": "surface_vessel_navigation",
        "domain": "水面艇",
        "scenario": "GNSS/INS 与视觉定位的时间对齐",
        "signals": "GNSS + IMU + camera",
        "target_dataset": "USV 自采集",
        "priority": "P3",
    },
    {
        "idea_id": "rail_robot_inspection",
        "domain": "轨道巡检",
        "scenario": "轨道车定位轨迹对齐与漂移检测",
        "signals": "encoder + IMU + vision",
        "target_dataset": "轨道巡检日志",
        "priority": "P3",
    },
    {
        "idea_id": "construction_robot_mapping",
        "domain": "工程机器人",
        "scenario": "施工现场多机建图轨迹对齐",
        "signals": "lidar odom + imu + GNSS",
        "target_dataset": "施工现场自采集",
        "priority": "P3",
    },
    {
        "idea_id": "agri_robot_row_navigation",
        "domain": "农业机器人",
        "scenario": "农田行间导航轨迹一致性",
        "signals": "RTK + vision + wheel odom",
        "target_dataset": "农业场景自采集",
        "priority": "P3",
    },
    {
        "idea_id": "delivery_robot_sidewalk",
        "domain": "配送机器人",
        "scenario": "人行道场景多源轨迹对齐",
        "signals": "wheel odom + camera + GNSS",
        "target_dataset": "配送日志",
        "priority": "P2",
    },
    {
        "idea_id": "forklift_indoor_outdoor",
        "domain": "工业车辆",
        "scenario": "室内外切换定位对齐",
        "signals": "lidar odom + GNSS",
        "target_dataset": "叉车自采集",
        "priority": "P3",
    },
    {
        "idea_id": "camera_network_relocalization",
        "domain": "多相机网络",
        "scenario": "多相机子系统轨迹拼接与对齐",
        "signals": "multi-camera poses",
        "target_dataset": "多相机自采集",
        "priority": "P3",
    },
    {
        "idea_id": "multi_robot_swarm_sync",
        "domain": "多机器人",
        "scenario": "群体机器人时间同步误差评估",
        "signals": "robot_i trajectory streams",
        "target_dataset": "swarm logs",
        "priority": "P2",
    },
    {
        "idea_id": "slam_regression_ci",
        "domain": "算法工程",
        "scenario": "CI 中新旧版本轨迹回归对齐",
        "signals": "run A/B trajectories",
        "target_dataset": "项目 nightly outputs",
        "priority": "P0",
    },
    {
        "idea_id": "sim2real_gap_eval",
        "domain": "仿真迁移",
        "scenario": "仿真轨迹与实机轨迹对齐偏差",
        "signals": "sim trajectory + real trajectory",
        "target_dataset": "Gazebo/Isaac + real logs",
        "priority": "P2",
    },
    {
        "idea_id": "sports_tracking",
        "domain": "运动分析",
        "scenario": "可穿戴与视觉系统轨迹对齐",
        "signals": "IMU wearable + camera tracking",
        "target_dataset": "运动实验数据",
        "priority": "P3",
    },
    {
        "idea_id": "medical_tool_tracking",
        "domain": "医疗机器人",
        "scenario": "手术器械定位系统对齐",
        "signals": "optical tracker + robot kinematics",
        "target_dataset": "医疗实验数据",
        "priority": "P3",
    },
    {
        "idea_id": "mining_robot_underground",
        "domain": "矿山机器人",
        "scenario": "无 GNSS 地下环境轨迹对齐",
        "signals": "lidar odom + imu",
        "target_dataset": "矿山自采集",
        "priority": "P3",
    },
    {
        "idea_id": "rescue_robot_smoke",
        "domain": "救援机器人",
        "scenario": "低可见度场景轨迹稳健性",
        "signals": "thermal camera + lidar + imu",
        "target_dataset": "灾害场景自采集",
        "priority": "P3",
    },
    {
        "idea_id": "space_robot_docking",
        "domain": "航天机器人",
        "scenario": "对接过程相对轨迹对齐",
        "signals": "relative pose stream",
        "target_dataset": "仿真/实验平台",
        "priority": "P4",
    },
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _collect_alignanything_rows(align_root: Path) -> tuple[list[dict], list[dict], dict]:
    cases, unresolved = discover_cases(align_root)
    rows: list[dict] = []
    for case in cases:
        rows.append(
            {
                "use_case_id": case.case_id,
                "source": "alignanything",
                "domain": "VIO/SLAM benchmark",
                "dataset": case.dataset,
                "sequence": case.sequence,
                "method": case.method,
                "status": "ready",
                "gt_path": str(case.gt_path),
                "est_path": str(case.est_path),
                "run_hint": (
                    "epa_bench --cases-root "
                    f"{align_root} --case-pattern ^{case.case_id}$ --limit 1"
                ),
                "notes": "auto-discovered by epa benchmark harness",
            }
        )

    unresolved_rows: list[dict] = []
    for item in unresolved:
        unresolved_rows.append(
            {
                "use_case_id": f"{item['dataset']}_{item['sequence']}_{item['method']}",
                "source": "alignanything",
                "domain": "VIO/SLAM benchmark",
                "dataset": item["dataset"],
                "sequence": item["sequence"],
                "method": item["method"],
                "status": "unresolved",
                "gt_path": "",
                "est_path": str(align_root / "benchmark" / item["est_relpath"]),
                "run_hint": "fix GT mapping first (see unresolved CSV)",
                "notes": "est exists but matching GT not found",
            }
        )

    summary = {
        "ready_cases": len(rows),
        "unresolved_cases": len(unresolved_rows),
        "dataset_counts": dict(Counter([r["dataset"] for r in rows])),
        "method_counts": dict(Counter([r["method"] for r in rows])),
    }
    return rows, unresolved_rows, summary


def _collect_evo_rows(evo_data_root: Path) -> list[dict]:
    rows: list[dict] = []

    def add_pair(case_id: str, dataset: str, gt_name: str, est_name: str, fmt: str = "tum") -> None:
        gt = evo_data_root / gt_name
        est = evo_data_root / est_name
        status = "ready" if gt.exists() and est.exists() else "missing"
        rows.append(
            {
                "use_case_id": case_id,
                "source": "evo_test_data",
                "domain": "Trajectory toolchain sanity",
                "dataset": dataset,
                "sequence": case_id,
                "method": est_name,
                "status": status,
                "gt_path": str(gt) if gt.exists() else "",
                "est_path": str(est) if est.exists() else "",
                "run_hint": (
                    f"epa_ape {fmt} {gt} {est} --align --plot" if status == "ready" else "missing file(s)"
                ),
                "notes": "ported from local evo test data",
            }
        )

    add_pair("freiburg1_xyz_orb_kf", "tum_rgbd", "freiburg1_xyz-groundtruth.txt", "freiburg1_xyz-ORB_kf_mono.txt")
    add_pair("freiburg1_xyz_rgbdslam", "tum_rgbd", "freiburg1_xyz-groundtruth.txt", "freiburg1_xyz-rgbdslam.txt")
    add_pair("freiburg1_xyz_rgbdslam_drift", "tum_rgbd", "freiburg1_xyz-groundtruth.txt", "freiburg1_xyz-rgbdslam_drift.txt")
    add_pair("freiburg1_xyz_rgbdslam_drift_short", "tum_rgbd", "freiburg1_xyz-groundtruth.txt", "freiburg1_xyz-rgbdslam_drift_short.txt")
    add_pair("fr2_desk_orb", "tum_rgbd", "fr2_desk_groundtruth.txt", "fr2_desk_ORB.txt")
    add_pair("fr2_desk_orb_kf", "tum_rgbd", "fr2_desk_groundtruth.txt", "fr2_desk_ORB_kf_mono.txt")
    add_pair("kitti00_orb", "kitti", "KITTI_00_gt.txt", "KITTI_00_ORB.txt", fmt="kitti")
    add_pair("kitti00_sptam", "kitti", "KITTI_00_gt.txt", "KITTI_00_SPTAM.txt", fmt="kitti")
    add_pair("euroc_v102_example", "euroc_csv", "V102_groundtruth.csv", "V102.txt")

    tf_bag = evo_data_root / "tf_example.bag"
    ros_bag = evo_data_root / "ROS_example.bag"
    for bag in [tf_bag, ros_bag]:
        rows.append(
            {
                "use_case_id": f"bag_{bag.stem}",
                "source": "evo_test_data",
                "domain": "ROS bag trajectory extraction",
                "dataset": "ros_bag",
                "sequence": bag.stem,
                "method": "bag",
                "status": "needs_topics",
                "gt_path": "",
                "est_path": str(bag) if bag.exists() else "",
                "run_hint": f"epa_traj bag {bag} /topic_gt /topic_est --plot",
                "notes": "bag exists; set ROS topics before running",
            }
        )

    return rows


def _collect_vicon_room_rows(vicon_root: Path) -> list[dict]:
    rows: list[dict] = []
    for seq_dir in sorted(vicon_root.glob("V*_*/")):
        seq_name = seq_dir.name
        bag = seq_dir / f"{seq_name}.bag"
        gt_csv = seq_dir / "vicon2gt" / f"{seq_name}_vicon2gt_states.csv"
        rows.append(
            {
                "use_case_id": f"vicon_room1_{seq_name}",
                "source": "vicon_room1",
                "domain": "Indoor Vicon-grounded evaluation",
                "dataset": "vicon_room1",
                "sequence": seq_name,
                "method": "rosbag",
                "status": "needs_conversion" if bag.exists() else "missing",
                "gt_path": str(gt_csv) if gt_csv.exists() else "",
                "est_path": str(bag) if bag.exists() else "",
                "run_hint": f"epa_traj bag {bag} /vicon/pose /est/pose --plot",
                "notes": "convert GT/est from bag topics as needed",
            }
        )
    return rows


def _write_idea_markdown(path: Path, ideas: list[dict]) -> None:
    lines = [
        "# EPA Potential Use Cases",
        "",
        "这个文件是跨领域 use case 候选池，优先级用于后续排期。",
        "",
        "| ID | Domain | Scenario | Signals | Candidate Data | Priority |",
        "|---|---|---|---|---|---|",
    ]
    for item in ideas:
        lines.append(
            f"| {item['idea_id']} | {item['domain']} | {item['scenario']} | {item['signals']} | {item['target_dataset']} | {item['priority']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect local and potential EPA use cases into epa_data.")
    parser.add_argument(
        "--cases-root",
        "--alignanything-root",
        dest="alignanything_root",
        default="/home/yifu/epa_data/AlignAnything/AlignAnything",
        help="Path containing benchmark/ and GT/.",
    )
    parser.add_argument(
        "--evo-data-root",
        default="/home/yifu/evo/test/data",
        help="Path to local evo test data.",
    )
    parser.add_argument(
        "--vicon-room-root",
        default="/home/yifu/vicon_room1",
        help="Path to vicon_room1 data root.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/yifu/epa_data/use_cases",
        help="Output directory under epa_data.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    align_rows, unresolved_rows, align_summary = _collect_alignanything_rows(Path(args.alignanything_root).resolve())
    evo_rows = _collect_evo_rows(Path(args.evo_data_root).resolve())
    vicon_rows = _collect_vicon_room_rows(Path(args.vicon_room_root).resolve())

    all_rows = align_rows + evo_rows + vicon_rows
    fieldnames = [
        "use_case_id",
        "source",
        "domain",
        "dataset",
        "sequence",
        "method",
        "status",
        "gt_path",
        "est_path",
        "run_hint",
        "notes",
    ]

    _write_csv(out_dir / "local_use_cases.csv", all_rows, fieldnames)
    _write_csv(out_dir / "alignanything_unresolved.csv", unresolved_rows, fieldnames)
    _write_idea_markdown(out_dir / "potential_use_cases.md", IDEA_ROWS)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = {
        "generated_at": now,
        "out_dir": str(out_dir),
        "sources": {
            "alignanything": align_summary,
            "evo_test_data": {
                "total": len(evo_rows),
                "status_counts": dict(Counter([r["status"] for r in evo_rows])),
            },
            "vicon_room1": {
                "total": len(vicon_rows),
                "status_counts": dict(Counter([r["status"] for r in vicon_rows])),
            },
        },
        "all_local_use_cases": {
            "total": len(all_rows),
            "status_counts": dict(Counter([r["status"] for r in all_rows])),
            "dataset_counts": dict(Counter([r["dataset"] for r in all_rows])),
        },
        "potential_idea_count": len(IDEA_ROWS),
        "files": {
            "local_use_cases_csv": str(out_dir / "local_use_cases.csv"),
            "alignanything_unresolved_csv": str(out_dir / "alignanything_unresolved.csv"),
            "potential_use_cases_md": str(out_dir / "potential_use_cases.md"),
            "summary_json": str(out_dir / "summary.json"),
            "readme_md": str(out_dir / "README.md"),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme_lines = [
        "# EPA Use Cases",
        "",
        f"Generated at: {now}",
        "",
        "## Files",
        "",
        "- `local_use_cases.csv`: 本机可直接使用或快速改造的 use cases",
        "- `alignanything_unresolved.csv`: cases root 中尚未匹配 GT 的条目",
        "- `potential_use_cases.md`: 跨领域候选场景池",
        "- `summary.json`: 统计摘要",
        "",
        "## Quick Check",
        "",
        f"- total local rows: {len(all_rows)}",
        f"- ready rows: {sum(1 for r in all_rows if r['status'] == 'ready')}",
        f"- unresolved alignanything rows: {len(unresolved_rows)}",
        f"- potential ideas: {len(IDEA_ROWS)}",
        "",
        "## Refresh",
        "",
        "```bash",
        "python scripts/collect_use_cases.py",
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
