#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests

POTENTIAL_MD = Path('/home/yifu/epa_data/use_cases/potential_use_cases.md')
OUT_ROOT = Path('/home/yifu/epa_data/potential_datasets')
MAX_AUTO_DOWNLOAD_BYTES = 600 * 1024 * 1024  # 600 MB


@dataclass
class DownloadItem:
    name: str
    url: str
    rel_path: str
    size_bytes: int | None = None
    source_page: str | None = None


def _parse_potential_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('| ') or line.startswith('| ID ') or line.startswith('|---'):
            continue
        cols = [c.strip() for c in line.strip('|').split('|')]
        if len(cols) < 6:
            continue
        rows.append(
            {
                'idea_id': cols[0],
                'domain': cols[1],
                'scenario': cols[2],
                'candidate_data': cols[4],
                'priority': cols[5],
            }
        )
    return rows


def _download(url: str, out_path: Path) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.part')
    with requests.get(url, stream=True, timeout=120) as resp:
        if resp.status_code != 200:
            return False, f'http_{resp.status_code}'
        with tmp.open('wb') as f:
            for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.rename(out_path)
    return True, 'downloaded'


def _existing_alignanything_drone() -> bool:
    root = Path('/home/yifu/epa_data/AlignAnything/AlignAnything')
    return (
        (root / 'benchmark' / 'euroc_mav').exists()
        and (root / 'benchmark' / 'uzh_fpv').exists()
        and (root / 'GT' / 'euroc_mav').exists()
    )


def _existing_alignanything_uzh() -> bool:
    root = Path('/home/yifu/epa_data/AlignAnything/AlignAnything')
    return (root / 'benchmark' / 'uzh_fpv').exists() and any((root / 'GT').glob('uzhfpv*'))


def _build_plan() -> tuple[dict[str, list[DownloadItem]], dict[str, str], dict[str, Callable[[], bool]]]:
    plan: dict[str, list[DownloadItem]] = {
        'robot_arm_handeye_static': [
            DownloadItem(
                name='handeye_README',
                url='https://www.research-collection.ethz.ch/bitstreams/20bd394e-15e3-4da8-9fa2-01c746a200e3/download',
                rel_path='01_robot_arm_handeye_static/eth_handeye/README.pdf',
                size_bytes=150769,
                source_page='https://www.research-collection.ethz.ch/entities/researchdata/6fcaddca-7c66-4c74-9515-318de795ac32',
            ),
            DownloadItem(
                name='handeye_tango_easy',
                url='https://www.research-collection.ethz.ch/bitstreams/3f5db9e9-5dba-40c8-ab45-a4d1379fa8aa/download',
                rel_path='01_robot_arm_handeye_static/eth_handeye/tango_easy.zip',
                size_bytes=1507728,
                source_page='https://www.research-collection.ethz.ch/entities/researchdata/6fcaddca-7c66-4c74-9515-318de795ac32',
            ),
            DownloadItem(
                name='handeye_results',
                url='https://www.research-collection.ethz.ch/bitstreams/56257dff-6aa3-4d51-982c-fd7a5fa645ac/download',
                rel_path='01_robot_arm_handeye_static/eth_handeye/RESULTS.zip',
                size_bytes=186154,
                source_page='https://www.research-collection.ethz.ch/entities/researchdata/6fcaddca-7c66-4c74-9515-318de795ac32',
            ),
        ],
        'robot_arm_handeye_dynamic': [
            DownloadItem(
                name='handeye_robot_arm_sim',
                url='https://www.research-collection.ethz.ch/bitstreams/b9eaf174-83f4-4917-8312-c71e88196d69/download',
                rel_path='02_robot_arm_handeye_dynamic/eth_handeye/robot_arm_sim.zip',
                size_bytes=118678525,
                source_page='https://www.research-collection.ethz.ch/entities/researchdata/6fcaddca-7c66-4c74-9515-318de795ac32',
            ),
        ],
        'dual_arm_coordination': [
            DownloadItem(
                name='handeye_robot_arm_real',
                url='https://www.research-collection.ethz.ch/bitstreams/47a1ef94-f315-4eb8-8dff-9a763d01c2f0/download',
                rel_path='03_dual_arm_coordination/eth_handeye/robot_arm_real.zip',
                size_bytes=2020927539,
                source_page='https://www.research-collection.ethz.ch/entities/researchdata/6fcaddca-7c66-4c74-9515-318de795ac32',
            )
        ],
        'ground_robot_loop': [
            DownloadItem(
                name='kitti_odometry_poses',
                url='https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip',
                rel_path='09_ground_robot_loop/kitti/data_odometry_poses.zip',
                size_bytes=1309620,
                source_page='https://www.cvlibs.net/datasets/kitti/eval_odometry.php',
            ),
            DownloadItem(
                name='tum_freiburg1_gt',
                url='https://raw.githubusercontent.com/MichaelGrupp/evo/master/test/data/freiburg1_xyz-groundtruth.txt',
                rel_path='09_ground_robot_loop/tum_rgbd/freiburg1_xyz-groundtruth.txt',
                source_page='https://vision.in.tum.de/data/datasets/rgbd-dataset',
            ),
            DownloadItem(
                name='tum_freiburg1_orb',
                url='https://raw.githubusercontent.com/MichaelGrupp/evo/master/test/data/freiburg1_xyz-ORB_kf_mono.txt',
                rel_path='09_ground_robot_loop/tum_rgbd/freiburg1_xyz-ORB_kf_mono.txt',
                source_page='https://vision.in.tum.de/data/datasets/rgbd-dataset',
            ),
            DownloadItem(
                name='tum_freiburg1_rgbdslam',
                url='https://raw.githubusercontent.com/MichaelGrupp/evo/master/test/data/freiburg1_xyz-rgbdslam.txt',
                rel_path='09_ground_robot_loop/tum_rgbd/freiburg1_xyz-rgbdslam.txt',
                source_page='https://vision.in.tum.de/data/datasets/rgbd-dataset',
            ),
        ],
        'autonomous_driving_localization': [
            DownloadItem(
                name='nclt_groundtruth_2012-01-08',
                url='https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2012-01-08.csv',
                rel_path='10_autonomous_driving_localization/nclt/groundtruth_2012-01-08.csv',
                size_bytes=115860468,
                source_page='https://robots.engin.umich.edu/nclt/',
            ),
            DownloadItem(
                name='nclt_cov_2012-01-08',
                url='https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/covariance/cov_2012-01-08.csv',
                rel_path='10_autonomous_driving_localization/nclt/cov_2012-01-08.csv',
                size_bytes=15881839,
                source_page='https://robots.engin.umich.edu/nclt/',
            ),
        ],
        'autonomous_driving_multi_sensor': [
            DownloadItem(
                name='nuscenes_mini',
                url='https://www.nuscenes.org/data/v1.0-mini.tgz',
                rel_path='11_autonomous_driving_multi_sensor/nuscenes/v1.0-mini.tgz',
                size_bytes=4167696325,
                source_page='https://www.nuscenes.org/download',
            ),
        ],
    }

    manual_links: dict[str, str] = {
        'mobile_manipulation': 'https://rh20t.github.io/',
        'drone_gps_dropout': 'https://fpv.ifi.uzh.ch/datasets/',
        'ground_robot_warehouse': 'https://www.nuscenes.org/',
        'ar_vr_headset_tracking': 'https://tum-vi.vision.in.tum.de/',
        'human_motion_capture_fusion': 'https://amass.is.tue.mpg.de/',
        'legged_robot_indoor': 'https://leggedrobotics.github.io/legged_gym/',
        'legged_robot_rough_terrain': 'https://www.ipb.uni-bonn.de/datasets/',
        'underwater_auv_localization': 'https://www.lirmm.fr/aqualoc/',
        'surface_vessel_navigation': 'https://zenodo.org/communities/marine-robotics/',
        'rail_robot_inspection': 'https://doi.org/10.1109/ACCESS.2022.3176236',
        'construction_robot_mapping': 'https://www.cs.cmu.edu/~ILIM/3DLabeling/',
        'agri_robot_row_navigation': 'https://www.ipb.uni-bonn.de/data/',
        'delivery_robot_sidewalk': 'https://www.cityscapes-dataset.com/',
        'forklift_indoor_outdoor': 'https://www.ipb.uni-bonn.de/data/',
        'camera_network_relocalization': 'https://www.epfl.ch/labs/cvlab/data/data-cvlab/',
        'multi_robot_swarm_sync': 'https://cps-vo.org/group/swarmlab',
        'slam_regression_ci': 'https://github.com/MichaelGrupp/evo/tree/master/test/data',
        'sim2real_gap_eval': 'https://developer.nvidia.com/isaac/sim',
        'sports_tracking': 'https://www.epic-kitchens.org/',
        'medical_tool_tracking': 'https://endovis.grand-challenge.org/',
        'mining_robot_underground': 'https://www.subtchallenge.com/',
        'rescue_robot_smoke': 'https://darpa.mil/program/subterranean-challenge',
        'space_robot_docking': 'https://kelvins.esa.int/satellite-pose-estimation-challenge/',
    }

    existing_checkers: dict[str, Callable[[], bool]] = {
        'drone_vio_indoor': _existing_alignanything_drone,
        'drone_vio_outdoor': _existing_alignanything_uzh,
    }

    return plan, manual_links, existing_checkers


def main() -> int:
    rows = _parse_potential_rows(POTENTIAL_MD)
    plan, manual_links, existing_checkers = _build_plan()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        idea = row['idea_id']

        if idea in existing_checkers and existing_checkers[idea]():
            report_rows.append(
                {
                    'order': str(idx),
                    'idea_id': idea,
                    'priority': row['priority'],
                    'candidate_data': row['candidate_data'],
                    'status': 'already_exists_skipped',
                    'item': '',
                    'local_path': '',
                    'url': '',
                    'note': 'dataset already present under /home/yifu/epa_data/AlignAnything',
                }
            )
            continue

        items = plan.get(idea, [])
        if not items:
            report_rows.append(
                {
                    'order': str(idx),
                    'idea_id': idea,
                    'priority': row['priority'],
                    'candidate_data': row['candidate_data'],
                    'status': 'manual_source_only',
                    'item': '',
                    'local_path': '',
                    'url': manual_links.get(idea, ''),
                    'note': 'no stable direct auto-download selected (often self-collected or license-gated)',
                }
            )
            continue

        for it in items:
            out_path = OUT_ROOT / it.rel_path
            partial_path = out_path.with_suffix('.partial' + out_path.suffix)
            if out_path.exists():
                status = 'already_exists_skipped'
                note = 'file already exists'
            elif partial_path.exists():
                status = 'partial_download'
                note = 'partial file exists, resume manually with wget -c'
            elif it.size_bytes is not None and it.size_bytes > MAX_AUTO_DOWNLOAD_BYTES:
                status = 'skipped_too_large'
                note = f'size {it.size_bytes} > {MAX_AUTO_DOWNLOAD_BYTES}'
            else:
                try:
                    ok, reason = _download(it.url, out_path)
                except Exception as exc:  # noqa: BLE001
                    ok, reason = False, f'error:{exc}'
                status = 'downloaded' if ok else 'failed'
                note = reason

            report_rows.append(
                {
                    'order': str(idx),
                    'idea_id': idea,
                    'priority': row['priority'],
                    'candidate_data': row['candidate_data'],
                    'status': status,
                    'item': it.name,
                    'local_path': str(out_path if out_path.exists() else ''),
                    'url': it.url,
                    'note': note,
                }
            )

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    csv_path = OUT_ROOT / 'acquisition_report.csv'
    json_path = OUT_ROOT / 'acquisition_report.json'
    md_path = OUT_ROOT / 'README.md'

    fieldnames = ['order', 'idea_id', 'priority', 'candidate_data', 'status', 'item', 'local_path', 'url', 'note']
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in report_rows:
            w.writerow(r)

    summary = {
        'generated_at': ts,
        'out_root': str(OUT_ROOT),
        'max_auto_download_bytes': MAX_AUTO_DOWNLOAD_BYTES,
        'counts': {
            k: sum(1 for r in report_rows if r['status'] == k)
            for k in sorted({r['status'] for r in report_rows})
        },
        'rows': report_rows,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = [
        '# Ordered Potential Dataset Acquisition',
        '',
        f'Generated at: {ts}',
        f'- Output root: `{OUT_ROOT}`',
        f'- Auto-download size cap: `{MAX_AUTO_DOWNLOAD_BYTES}` bytes',
        '',
        '## Status Counts',
        '',
    ]
    for k, v in summary['counts'].items():
        lines.append(f'- `{k}`: {v}')
    lines.extend(
        [
            '',
            '## Files',
            '',
            '- `acquisition_report.csv`',
            '- `acquisition_report.json`',
        ]
    )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps(summary['counts'], ensure_ascii=False))
    print(f'report_csv={csv_path}')
    print(f'report_json={json_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
