# Quick Start

## 0. Preview Docs Locally

```bash
pip install mkdocs
mkdocs serve
```

Open `http://127.0.0.1:8000`.

## 1. Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev]
pip install -e .[rerun]
pip install -e .[ros]
pip install -e .[geo]
```

## 2. Run Main Pipeline

```bash
vicon_ws \
  --engine modular \
  --gt-csv gt.csv \
  --est-path outputs/traj_estimate_v1_01.txt \
  --est-format tum \
  --t-max-diff 0.02 \
  --plot
```

Expected outputs (example):

- `outputs/run_YYYYmmdd_HHMMSS/metrics.json`
- `outputs/run_YYYYmmdd_HHMMSS/metrics_summary.csv`
- `outputs/run_YYYYmmdd_HHMMSS/plots/*.png`

## 3. Run APE / RPE

```bash
vicon_ws_ape tum gt.tum est.tum --pose_relation trans_part --plot
vicon_ws_rpe tum gt.tum est.tum --pose_relation trans_part --delta 1 --delta_unit f --plot
```

## 4. Optional rerun

```bash
vicon_ws --engine modular --gt-csv gt.csv --est-path outputs/traj_estimate_v1_01.txt --est-format tum --rerun
```

## 5. Verify Local Quality Gates

```bash
ruff check src tests pipeline.py
pytest -q tests/unit
```
