# Quick Start

This page walks through a first run using files provided by `epa`.

## Requirements

- Python 3.10 or newer

## Install

Clone the repository and enter the project directory:

```bash
git clone https://github.com/epic-lab-gwu/epa.git epa
cd epa
```

Install the base package:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev]
pip install -e .[rerun]
pip install -e .[ros]
pip install -e .[geo]
pip install -e .[ipython]
```

Use:

- `.[rerun]` if you want Rerun visualization
- `.[ros]` if you want to read `bag`, `bag2`, or `mcap`
- `.[geo]` if you want map overlays
- `.[dev]` if you want tests and lint tools

## First End-to-End Run

Template for your own dataset (replace paths and formats):

```bash
epa \
  --gt-csv /path/to/your/gt.tum \
  --est-path /path/to/your/est.tum \
  --t-max-diff 0.02 \
  --plot \
  --rerun
```

`--gt-format` and `--est-format` are optional. `epa` defaults to `auto`; only set them when auto-detection is incorrect.

Example:

```bash
epa \
  --engine modular \
  --gt-csv example_groundtruth.csv \
  --est-path example_data/example_estimation.txt \
  --est-format tum \
  --t-max-diff 0.02 \
  --plot
```

This command:

1. Loads the reference trajectory from `example_groundtruth.csv`
2. Loads the estimated trajectory from `example_data/example_estimation.txt`
3. Runs time alignment, extrinsic calibration, and world-frame alignment
4. Computes evaluation metrics
5. Exports figures and summaries

## What You Should See

Each run creates a new directory under `outputs/`:

```text
outputs/run_YYYYmmdd_HHMMSS/
```

Typical outputs:

- `metrics.json`
- `metrics_summary.csv`
- `report_zh.md`
- `report_en.md`
- `plots/*.png`
- stage figures such as time alignment and 3D trajectory plots (also in `plots/`)

If `--plot` is enabled, `epa` also writes metric plots into `outputs/run_.../plots/`.

![Step 1 time alignment result](images/quickstart_step1_time_alignment.png)

*Step 1 output: rotational signals after temporal alignment.*

![Step 2/3 trajectory alignment result](images/quickstart_step23_alignment_3d.png)

*Step 2 and Step 3 output: aligned trajectories in 3D.*

## Run APE and RPE Directly

You can evaluate the same trajectories directly with the metric tools:

```bash
epa_ape tum gt.tum est.tum \
  --pose_relation trans_part \
  --align \
  --t_max_diff 0.02 \
  --plot
```

```bash
epa_rpe tum gt.tum est.tum \
  --pose_relation trans_part \
  --delta 1 \
  --delta_unit f \
  --all_pairs \
  --align \
  --plot
```

Use these commands when you want metric analysis without running the full pipeline.

## Inspect Trajectories

Use `epa_traj` to inspect, synchronize, and plot trajectories:

```bash
epa_traj --format tum --plot --plot-mode xz gt.tum est.tum
```

Common variants:

- Synchronize before plotting: `epa_traj --format tum --sync --ref 1 gt.tum est.tum --plot`
- Align to the reference: `epa_traj --format tum --sync --align --ref 1 gt.tum est.tum --plot`
- Export converted trajectories: `epa_traj --format auto --save-as tum --out-dir outputs/traj_exports example_groundtruth.csv example_data/example_estimation.txt`

## ROS Bag Inputs

If your trajectories are stored in ROS logs, install ROS support first:

```bash
pip install -e .[ros]
```

Then provide both the format and topic:

```bash
epa \
  --engine modular \
  --gt-csv /path/to/run.bag \
  --gt-format bag \
  --gt-topic /vicon/pose \
  --est-path /path/to/run.bag \
  --est-format bag \
  --est-topic /odom
```

Supported ROS log inputs:

- `bag`
- `bag2`
- `mcap`

## Optional Rerun Visualization

If Rerun support is installed, add `--rerun` to the main pipeline or metric tools:

```bash
epa \
  --engine modular \
  --gt-csv example_groundtruth.csv \
  --est-path example_data/example_estimation.txt \
  --est-format tum \
  --plot \
  --rerun
```

<video class="doc-video" controls muted loop playsinline preload="metadata">
  <source src="../images/rerun.mp4" type="video/mp4">
</video>

*Rerun demo: interactive follow-view playback during trajectory inspection.*

Use this to inspect trajectory geometry and intermediate alignment stages.

## Useful Help Commands

```bash
epa --help
epa_traj --help
epa_ape --help
epa_rpe --help
epa_res --help
epa_config --help
```

## Next Steps

- Go to [CLI Reference](cli.md) for command-level options
- Go to [Benchmark](benchmark.md) for batch evaluation workflows
- Go to [Troubleshooting](troubleshooting.md) if your first run fails
