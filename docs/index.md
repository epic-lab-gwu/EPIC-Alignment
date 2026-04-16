# epa Documentation

`epa` is a toolkit for trajectory alignment and evaluation in Vicon-grounded workflows. It is built for reproducible benchmarking, practical analysis, and scalable experiment comparison.

<video class="doc-video" controls autoplay muted loop playsinline preload="metadata">
  <source src="images/rerun.mp4" type="video/mp4">
</video>

*Rerun demo: follow-view inspection of GT and Step3 trajectories.*

## What epa Does

`epa` helps you:

- Align estimated trajectories with reference trajectories
- Evaluate trajectory quality with reproducible metrics
- Generate plots and outputs for analysis, comparison, and reporting
- Run batch benchmarks across datasets and cases

## Features

- 3-step alignment pipeline: time alignment -> extrinsic solve -> world-frame alignment
- Dedicated CLI suite for end-to-end runs, metric analysis, plotting, and result comparison
- Reusable configuration system with global and tool-level defaults
- Plot serialization and offline re-rendering via `epa_fig`
- Optional Rerun-based visual inspection for trajectories and intermediate stages
- Benchmark harness for large-scale evaluation and summary generation

## Supported Input Formats

`epa` supports these trajectory and log formats:

- `auto` for automatic format detection when possible
- `csv` / `euroc`
- `tum`
- `kitti`
- `bag` (ROS1)
- `bag2` / `mcap` (ROS2)

For ROS log formats (`bag`, `bag2`, `mcap`), install ROS support first:

```bash
pip install -e .[ros]
```

## Main Commands

Core pipeline:

- `epa`: run the full 3-step alignment and evaluation pipeline

Trajectory and metric tools:

- `epa_traj`: inspect, sync, align, and plot trajectories
- `epa_ape`: compute absolute pose error metrics
- `epa_rpe`: compute relative pose error metrics
- `epa_res`: compare result bundles and metric outputs

Utilities:

- `epa_config`: manage reusable configuration defaults
- `epa_fig`: re-render plots from serialized plot specifications

Benchmark tools:

- `epa_benchmark`: run batch benchmarks across many cases
- `epa_plot_summary`: generate summary plots from benchmark CSV outputs
- `epa_metric_res`: aggregate and compare metric result files

## Typical Workflow

1. Prepare a reference trajectory and an estimated trajectory
2. Run `epa` for end-to-end alignment and evaluation
3. Use `epa_ape` or `epa_rpe` for focused metric analysis
4. Use `epa_res` to compare multiple runs
5. Use benchmark tools for large-scale experiment summaries

## Typical Outputs

A typical run produces:

- Alignment and evaluation metrics
- CSV and JSON summaries
- Trajectory and error plots
- Serialized plot specifications for reproducible figure generation
- Benchmark case summaries and aggregated reports

## Next Pages

- [Quick Start](quickstart.md)
- [CLI Reference](cli.md)
- [Benchmark](benchmark.md)
- [Troubleshooting](troubleshooting.md)
- [Architecture](architecture.md)
