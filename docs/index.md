# vicon_ws Documentation

`vicon_ws` is a toolkit for trajectory alignment and evaluation in Vicon-grounded workflows. It is built for reproducible benchmarking, practical analysis, and scalable experiment comparison.

<video 
  src="https://videotourl.com/videos/1776264406889-7a49889b-4ab6-4201-b69d-6190ca2ca008.mp4" 
  controls 
  autoplay 
  muted 
  loop 
  style="display: block; margin: 0 auto; width: 80%; max-width: 800px;">
</video>

## What vicon_ws Does

`vicon_ws` helps you:

- Align estimated trajectories with reference trajectories
- Evaluate trajectory quality with reproducible metrics
- Generate plots and outputs for analysis, comparison, and reporting
- Run batch benchmarks across datasets and cases

## Features

- 3-step alignment pipeline: time alignment -> extrinsic solve -> world-frame alignment
- Dedicated CLI suite for end-to-end runs, metric analysis, plotting, and result comparison
- Reusable configuration system with global and tool-level defaults
- Plot serialization and offline re-rendering via `vicon_ws_fig`
- Optional Rerun-based visual inspection for trajectories and intermediate stages
- Benchmark harness for large-scale evaluation and summary generation

## Supported Input Formats

`vicon_ws` supports these trajectory and log formats:

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

- `vicon_ws`: run the full 3-step alignment and evaluation pipeline

Trajectory and metric tools:

- `vicon_ws_traj`: inspect, sync, align, and plot trajectories
- `vicon_ws_ape`: compute absolute pose error metrics
- `vicon_ws_rpe`: compute relative pose error metrics
- `vicon_ws_res`: compare result bundles and metric outputs

Utilities:

- `vicon_ws_config`: manage reusable configuration defaults
- `vicon_ws_fig`: re-render plots from serialized plot specifications

Benchmark tools:

- `vicon_ws_benchmark`: run batch benchmarks across many cases
- `vicon_ws_plot_summary`: generate summary plots from benchmark CSV outputs
- `vicon_ws_metric_res`: aggregate and compare metric result files

## Typical Workflow

1. Prepare a reference trajectory and an estimated trajectory
2. Run `vicon_ws` for end-to-end alignment and evaluation
3. Use `vicon_ws_ape` or `vicon_ws_rpe` for focused metric analysis
4. Use `vicon_ws_res` to compare multiple runs
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
