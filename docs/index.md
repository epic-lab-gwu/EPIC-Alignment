# EPICA Documentation

`epica` is a trajectory alignment and evaluation toolkit. The package installs the command-line entrypoints `epa` and `epica`, which are interchangeable. It is built for reproducible benchmarking, practical analysis, and scalable experiment comparison.

<video class="doc-video" controls autoplay muted loop playsinline preload="metadata">
  <source src="images/rerun.mp4" type="video/mp4">
</video>

*Rerun demo: follow-view inspection of Ground truth and Estimation trajectories.*

## Features

- End-to-end trajectory alignment from time synchronization to final world-frame evaluation
- Command-line tools for single-run analysis, plotting, metric inspection, and cross-run comparison
- Reusable configuration defaults to keep repeated workflows consistent
- Reproducible figure generation through serialized plot specs and offline re-rendering
- Optional Rerun-based visualization for trajectory inspection and intermediate-stage debugging
- Batch benchmark workflows for large-scale evaluation, summary plots, and report generation

## Supported Input Formats

The `epa` / `epica` command supports these trajectory and log formats:

- `auto` for automatic format detection when possible
- `csv` / `euroc`
- `tum`
- `kitti`
- `bag` (ROS1)
- `bag2` / `mcap` (ROS2)

For ROS log formats (`bag`, `bag2`, `mcap`), install ROS support first:

```bash
python -m pip install "epica[ros]"
```

## Main Commands

You may start with these commands:

- `epa` or `epica`: run the full alignment and evaluation pipeline on one reference-estimation pair
- `epa_all`: run the single-case workflow in one command, including plots and packaged outputs

Trajectory and metric tools:

- `epa_traj`: inspect, synchronize, align, and plot trajectories
- `epa_ape`: compute absolute pose error metrics
- `epa_rpe`: compute relative pose error metrics
- `epa_res`: compare multiple runs, result bundles, or metric outputs

Benchmark tools:

- `epa_bench`: run multi-case benchmarks across many prepared cases
- `epa_benchall`: run the full benchmark workflow, including summary plots and LaTeX tables

Advanced utilities such as `epa_config`, `epa_fig`, `epa_plot_summary`, and `epa_metric_res` are documented in the [CLI Reference](cli.md).

## Typical Workflow

1. Prepare one reference trajectory and one estimated trajectory
2. Run `epa` (or `epica`) to align them and generate a run directory with metrics and plots
3. Use `epa_traj`, `epa_ape`, or `epa_rpe` when you want more focused inspection
4. Use `epa_res` to compare multiple runs
5. Move to `epa_bench` or `epa_benchall` when you need batch evaluation

## Alignment Modes

The main EPA modes are `epa_se3`, `epa_sim3`, and `epa_posyaw`.

- `epa_se3`: metric-scale SE3 evaluation for VIO/odometry.
- `epa_sim3`: scale-aware Sim3 evaluation for scale-ambiguous visual SLAM/VO.
- `epa_posyaw`: yaw-only + translation evaluation for gravity-aligned VIO.

See [EPA Alignment Modes](cli.md#epa-alignment-modes) for the exact behavior,
reliability notes, and one-command examples for each mode.

## Next Pages

- [Quick Start](quickstart.md): run your first example end to end
- [CLI Reference](cli.md): look up commands and frequently used options
- [Error Metrics](error_metrics.md): understand ATE, RPE, drift rate, and drift-valid metrics
- [Benchmark Workflow](benchmark_workflow.md): run batch evaluation and generate summary artifacts
- [Troubleshooting](troubleshooting.md): resolve common setup and runtime issues
- [Architecture](architecture.md): understand the pipeline internals and module layout
