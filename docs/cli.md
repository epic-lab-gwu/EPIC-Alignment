# CLI Reference

This page summarizes the main command-line tools provided by `epica`. The package installs both `epa` and `epica`; this page uses `epa` in examples, but the two commands are interchangeable.

## Command Overview

Core pipeline:

- `epa` or `epica`: run the full 3-step alignment and evaluation pipeline

Trajectory and metric tools:

- `epa_traj`: inspect, synchronize, align, transform, plot, and export trajectories
- `epa_ape`: compute absolute pose error metrics
- `epa_rpe`: compute relative pose error metrics
- `epa_res`: compare result bundles, run directories, or metrics files

Utilities:

- `epa_config`: manage reusable global and tool-level defaults
- `epa_fig`: re-render plots from serialized plot bundles

Benchmark tools are documented separately in [Benchmark Workflow](benchmark_workflow.md).

## Shared Concepts

### Supported Formats

Most tools support these input formats:

- `auto`
- `csv`
- `euroc`
- `tum`
- `kitti`
- `bag`
- `bag2`
- `mcap`

For ROS log formats, install:

```bash
python -m pip install "epica[ros]"
```

### Topics for ROS Inputs

For ROS logs, you usually need to provide a topic:

```bash
epa /path/to/run.bag /path/to/run.bag \
  --gt-format bag --gt-topic /vicon/pose \
  --est-format bag --est-topic /odom
```

For `epa_traj`, you can also encode the topic inside each trajectory spec:

```bash
epa_traj --format bag /path/to/run.bag::/vicon/pose /path/to/run.bag::/odom --plot
```

### Time Association

Several tools match trajectories by timestamp before computing metrics.

Key options:

- `t_offset`: shifts estimation timestamps before matching
- `t_max_diff`: maximum residual timestamp difference allowed for a valid match
- `t_start` and `t_end`: crop trajectories to a time window before evaluation

Guidelines:

- Use smaller `t_max_diff` values when timestamps are already well aligned
- Increase `t_max_diff` only when you know the logs are sparse or noisy
- Use `t_offset` when one trajectory is consistently ahead of or behind the other

### Help Commands

```bash
epa --help
epa_traj --help
epa_ape --help
epa_rpe --help
epa_res --help
epa_config --help
```

## `epa` / `epica`

`epa` / `epica` is the main entry point. It loads a reference trajectory and an estimated trajectory, runs the 3-step pipeline, computes metrics, and writes plots and summaries into a new run directory.

Common options:

- positional `<gt_file>` or `--gt`: reference trajectory path
- `--gt-format`: reference format
- `--gt-topic`: reference topic for ROS logs
- positional `<est_file>` or `--est`: estimation trajectory path
- `--est-format`: estimation format
- `--est-topic`: estimation topic for ROS logs
- `--t-max-diff`: maximum timestamp association gap
- `--t-offset`: constant offset applied to estimation timestamps before sync
- `--plot` and `--no-plot`: enable or disable metric plot generation
- `--save-results`: write a bundled result zip
- `--save-full-metrics`: keep full per-sample APE/RPE arrays in `metrics.json`
- `--rerun`: enable Rerun logging

Minimal example:

```bash
epa example_data/example_groundtruth.csv example_data/example_estimation.txt
```

With result bundle export:

```bash
epa example_data/example_groundtruth.csv example_data/example_estimation.txt \
  --save-results outputs/results/run_a.zip
```

<video class="doc-video" controls muted loop playsinline preload="metadata">
  <source src="../images/rerun.mp4" type="video/mp4">
</video>

*Optional Rerun inspection flow for the main `epa` / `epica` pipeline.*

## `epa_traj`

`epa_traj` is the general trajectory utility. Use it to inspect, synchronize, align, plot, or export trajectories.

Common options:

- `--format`: input format for all trajectories
- `--topic`: default topic for bag-style inputs
- `--sync`: associate all non-reference trajectories to the reference by timestamp
- `--sync-max-diff`: timestamp tolerance for `--sync`
- `--sync-offset`: constant timestamp offset before `--sync`
- `--align`: align non-reference trajectories to the reference
- `--correct-scale`: enable scale correction with alignment
- `--ref`: reference trajectory label or 1-based index
- `--plot`: generate plots
- `--plot-mode`: choose `xy`, `xz`, `yz`, or `xyz`
- `--save-as`: export loaded trajectories in another format
- `--out-dir`: output directory for exported files or generated outputs

Plot two TUM trajectories:

```bash
epa_traj --format tum --plot --plot-mode xz gt.tum est.tum
```

Synchronize and align to the first trajectory:

```bash
epa_traj --format tum --sync --align --ref 1 gt.tum est.tum --plot
```

Export trajectories as TUM:

```bash
epa_traj \
  --format auto \
  --save-as tum \
  --out-dir outputs/traj_exports \
  example_data/example_groundtruth.csv example_data/example_estimation.txt
```

## `epa_ape`

`epa_ape` computes absolute pose error between a reference trajectory and an estimated trajectory.

Common options:

- `--pose_relation`: metric relation such as `trans_part`, `rot_part`, `angle_deg`, or `full`
- `--align`: run SE(3) alignment before evaluation
- `--correct_scale`: enable scale correction
- `--align_origin`: align the first pose to the reference origin
- `--t_max_diff`: timestamp matching tolerance
- `--t_offset`: constant timestamp shift before matching
- `--plot`: generate metric plots
- `--save_results`: export a result zip
- `--serialize_plot`: save a reusable plot bundle

Example:

```bash
epa_ape tum gt.tum est.tum \
  --pose_relation trans_part \
  --align \
  --t_max_diff 0.02 \
  --plot
```

<div class="doc-image-grid two-col">
  <img src="../images/cli_ape_raw.png" alt="APE raw error curve">
  <img src="../images/cli_ape_map.png" alt="APE map view">
</div>

*Example `epa_ape` outputs: raw error curve and map-colored trajectory.*

Save a reusable plot bundle:

```bash
epa_ape tum gt.tum est.tum \
  --pose_relation trans_part \
  --serialize_plot outputs/ape_plot.json
```

## `epa_rpe`

`epa_rpe` computes relative pose error. The key extra concept is `delta`, which defines the spacing between pose pairs.

Common options:

- `--pose_relation`: metric relation such as `trans_part`, `rot_part`, or `point_distance_error_ratio`
- `--delta`: separation between pose pairs
- `--delta_unit`: `f` for frames, `m` for meters, `d` for degrees, `r` for radians, `s` for seconds
- `--delta_tol`: relative tolerance used in all-pairs mode for non-frame deltas
- `--all_pairs`: use all candidate pairs
- `--pairs_from_reference`: build RPE pairs from the reference instead of the estimate
- `--align`: run SE(3) alignment before evaluation
- `--plot`: generate metric plots

Example:

```bash
epa_rpe tum gt.tum est.tum \
  --pose_relation trans_part \
  --delta 1 \
  --delta_unit f \
  --all_pairs \
  --align \
  --plot
```

<div class="doc-image-grid two-col">
  <img src="../images/cli_rpe_raw.png" alt="RPE raw error curve">
  <img src="../images/cli_rpe_map.png" alt="RPE map view">
</div>

*Example `epa_rpe` outputs: raw relative error curve and map-colored trajectory.*

Path-based RPE example:

```bash
epa_rpe tum gt.tum est.tum \
  --pose_relation trans_part \
  --delta 1.0 \
  --delta_unit m \
  --all_pairs
```

Time-based RPE example:

```bash
epa_rpe tum gt.tum est.tum \
  --pose_relation trans_part \
  --delta 1.0 \
  --delta_unit s \
  --all_pairs
```

## `epa_res`

`epa_res` compares previous evaluation outputs. Each input can be:

- a `.zip` result bundle
- a `metrics.json` file
- a run directory containing `metrics.json`

Common options:

- `--metric`: compare `ape`, `rpe`, or `all`
- `--stage`: select `raw`, `step2`, or `step3`
- `--stat`: choose the statistic to compare such as `rmse`, `mean`, or `median`
- `--plot`: generate comparison plots
- `--out-dir`: directory for generated outputs
- `--save-plot`: export plots
- `--save-table`: export a comparison table

Example:

```bash
epa_res outputs/results/run_a.zip outputs/results/run_b.zip \
  --metric all \
  --stage step3 \
  --plot \
  --out-dir outputs/res_compare
```

<div class="doc-image-grid two-col">
  <img src="../images/cli_res_aggregated_raw.png" alt="Aggregated raw comparison">
  <img src="../images/cli_res_aggregated_hist.png" alt="Aggregated histogram comparison">
  <img src="../images/cli_res_aggregated_box.png" alt="Aggregated box comparison">
  <img src="../images/cli_res_aggregated_violin.png" alt="Aggregated violin comparison">
</div>

*Example `epa_res` outputs for multi-run comparison.*

## `epa_config`

`epa_config` manages reusable defaults so you do not need to repeat the same flags.

It supports:

- root-level defaults shared across tools
- tool-specific defaults such as `epa_ape` or `epa_traj`
- generated template configs for supported tools

Common commands:

Show current settings:

```bash
epa_config show
```

Show effective config for one tool:

```bash
epa_config show --tool epa_ape
```

Set shared defaults:

```bash
epa_config set plot false rpe_delta 3
```

Set tool-specific defaults:

```bash
epa_config set --tool epa_ape plot_mode xy t_max_diff 0.05
epa_config set --tool epa_traj plot_mode xyz sync_max_diff 0.01
```

Generate a template config:

```bash
epa_config generate --tool epa_ape --out ape_config.json
```

## `epa_fig`

`epa_fig` re-renders serialized plot bundles produced by tools such as `epa_ape` and `epa_rpe`.

Common options:

- `--out_dir`: directory for rendered figures
- `--save_plot`: custom output name stem
- `--dpi`: output resolution
- `--list`: list figure names in a bundle without rendering

Example:

```bash
epa_fig outputs/ape_plot.json --save_plot outputs/ape_rerender.png
```

## Common Recipes

Run one pair end to end:

```bash
epa example_data/example_groundtruth.csv example_data/example_estimation.txt
```

Inspect alignment before metric evaluation:

```bash
epa_traj --format tum --sync --align --ref 1 gt.tum est.tum --plot
```

Compute APE with explicit sync tolerance:

```bash
epa_ape tum gt.tum est.tum \
  --pose_relation trans_part \
  --t_max_diff 0.02 \
  --t_offset 0.0 \
  --plot
```

Compare two previous runs:

```bash
epa_res outputs/results/run_a.zip outputs/results/run_b.zip \
  --metric ape \
  --stage step3 \
  --stat rmse \
  --plot
```

## Config Priority

All core tools support `--config <file.json>`.

Priority order:

1. values loaded from `--config`
2. values passed on the command line
3. defaults stored through `epa_config`
