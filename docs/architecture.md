# vicon_ws Architecture

This page explains how `vicon_ws` is organized, how the main pipeline runs, and where each responsibility lives in the codebase.

## Design Goals

The architecture is built around a few goals:

- keep the main pipeline modular and testable
- separate trajectory I/O, alignment logic, evaluation logic, and visualization
- preserve compatibility with the historical `pipeline.py` entry point
- support both one-off runs and reusable CLI tools

## High-Level Execution Flow

For the main pipeline, the execution path is:

```text
vicon_ws CLI
  -> vicon_ws.runner
  -> vicon_ws.core.pipeline_modular
  -> outputs/run_YYYYmmdd_HHMMSS/
```

The modular pipeline:

1. load reference and estimation trajectories
2. optionally crop them by time window
3. perform step-1 time alignment
4. solve step-2 extrinsic calibration
5. solve step-3 world-frame alignment
6. compute APE and RPE metrics
7. write metrics, reports, plots, and optional result bundles

## Main Pipeline Stages

### Step 1: Time Alignment

The pipeline estimates temporal offset by comparing rotational motion signals derived from the trajectories.

Key operations:

- converting quaternion sequences into angular-velocity magnitude signals
- resampling onto a uniform timeline
- cross-correlating the two signals
- selecting the best offset candidate

The angular-motion signal is built from consecutive relative rotations:

\[
\Delta R_i = R_i^{\top} R_{i+1}, \qquad
\omega_i = \frac{\lVert \log(\Delta R_i) \rVert}{\Delta t_i}
\]

After resampling both signals onto a common timeline, the pipeline searches for the offset that maximizes cross-correlation:

\[
\tau^{\ast} = \arg\max_{\tau}\; \mathrm{corr}\!\left(\omega_{\mathrm{ref}}(t), \omega_{\mathrm{est}}(t+\tau)\right)
\]

This gives the time shift that best aligns the rotational dynamics of the two trajectories.

![Step 1 cross-correlation result](images/architecture_step1_cross_correlation.png)

*Example Step 1 output: cross-correlation used to estimate temporal offset.*

This logic lives primarily in:

- `src/vicon_ws/core/time_alignment.py`
- `src/vicon_ws/core/pipeline_modular.py`

### Step 2: Extrinsic Calibration

After temporal alignment, `vicon_ws` estimates the rigid relationship between the reference-side and estimation-side motion.

This stage includes:

- extrinsic rotation solve from relative motion
- extrinsic translation solve from a linear system
- residual and conditioning metrics for solution quality

The rotation solve is based on consistency between relative motions measured in the two trajectory frames:

\[
R_{v,i}^{\top} R_{v,i+1} \, R_{\mathrm{ext}}
\approx
R_{\mathrm{ext}} \, R_{r,i}^{\top} R_{r,i+1}
\]

This finds the extrinsic rotation \(R_{\mathrm{ext}}\) that best maps reference-side relative rotations to estimation-side relative rotations.

Once the rotation is estimated, translation is solved from a linear system of the form:

\[
C \, t_{\mathrm{ext}} \approx d
\]

where each row block comes from consecutive-pose motion constraints after substituting the solved rotation.

Core implementation:

- `src/vicon_ws/core/calibration.py`

### Step 3: World-Frame Alignment

After extrinsic calibration, the pipeline estimates the world-frame transform needed to bring the two trajectories into a comparable frame.

This is a rigid alignment problem over corresponding 3D positions:

\[
R_w, t_w
=
\arg\min_{R,t}
\sum_i
\left\lVert
p_{\mathrm{ref},i} - \left(R \, p_{\mathrm{est},i} + t\right)
\right\rVert^2
\]

The result is the world-frame transform that best overlays the aligned estimation trajectory onto the reference trajectory.

![Step 2/3 alignment result](images/quickstart_step23_alignment_3d.png)

*Example Step 2 and Step 3 output: aligned trajectories after extrinsic and world-frame alignment.*

This stage is also implemented in:

- `src/vicon_ws/core/calibration.py`

## Module Layout

The codebase is organized by responsibility.

Core modules:

- `src/vicon_ws/core/time_alignment.py`: temporal association and signal-based offset estimation
- `src/vicon_ws/core/calibration.py`: extrinsic and world alignment solvers
- `src/vicon_ws/core/evaluation.py`: APE/RPE computation and pose-relation handling
- `src/vicon_ws/core/io_utils.py`: trajectory loading, output directory creation, metrics export, and result bundling
- `src/vicon_ws/core/math_utils.py`: shared math helpers and trajectory transforms
- `src/vicon_ws/core/pipeline_modular.py`: orchestration of the full modular pipeline

CLI and tool modules:

- `src/vicon_ws/cli.py`: main `vicon_ws` CLI parser
- `src/vicon_ws/runner.py`: dispatch between modular and legacy execution
- `src/vicon_ws/traj_tool.py`: trajectory inspection and export tool
- `src/vicon_ws/ape_tool.py`: APE tool
- `src/vicon_ws/rpe_tool.py`: RPE tool
- `src/vicon_ws/config_tool.py`: configuration manager
- `src/vicon_ws/fig_tool.py`: plot bundle re-rendering

Visualization modules:

- `src/vicon_ws/viz/metric_plots.py`: metric figure generation
- `src/vicon_ws/viz/plot_bundle.py`: serialized plot bundle format and rendering
- `src/vicon_ws/viz/rerun_viz.py`: Rerun logging

Benchmark modules:

- `src/vicon_ws/benchmark/alignanything_harness.py`
- `src/vicon_ws/benchmark/plot_summary.py`
- `src/vicon_ws/benchmark/metrics_res.py`
- `src/vicon_ws/benchmark/res.py`

## Main Entry Points

The project exposes several entry points.

Primary user-facing commands:

- `vicon_ws`
- `vicon_ws_traj`
- `vicon_ws_ape`
- `vicon_ws_rpe`
- `vicon_ws_res`
- `vicon_ws_config`
- `vicon_ws_fig`

Compatibility entry point:

- `python pipeline.py`

The compatibility path is still available, but the core algorithm implementation now lives in `src/vicon_ws/core`.

## Modular vs Legacy Execution

`vicon_ws` currently supports two execution modes:

- `--engine modular`
- `--engine legacy`

Behavior:

- `modular` runs the actively maintained implementation in `src/vicon_ws/core/pipeline_modular.py`
- `legacy` routes through `src/vicon_ws/bridge_legacy.py` and then invokes `pipeline.py`

`pipeline.py` is now primarily a compatibility shim and import bridge for older command paths.

## Data Loading Layer

Trajectory loading is centralized so the rest of the pipeline can operate on normalized in-memory arrays.

Supported source families include:

- text trajectory formats such as `csv`, `tum`, `kitti`, and `euroc`
- ROS log formats such as `bag`, `bag2`, and `mcap`

The loader layer normalizes:

- timestamps
- positions
- quaternions
- optional topic selection for ROS inputs

This logic is concentrated in:

- `src/vicon_ws/core/io_utils.py`

## Evaluation Layer

APE and RPE are separate metric families, but they share synchronization and trajectory preprocessing.

The evaluation layer is responsible for:

- timestamp-based trajectory association
- optional alignment before metric computation
- pose-relation selection
- raw error arrays for plots and result export

Main implementation files:

- `src/vicon_ws/core/evaluation.py`
- `src/vicon_ws/metric_cli_common.py`
- `src/vicon_ws/ape_tool.py`
- `src/vicon_ws/rpe_tool.py`

## Output Layer

Each main run creates a timestamped output directory under `outputs/`.

Typical outputs:

- `metrics.json`
- `metrics_summary.csv`
- `metrics_zh.md`
- `plots/*.png`
- stage visualization figures
- optional result bundle `.zip`

This output layer is handled mainly by:

- `src/vicon_ws/core/io_utils.py`
- `src/vicon_ws/viz/metric_plots.py`

## Visualization Layer

Visualization is separated from the numerical pipeline so plotting does not dominate the core alignment logic.

The visualization layer supports:

- saved static metric plots
- trajectory plots
- serialized plot bundles for later re-rendering
- optional Rerun inspection

This separation makes it easier to:

- rerender figures without recomputing metrics
- compare runs after the fact
- keep the core pipeline reusable in batch workflows

## Configuration Layer

Configuration is handled at two levels:

- per-run CLI arguments
- reusable defaults managed by `vicon_ws_config`

The configuration layer is implemented through:

- `src/vicon_ws/config.py`
- `src/vicon_ws/config_cli.py`
- `src/vicon_ws/config_tool.py`

This allows:

- shared defaults across tools
- tool-specific defaults
- JSON config injection through `--config`

## Extension Points

Common extension points:

- add a new trajectory format in `src/vicon_ws/core/io_utils.py`
- add a new metric or pose relation in `src/vicon_ws/core/evaluation.py`
- add new plotting behavior in `src/vicon_ws/viz/`
- add new benchmark workflows in `src/vicon_ws/benchmark/`

Because the architecture separates algorithm logic from command-line parsing, most features can be added without rewriting the whole pipeline.
