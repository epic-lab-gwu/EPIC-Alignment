# epa Architecture

This page explains how `epa` is organized, how the main pipeline runs, and where each responsibility lives in the codebase.

## Design Goals

The architecture is built around a few goals:

- keep the main pipeline modular and testable
- separate trajectory I/O, alignment logic, evaluation logic, and visualization
- support both one-off runs and reusable CLI tools

## High-Level Execution Flow

For the main pipeline, the execution path is:

```text
epa CLI
  -> epa.runner
  -> epa.core.pipeline_modular
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

- `src/epa/core/time_alignment.py`
- `src/epa/core/pipeline_modular.py`

### Step 2: Extrinsic Calibration

After temporal alignment, `epa` estimates the rigid relationship between the reference-side and estimation-side motion.

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

- `src/epa/core/calibration.py`

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

- `src/epa/core/calibration.py`

## Module Layout

The codebase is organized by responsibility.

Core modules:

- `src/epa/core/time_alignment.py`: temporal association and signal-based offset estimation
- `src/epa/core/calibration.py`: extrinsic and world alignment solvers
- `src/epa/core/evaluation.py`: APE/RPE computation and pose-relation handling
- `src/epa/core/io_utils.py`: trajectory loading, output directory creation, metrics export, and result bundling
- `src/epa/core/math_utils.py`: shared math helpers and trajectory transforms
- `src/epa/core/pipeline_modular.py`: orchestration of the full modular pipeline

CLI and tool modules:

- `src/epa/cli.py`: main `epa` CLI parser
- `src/epa/runner.py`: execute modular pipeline and dry-run preview
- `src/epa/traj_tool.py`: trajectory inspection and export tool
- `src/epa/ape_tool.py`: APE tool
- `src/epa/rpe_tool.py`: RPE tool
- `src/epa/config_tool.py`: configuration manager
- `src/epa/fig_tool.py`: plot bundle re-rendering

Visualization modules:

- `src/epa/viz/metric_plots.py`: metric figure generation
- `src/epa/viz/plot_bundle.py`: serialized plot bundle format and rendering
- `src/epa/viz/rerun_viz.py`: Rerun logging

Benchmark modules:

- `src/epa/benchmark/alignanything_harness.py`
- `src/epa/benchmark/plot_summary.py`
- `src/epa/benchmark/metrics_res.py`
- `src/epa/benchmark/res.py`

## Main Entry Points

The project exposes several entry points.

Primary user-facing commands:

- `epa`
- `epa_traj`
- `epa_ape`
- `epa_rpe`
- `epa_res`
- `epa_config`
- `epa_fig`

`epa` runs the actively maintained modular implementation in `src/epa/core/pipeline_modular.py`.

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

- `src/epa/core/io_utils.py`

## Evaluation Layer

APE and RPE are separate metric families, but they share synchronization and trajectory preprocessing.

The evaluation layer is responsible for:

- timestamp-based trajectory association
- optional alignment before metric computation
- pose-relation selection
- raw error arrays for plots and result export

Main implementation files:

- `src/epa/core/evaluation.py`
- `src/epa/metric_cli_common.py`
- `src/epa/ape_tool.py`
- `src/epa/rpe_tool.py`

## Output Layer

Each main run creates a timestamped output directory under `outputs/`.

Typical outputs:

- `metrics.json`
- `metrics_summary.csv`
- `report_zh.md`
- `report_en.md`
- `plots/*.png`
- stage visualization figures
- optional result bundle `.zip`

This output layer is handled mainly by:

- `src/epa/core/io_utils.py`
- `src/epa/viz/metric_plots.py`

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
- reusable defaults managed by `epa_config`

The configuration layer is implemented through:

- `src/epa/config.py`
- `src/epa/config_cli.py`
- `src/epa/config_tool.py`

This allows:

- shared defaults across tools
- tool-specific defaults
- JSON config injection through `--config`

## Extension Points

Common extension points:

- add a new trajectory format in `src/epa/core/io_utils.py`
- add a new metric or pose relation in `src/epa/core/evaluation.py`
- add new plotting behavior in `src/epa/viz/`
- add new benchmark workflows in `src/epa/benchmark/`

Because the architecture separates algorithm logic from command-line parsing, most features can be added without rewriting the whole pipeline.
