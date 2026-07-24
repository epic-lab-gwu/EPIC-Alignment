# OpenVINS Integration

This page describes the lightweight OpenVINS integration path provided by `epica`.
The goal is to keep `epica` as an external module while still supporting
OpenVINS-style evaluation commands and workflows.

## Quick Start

For most users, start with the OpenVINS-style comparison command:

```bash
conda activate epa

python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder
```

Example folder layout:

```text
gt_folder/
  R_11_5cp.txt
  R_12_10cp.txt

algorithm_pose_folder/
  svo_mono/
    R_11_5cp/
      svo_poses.txt
    R_12_10cp/
      svo_poses.txt
```

The default `se3` mode uses EPA's time alignment, extrinsic calibration, and world-frame
alignment for evaluation when possible. If EPA's direct alignment cannot evaluate a case, it quietly falls back to ov_eval-style SE3 and
reports the count in `TOOL SOURCE`.

## Common Commands

Run a full OpenVINS-style multi-sequence comparison:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder
```

Run one GT/EST pair:

```bash
python -m epa.ov_eval_compat error_singlerun se3 \
  /path/to/gt.txt \
  /path/to/est.txt
```

Run one dataset against all algorithms in a folder:

```bash
python -m epa.ov_eval_compat error_dataset se3 \
  /path/to/gt.txt \
  /path/to/algorithm_pose_folder
```

Convert OpenVINS/EuRoC-style CSV pose files:

```bash
python -m epa.ov_eval_compat format_converter /path/to/csv_or_folder
```

Plot trajectories in compatibility mode:

```bash
python -m epa.ov_eval_compat plot_trajectories se3 \
  /path/to/gt.txt \
  /path/to/est.txt
```

## Optional ov_eval-Style Shims

If you want shorter command names such as `error_comparison`, install the
optional shims:

```bash
bash scripts/openvins/install_ov_eval_epa_shims.sh
export PATH="$(pwd)/tools/epa_ov_eval_shims:$PATH"
```

After that, both forms are supported:

```bash
error_comparison se3 /path/to/gt_folder /path/to/algorithm_pose_folder
error_comparison_se3 /path/to/gt_folder /path/to/algorithm_pose_folder
```

These command names are forwarded to `python -m epa.ov_eval_compat ...`.

## Output Summary

The compatibility output reports:

- `TOOL SOURCE: epa=N, ov_eval=M`: how many cases used EPA direct alignment versus the
  ov_eval-style fallback.
- `FULL TRAJECTORY ATE`: full-sequence absolute trajectory error.
- `FULL TRAJECTORY DISTANCE RPE`: distance-segment relative pose error.
- `FULL TRAJECTORY DISTANCE DRIFT RATE`: `mean(translation RPE / segment length) * 100`.
- `FULL TRAJECTORY 1S TIME RPE`: 1-second time-based relative pose error.
- `DRIFT-VALID SUCCESS RATE`: percentage of GT path length not marked as local
  drift/fail.
- `DRIFT-VALID ONLY ...`: the same metric families computed only on detected
  non-fail segments.

For `error_comparison`, distance RPE and drift-rate currently use fixed segment
lengths: `8, 16, 24, 32, 40, 48` meters. These values are not automatically read
from dataset metadata. If a dataset-specific segment policy is later defined,
the compatibility layer can be extended to accept manual or automatic segment
selection.

Full-trajectory ATE remains the primary ATE metric. Drift-valid-only metrics use
the same EPA metric definitions as the main pipeline and are intended to show
local performance after excluding detected fail segments. See
[Error Metrics](error_metrics.md) for the general EPA metric definitions.

## Advanced Options

The defaults are intended to work for typical OpenVINS-style evaluation. Use the
options below only when a dataset needs tuning.

Force a fixed APE threshold metadata value:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder \
  --epa-success-threshold-mode fixed \
  --epa-success-threshold-m 20
```

Override the adaptive clamp range:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder \
  --epa-success-threshold-min-m 5 \
  --epa-success-threshold-max-m 30
```

Relax or tighten the global failed-case gate:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder \
  --epa-success-global-gate-m 50 \
  --epa-success-global-gate-percentile 5
```

Adjust local drift detection:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder \
  --epa-success-drift-rpe-1s-m 2 \
  --epa-success-drift-ape-slope-mps 1 \
  --epa-success-drift-ape-jump-m 5
```

For low-rate, short, or difficult sequences, the `se3` compatibility path exposes
EPA Step-3 debug parameters:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder \
  --epa-dt-resample 0.01 \
  --epa-offset-min-match-ratio 0.1 \
  --epa-downsample-hz 20 \
  --epa-quat-interp slerp
```

Print fallback details:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder \
  --epa-verbose-fallback
```

Fail immediately instead of falling back:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder \
  --epa-no-fallback
```

Return a non-zero exit code if any run is skipped or no valid run is found:

```bash
python -m epa.ov_eval_compat error_comparison se3 \
  /path/to/gt_folder \
  /path/to/algorithm_pose_folder \
  --fail-on-skipped
```

## Full EPA Pipeline Wrapper

For OpenVINS case folders where you want the full EPA pipeline and saved plots,
use:

```bash
epa_openvins /path/to/open_vins/ov_eval/example se3
```

Multi-case usage:

```bash
epa_openvins /case/A /case/B --align-mode se3
```

The default multi-case save root is `/tmp/epa_batch`. To choose a location:

```bash
epa_openvins /case/A /case/B --align-mode se3 --save-root /tmp/epa_batch
```

To skip plotting:

```bash
epa_openvins /path/to/open_vins/ov_eval/example se3 --no-plot
```

To keep intermediate files on disk:

```bash
epa_openvins /path/to/open_vins/ov_eval/example se3 --keep-output
```

Full pipeline plots are saved under:

```text
outputs/run_*/plots/
```

Repo-local wrapper equivalent:

```bash
bash scripts/openvins/run_minimal_eval_demo.sh /path/to/open_vins/ov_eval/example se3
```
