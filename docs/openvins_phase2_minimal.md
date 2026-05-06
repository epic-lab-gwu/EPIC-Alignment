# OpenVINS Integration

This page describes the lightweight OpenVINS integration path provided by `epica`.

The goal is to keep `epica` as an external module while still supporting OpenVINS-style evaluation commands and workflows.

## Included Scripts

- `scripts/openvins/install_ov_eval_epa_shims.sh`
- `scripts/openvins/run_minimal_eval_demo.sh`

## Supported OpenVINS-Style Commands

Keep `epica` as an external module, but allow OpenVINS-style commands such as:

- `format_converter`
- `error_singlerun`
- `error_dataset`
- `error_comparison`
- `plot_trajectories`

These command names are forwarded to `python -m epa.ov_eval_compat ...`.
The shim installer also provides suffix-style aliases for common alignment modes, such as
`error_comparison_se3`, `error_singlerun_se3`, and `plot_trajectories_sim3`.

## Quick Start

```bash
conda activate epa

# Pip-installed command (recommended)
epa_openvins /path/to/open_vins/ov_eval/example se3

# Multi case (default: no popup, save to --save-root)
epa_openvins /case/A /case/B --align-mode se3 --save-root /tmp/epa_batch

# Multi case default save root (if --save-root is omitted): /tmp/epa_batch
epa_openvins /case/A /case/B --align-mode se3

# Repo-local wrapper (equivalent)
bash scripts/openvins/run_minimal_eval_demo.sh /path/to/open_vins/ov_eval/example se3
```

The demo now includes:

- the `epica` full pipeline run via `epa.cli` only (produces a `plots/` bundle like
  `step1_cross_correlation.png`, `step23_trajectory_alignment_3d.png`,
  `ape_translation_part_*.png`, `rpe_translation_part_*.png`)

To skip plotting:

```bash
epa_openvins /path/to/open_vins/ov_eval/example se3 --no-plot
```

To keep files on disk (single-case default is cleanup):

```bash
epa_openvins /path/to/open_vins/ov_eval/example se3 --keep-output
```

Full pipeline plots are saved under:

- `outputs/run_*/plots/`

If you still want `ov_eval`-style commands, install the optional shims:

```bash
bash scripts/openvins/install_ov_eval_epa_shims.sh
export PATH="$(pwd)/tools/epa_ov_eval_shims:$PATH"

# Both forms are supported:
error_comparison se3 /path/to/gt /path/to/algorithms
error_comparison_se3 /path/to/gt /path/to/algorithms
```

For low-rate, short, or difficult sequences, the `se3` compatibility path exposes EPA Step-3
debug parameters:

```bash
error_comparison_se3 /path/to/gt /path/to/algorithms \
  --epa-dt-resample 0.01 \
  --epa-offset-min-match-ratio 0.1 \
  --epa-downsample-hz 20 \
  --epa-quat-interp slerp
```

By default, if EPA Step-3 evaluation fails, the compatibility layer falls back to
ov_eval-style SE3 and prints the EPA parameters used. To fail immediately instead:

```bash
error_comparison_se3 /path/to/gt /path/to/algorithms --epa-no-fallback
```
