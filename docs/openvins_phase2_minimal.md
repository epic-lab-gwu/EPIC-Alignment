# OpenVINS Minimal Integration (Phase 2)

This phase adds a minimal OpenVINS-side calling path without changing OpenVINS source code.

## What was added

- `scripts/openvins/install_ov_eval_epa_shims.sh`
- `scripts/openvins/run_minimal_eval_demo.sh`

## Goal

Keep EPA as an external module, but allow OpenVINS-style commands:

- `format_converter`
- `error_singlerun`
- `error_dataset`
- `error_comparison`
- `plot_trajectories`

These command names are forwarded to `python -m epa.ov_eval_compat ...`.

## Quick test

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

- EPA full pipeline run via `epa.cli` only (produces rich `plots/` bundle like
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

EPA full pipeline plots are saved under:

- `outputs/run_*/plots/`

If you still want `ov_eval`-style commands, install optional shims:

```bash
bash scripts/openvins/install_ov_eval_epa_shims.sh
export PATH="$(pwd)/tools/epa_ov_eval_shims:$PATH"
```
