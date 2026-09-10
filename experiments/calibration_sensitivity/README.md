# Calibration-sensitivity heatmaps

This directory contains the reproducible controlled-simulation experiment for
Table 1 and Figs. 1--4 of the calibration-sensitivity study.

## Scientific question

How much do timestamp and camera--IMU extrinsic perturbations change trajectory
APE/ARE, and how much of that sensitivity remains after EPA calibration?

The four figure policies are:

1. position-only SE(3) world alignment, calibration disabled;
2. rotation-first SE3R world alignment, calibration disabled;
3. position-only SE(3) world alignment, EPA calibration enabled;
4. rotation-first SE3R world alignment, EPA calibration enabled.

The experiment uses the same 3,080 matched simulated trajectories under all four
policies. All four heatmaps use one shared symmetric color normalization.

## Files

- `run_calibration_sensitivity.py`: canonical one-command experiment and figure
  generator;
- `EXPERIMENT_SETUP.md`: frozen reviewer-facing setup, factors, metrics,
  replicate definition, and evidence boundary;
- `run_pilot.py`: trajectory loading, normalization, simulated-VIO generation,
  perturbation injection, and TUM I/O primitives;
- `run_main.py`: the nine main motion templates and five fixed random seeds;
- `run_relative_baseline.py`: the full eleven-template/fourteen-condition design
  and rotation-first no-calibration evaluator;
- `audit_panel_alignment.py`: local, dependency-free panel-layout gate used by
  the figure generator.

Generated result directories and private/local datasets are intentionally not
committed.

## Data layout

Set `EPA_CALIBRATION_DATA_ROOT` to a directory containing:

```text
<data-root>/
├── hot3d_gt1/
│   ├── P0009_1b21bb01.txt
│   └── P0011_451a7734.txt
├── hot3d_gt2/
│   └── P0012_915e71c6.txt
├── AlignAnything2/GT/euroc_mav/
│   ├── V1_01_easy_original.txt
│   ├── V1_02_medium.txt
│   ├── V2_02_medium.txt
│   └── V2_03_difficult.txt
├── kitti_gt/
│   ├── 00.txt
│   ├── 03.txt
│   └── 05.txt
└── aea_gt/
    └── loc1111.txt
```

These are real GT motion templates. The VIO errors and calibration perturbations
applied to them are simulated.

## Environment

Install the repository and its existing runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run

From the repository root:

```bash
export EPA_CALIBRATION_DATA_ROOT=/path/to/calibration_data
python experiments/calibration_sensitivity/run_calibration_sensitivity.py \
  --epa-repo . \
  --output-dir outputs/calibration_sensitivity \
  --workers 12
```

Use `--overwrite` to replace an output directory or `--resume` to reuse the
incremental `data/calibration_cache.jsonl`. Cache keys include fingerprints of
the EPA CLI and trajectory-alignment implementation, so changed EPA code does
not silently reuse stale results.

The script detects both public mode-name schemes used during development:

- current `main`: `se3-original` = position-only and `se3` = rotation-first;
- naming-cleanup branch: `se3` = position-only and `se3r` = rotation-first.

It validates the solver reported by every calibrated run, rather than relying
only on the mode spelling.

## Expected outputs

```text
<output-dir>/
├── config.json
├── setup.md
├── validation.json
├── data/
│   ├── calibration_cache.jsonl
│   ├── case_metrics.csv
│   ├── heatmap_values.csv
│   └── simulated_vio_rmse.csv
└── figures/
    ├── fig1_se3_original_without_calibration.{pdf,svg,png,tiff}
    ├── fig2_se3r_without_calibration.{pdf,svg,png,tiff}
    ├── fig3_se3_original_with_calibration.{pdf,svg,png,tiff}
    └── fig4_se3r_with_calibration.{pdf,svg,png,tiff}
```

Heatmap cells are axis-aligned vector rectangles. This deliberately avoids the
diagonal white rendering artifacts that some PDF viewers produce when an
`imshow` raster is displayed at a fractional zoom level.

## Evidence boundary

The design tests controlled perturbation sensitivity and calibration mitigation.
It does not establish that the simulated random walk represents the complete
error distribution of a native VIO system. Relative percentages can also become
large when a matched baseline is close to zero; absolute RMSE values remain in
`case_metrics.csv` and should be checked before making a strong claim.
