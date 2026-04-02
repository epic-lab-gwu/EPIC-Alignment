# vicon_ws

Minimal workspace for a 3-step trajectory alignment and extrinsic calibration pipeline.

## What It Does

`extrinsic.py` aligns two trajectories in three stages:

1. **Time alignment** using angular-velocity norm cross-correlation.
2. **Sensor extrinsic calibration** between two frames (`R_ext`, `t_ext`) from relative motion.
3. **World-frame alignment** (`R_w`, `t_w`) via rigid registration.

It also prints error metrics for:

- Time offset quality
- Step-2 residual consistency
- Trajectory-level ATE before/after alignment

## Files

- `extrinsic.py`: main pipeline script
- `data.csv`: input trajectory CSV (EuRoC-style columns)
- `track.md`: detailed Chinese walkthrough and metric interpretation

## Quick Start

1. Before run `extrinsic.py`, make sure `data.csv` in the same directory as the script.

2. Run `python3 extrinsic.py`

## Output

Each run creates a new timestamped folder:

`outputs/run_YYYYMMDD_HHMMSS/`

Saved artifacts include:

- `step1_cross_correlation.png`: cross-correlation curve and estimated time offset peak
- `step1_time_alignment.png`: before/after time alignment plots
- `step23_trajectory_alignment_3d.png`: 3D trajectory comparison (raw / step2 / step3)
- `metrics.json`: full metrics + estimated parameters + metadata
- `metrics_summary.csv`: flat scalar metric table for quick comparison across runs

### Example

Example folder:

`outputs/run_20260402_172740/`

Visualizations:

![Step 1 Cross Correlation](outputs/run_20260402_172740/step1_cross_correlation.png)
![Step 1 Time Alignment](outputs/run_20260402_172740/step1_time_alignment.png)
![Step 2/3 Trajectory Alignment](outputs/run_20260402_172740/step23_trajectory_alignment_3d.png)

Example metrics (`metrics.json`, excerpt):

```json
{
  "time_alignment": {
    "offset_est_s": 0.123,
    "xcorr_peak_normalized": 0.9985378085,
    "omega_rmse_improve_pct": 98.2061364654
  },
  "step2_residuals": {
    "translation_system_cond": 1.8796175517,
    "translation_constraints": 24182.0
  },
  "trajectory": {
    "ate_rmse_raw_m": 1.5147407977,
    "ate_rmse_step3_m": 6.4412706821e-15
  }
}
```

Example metrics (`metrics_summary.csv`, first rows):

```csv
section,metric,value
time_alignment,offset_est_s,0.123
time_alignment,xcorr_peak_normalized,0.9985378085071166
time_alignment,omega_rmse_improve_pct,98.20613646539792
step2_residuals,translation_system_cond,1.87961755166322
step2_residuals,translation_constraints,24182.0
trajectory,ate_rmse_raw_m,1.5147407976987484
trajectory,ate_rmse_step3_m,6.441270682070819e-15
```
