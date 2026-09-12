# Calibration sensitivity experiment setup

## Material Passport

- Origin: local EPIC-Alignment experiment
- Type: controlled simulation on real GT motion templates
- Verification status: locally generated; scientific interpretation remains evidence-bounded
- Case count: 3080
- EPA checkout: PR34-plus checkout recorded in each result manifest; the main rerun used commit `c50102dfdf87ab330f56206b71f44287e1b9dabe`.

## Research question

How do known timestamp/extrinsic perturbations affect APE and ARE across motion
profiles, simulated VIO accuracies, and global-alignment methods, and how much of
that sensitivity remains after EPA calibration?

## Controlled inputs

Every trajectory is expressed relative to its first pose, cropped to 80 s, and
resampled to 10 Hz. Position uses linear interpolation; orientation uses quaternion
SLERP. No spatial scaling is applied. KITTI timestamps assume 10 Hz.

| Motion group | Sequence | 80-s path length | Bounding-box diagonal |
|---|---|---:|---:|
| Hot3D / small | `P0012_915e71c6` | 6.71 m | 0.64 m |
| Hot3D / small | `P0009_1b21bb01` | 8.94 m | 1.03 m |
| Hot3D / small | `P0011_451a7734` | 15.55 m | 1.64 m |
| EuRoC / indoor | `V1_01_easy_original` | 28.85 m | 6.90 m |
| EuRoC / indoor | `V1_02_medium` | 74.97 m | 8.11 m |
| EuRoC / indoor | `V2_02_medium` | 60.44 m | 7.50 m |
| KITTI / driving | `03` | 560.85 m | 513.27 m |
| KITTI / driving | `05` | 589.00 m | 277.94 m |
| KITTI / driving | `00` | 557.90 m | 399.76 m |
| EuRoC V2_03 / aggressive | `V2_03_difficult` | 60.96 m | 7.97 m |
| AEA / exercise | `loc1111` | 45.28 m | 5.71 m |

The Hot3D, EuRoC, and AEA ground-truth inputs are assumed to use
gravity-aligned world frames; KITTI does not require this assumption. The
simulator records this profile-level convention but does not re-estimate
gravity from IMU data during resampling.

## Simulated VIO generation

The estimate is generated from GT using a smoothed translation walk and a
component-wise rotation error:

`p_vio(t) = p_gt(t) + delta_p(t)`

`R_vio(t) = Delta_R_world(t) R_gt(t)`

Translation drift is in the common world frame. Roll and pitch are stationary,
non-accumulating smoothed Gaussian errors. Yaw combines a deterministic drift
rate with a cumulative smoothed random walk. The components are composed as
`Delta_R_world = Rz Ry Rx` and left-multiplied onto the GT pose. Rotation is
specified by component rates and therefore depends on the 80-s time horizon.
The medium yaw drift rate is 1 deg/s, with high and low set to 0.5x and 2x;
this is a controlled synthetic candidate setting, not a claim about native VIO
error. Translation targets are injected world-frame RMS values; post-alignment
APE is reported separately in Table 1.

The injected medium translation RMS targets are 0.030 m for Hot3D, 0.200 m for
indoor EuRoC, 0.330 m for EuRoC V2_03 aggressive motion, 0.220 m for AEA, and
2% of the normalized sequence path length for KITTI. These values are chosen
so that the post-position-alignment APE approaches the supervisor's target
ranges. Oracle, high, medium, and low use multipliers 0, 0.5, 1, and 2. The
component-level rotation and profile-specific translation settings are recorded
in `setup.md` and `config.json` for the exact run.

Five fixed seeds are used: 1101, 2202, 3303, 4404, 5505. Within
each `(sequence, accuracy, seed)` block, every calibration-error condition uses the
identical simulated VIO realization.

## Injected calibration errors

| Condition | Time offset | Extrinsic rotation | Extrinsic translation |
|---|---:|---:|---:|
| `none` | 0 ms | 0 deg | 0.00 m |
| `time_1ms` | 1 ms | 0 deg | 0.00 m |
| `time_5ms` | 5 ms | 0 deg | 0.00 m |
| `time_10ms` | 10 ms | 0 deg | 0.00 m |
| `time_20ms` | 20 ms | 0 deg | 0.00 m |
| `time_30ms` | 30 ms | 0 deg | 0.00 m |
| `time_50ms` | 50 ms | 0 deg | 0.00 m |
| `rotation_5deg` | 0 ms | 5 deg | 0.00 m |
| `rotation_20deg` | 0 ms | 20 deg | 0.00 m |
| `rotation_45deg` | 0 ms | 45 deg | 0.00 m |
| `translation_0p01m` | 0 ms | 0 deg | 0.01 m |
| `translation_0p03m` | 0 ms | 0 deg | 0.03 m |
| `translation_0p05m` | 0 ms | 0 deg | 0.05 m |
| `combined_realistic` | 10 ms | 20 deg | 0.05 m |

Rotation axis is normalized `[1, -2, 3]`; translation direction is normalized
`[1, -0.4, 0.2]`.

## Evaluation policies

The exact same injected trajectory is evaluated under four policies:

1. **Fig. 1, SE(3)-original without calibration:** determine global rotation and
   translation by position-only Kabsch alignment.
2. **Fig. 2, SE3R without calibration:** determine global rotation from paired
   orientations first, then solve translation with that rotation fixed.
3. **Fig. 3, SE(3)-original with calibration:** run the current EPA pipeline with
   `--mode se3-original`; EPA estimates time offset and extrinsics before a position-only
   Umeyama world alignment.
4. **Fig. 4, SE3R with calibration:** run the current EPA pipeline with
   `--mode se3`; EPA estimates time offset and extrinsics before an orientation-first
   world rotation and fixed-rotation translation.

The calibrated command uses `--dt-resample 0.001`,
`--offset-search-window-s 0.5`, `--t-max-diff 0.06`, the common evaluation
window `[0.1, 79.8]` s, `--no-downsample`, and `--no-plot`. Terminal-only mode
changes output generation, not calibration or metrics. The actual EPA overlap
count is retained per case because time-offset correction can crop support.

APE is translational RMSE in metres. ARE is orientation-angle RMSE in degrees.
Each heatmap cell is the mean over the available trajectories in its motion group and
five seeds. Relative change is calculated against the matched no-perturbation baseline
under the same alignment and calibration policy:

`100 * (RMSE_condition - RMSE_control) / RMSE_control`.

Oracle controls remain in `case_metrics.csv` and `simulated_vio_rmse.csv`, but are
omitted from relative heatmaps because a numerical-zero denominator is undefined.

For the main paper representation, the eight heatmaps are compressed into two
composite figures: `composite_ape` contains four panels for relative APE and
`composite_are` contains four panels for relative ARE. In each composite, the
columns compare SE(3)-original and SE3R, and the rows compare evaluation without
and with calibration. The no-trim run is retained as an ablation with the same
case generation and evaluation protocol.

## Figure contract

```json
{
  "core_conclusion": "EPA calibration should reduce sensitivity to timestamp/extrinsic perturbations; the residual depends on motion, VIO accuracy, and global-alignment method.",
  "results_level_question": "Where do known perturbations change trajectory-evaluation error, and how much of that sensitivity remains after EPA calibration?",
  "archetype": "quantitative grid",
  "backend": "Python/matplotlib",
  "final_size": "183 mm wide composite figures",
  "panels": {
    "composite_ape": "2 x 2 panels: alignment method by calibration state",
    "composite_are": "2 x 2 panels: alignment method by calibration state"
  },
  "replicate_unit": "trajectory; five deterministic random-walk seeds per trajectory",
  "center": "mean over available trajectories and five seeds",
  "spread": "SD is retained in source data; not encoded in heatmap color",
  "reviewer_risks": [
    "Synthetic random-walk errors are not native VIO failure distributions.",
    "Hot3D/EuRoC/KITTI have three trajectories per group; aggressive EuRoC/AEA have one.",
    "KITTI timestamps use an explicit 10 Hz assumption.",
    "Relative metrics can be unstable near zero; oracle controls are excluded.",
    "All panels within each composite use one shared color normalization."
  ]
}
```

## Evidence boundary

This experiment tests sensitivity to controlled synthetic perturbations and whether
the current EPA pipeline mitigates their effect. It does not show that the random-walk
model matches the full error distribution of a real VIO system. A lower calibrated
APE/ARE is also not, by itself, proof that every estimated calibration parameter equals
its injected truth; parameter-recovery evidence belongs in a separate table.
