# Error Metrics

This page defines the main error metrics reported by EPA and the
OpenVINS-compatible summaries.

## Full-Trajectory Metrics

Full-trajectory metrics are computed on the complete associated trajectory after
the selected synchronization, alignment, and evaluation settings are applied.

These metrics are the primary way to report overall sequence quality:

- `ATE`: absolute trajectory error, reported as rotation error in degrees and
  translation error in meters.
- `RPE`: relative pose error over pose pairs separated by a configured delta.
- `1S TIME RPE`: relative pose error over approximately one-second intervals.

Full-trajectory ATE should remain the primary ATE metric because it reflects
whether the whole sequence succeeded or failed.

## Distance RPE

Distance RPE compares relative motion over path-length segments such as `8 m`,
`16 m`, or `48 m`.

For each segment pair, EPA computes the relative motion in the reference
trajectory and the estimated trajectory, then reports the relative pose error.
Tables use:

```text
rotation error in degrees / translation error in meters
```

In OpenVINS-compatible `error_comparison` summaries, distance RPE currently uses
fixed segment lengths:

```text
8, 16, 24, 32, 40, 48 meters
```

These values are not automatically read from dataset metadata. If a
dataset-specific segment policy is later defined, the compatibility layer can be
extended to accept manual or automatic segment selection.

## Time RPE

Time RPE compares relative motion over fixed time intervals. EPA reports a
default one-second variant:

```text
1S TIME RPE
```

This answers the question: approximately how much relative drift occurs over
one second?

Tables use:

```text
rotation error in degrees / translation error in meters
```

## Distance Drift Rate

Distance drift rate normalizes distance-RPE translation error by the segment
length:

```text
mean(translation RPE / segment length) * 100
```

The unit is percent translation drift per traveled distance. For example, an
average `0.4 m` translation RPE over `8 m` corresponds to:

```text
0.4 / 8 * 100 = 5%
```

EPA reports distance drift rate as a separate table because its unit differs
from the standard RPE table.

## Drift-Valid Success Rate

Drift-valid success rate is the percentage of the reference path that is not
marked as local drift/fail:

```text
valid GT path length / total GT path length
```

A sequence with `0%` success rate has no drift-valid segments, so drift-valid
metrics are reported as `nan`.

## Drift-Valid Metrics

Drift-valid metrics are computed only on local segments that are not marked as
drift/fail.

They are intended to answer a different question from full-trajectory metrics:

- Full-trajectory metrics answer: how good was the whole sequence?
- Drift-valid metrics answer: how good were the non-failed local segments?

The two should be read together. Drift-valid metrics should not be reported
without the corresponding success rate because success rate explains how much of
the trajectory remained after excluding fail segments.

## Drift Detection

Local drift detection uses two signals:

- 1-second translation RPE
- positive APE growth/jump checks

A stable constant bias is not treated as drift by itself. For example, if a
trajectory stays consistently offset by `6 m` but its relative motion is stable,
that bias contributes to ATE but does not by itself create a drift/fail segment.

EPA also applies a global failed-case gate. If the 5th percentile final-aligned APE
translation error is above `30 m`, the case is treated as globally failed and
drift-valid metrics are reported as `nan`.

EPA still estimates and records a per-case APE knee threshold, clamped to
`5-30 m` by default, as tolerance metadata for success-rate reporting and
diagnostics.

## Reading the Tables

OpenVINS-compatible `error_comparison` output contains both full-trajectory and
drift-valid-only tables:

- `FULL TRAJECTORY ATE`: full-sequence absolute error.
- `FULL TRAJECTORY DISTANCE RPE`: full-sequence distance RPE.
- `FULL TRAJECTORY DISTANCE DRIFT RATE`: full-sequence normalized translation
  drift rate.
- `FULL TRAJECTORY 1S TIME RPE`: full-sequence one-second time RPE.
- `DRIFT-VALID SUCCESS RATE`: how much path length remained valid.
- `DRIFT-VALID ONLY ...`: metrics computed only on non-fail local segments.

Use full-trajectory tables for overall sequence success and comparability. Use
drift-valid-only RPE and drift-rate tables to inspect local performance after
excluding detected fail segments.

Example `error_comparison se3` output:

```text
[COMP]: 1627 poses in R_11_5cp.txt => length of 475.20 meters
[COMP]: 3395 poses in R_12_10cp.txt => length of 1014.17 meters
[COMP]: 4382 poses in R_13_15cp.txt => length of 1172.17 meters
======================================
[COMP]: processing svo_mono algorithm
[COMP]: processing svo_mono algorithm => R_11_5cp dataset
        ATE: mean_ori = 8.020 | mean_pos = 21.163 (1 runs)
        eval_source: epa=1, ov_eval=0
        RPE: seg 8 - median_ori = 2.5953 | median_pos = 2.1285 (1404 samples)
        RPE: seg 16 - median_ori = 2.7115 | median_pos = 2.1668 (1295 samples)
        RPE: seg 24 - median_ori = 2.7209 | median_pos = 2.1082 (1230 samples)
        RPE: seg 32 - median_ori = 2.7569 | median_pos = 2.5895 (1154 samples)
        RPE: seg 40 - median_ori = 2.6837 | median_pos = 2.5001 (1065 samples)
        RPE: seg 48 - median_ori = 2.6756 | median_pos = 2.4689 (1002 samples)
        RPE time 1s - mean_ori = 3.350 | mean_pos = 2.223 (1122 samples)
        SR - distance = 53.36% | time = 49.65%
[COMP]: processing svo_mono algorithm => R_12_10cp dataset
        ATE: mean_ori = 23.106 | mean_pos = 1737.041 (1 runs)
        eval_source: epa=1, ov_eval=0
        RPE: seg 8 - median_ori = 0.5687 | median_pos = 7.8639 (155 samples)
        RPE: seg 16 - median_ori = 0.5605 | median_pos = 15.7360 (291 samples)
        RPE: seg 24 - median_ori = 0.8315 | median_pos = 23.2249 (251 samples)
        RPE: seg 32 - median_ori = 0.9508 | median_pos = 29.8896 (242 samples)
        RPE: seg 40 - median_ori = 1.0295 | median_pos = 36.9502 (236 samples)
        RPE: seg 48 - median_ori = 1.2350 | median_pos = 37.4510 (216 samples)
        RPE time 1s - mean_ori = 2.907 | mean_pos = 122.646 (2375 samples)
        SR - distance = 0.00% | time = 0.00%
[COMP]: processing svo_mono algorithm => R_13_15cp dataset
        ATE: mean_ori = 9.567 | mean_pos = 29.527 (1 runs)
        eval_source: epa=1, ov_eval=0
        RPE: seg 8 - median_ori = 1.3766 | median_pos = 1.3372 (3943 samples)
        RPE: seg 16 - median_ori = 1.4801 | median_pos = 1.5878 (3751 samples)
        RPE: seg 24 - median_ori = 1.5137 | median_pos = 1.8424 (3603 samples)
        RPE: seg 32 - median_ori = 1.5153 | median_pos = 1.9950 (3522 samples)
        RPE: seg 40 - median_ori = 1.5133 | median_pos = 2.2573 (3439 samples)
        RPE: seg 48 - median_ori = 1.5520 | median_pos = 2.3947 (3400 samples)
        RPE time 1s - mean_ori = 2.222 | mean_pos = 3.059 (3036 samples)
        SR - distance = 71.28% | time = 69.38%
============================================
TOOL SOURCE: epa=3, ov_eval=0
============================================
============================================
FULL TRAJECTORY ATE LATEX TABLE (ROT DEG / TRANS M)
============================================
 & \textbf{R\_11\_5cp} & \textbf{R\_12\_10cp} & \textbf{R\_13\_15cp} & \textbf{Average} \\hline
svo\_mono & 8.020 / 21.163 & 23.106 / 1737.041 & 9.567 / 29.527 & 13.564 / 595.910 \\
============================================
============================================
FULL TRAJECTORY DISTANCE RPE LATEX TABLE (ROT DEG / TRANS M)
============================================
 & \textbf{8m} & \textbf{16m} & \textbf{24m} & \textbf{32m} & \textbf{40m} & \textbf{48m} \\hline
svo\_mono & 2.378 / 1.867 & 2.603 / 2.669 & 2.774 / 3.051 & 2.831 / 3.605 & 2.845 / 4.068 & 2.938 / 4.353 \\
============================================
============================================
FULL TRAJECTORY DISTANCE DRIFT RATE LATEX TABLE (% TRANS / DIST)
============================================
 & \textbf{8m} & \textbf{16m} & \textbf{24m} & \textbf{32m} & \textbf{40m} & \textbf{48m} \\hline
svo\_mono & 23.334 & 16.684 & 12.714 & 11.266 & 10.171 & 9.069 \\
============================================
============================================
FULL TRAJECTORY 1S TIME RPE LATEX TABLE (ROT DEG / TRANS M)
============================================
 & \textbf{R\_11\_5cp} & \textbf{R\_12\_10cp} & \textbf{R\_13\_15cp} & \textbf{Average} \\hline
svo\_mono & 3.350 / 2.223 & 2.907 / 122.646 & 2.222 / 3.059 & 2.826 / 42.643 \\
============================================
============================================
DRIFT-VALID SUCCESS RATE LATEX TABLE (% PATH LENGTH)
============================================
 & \textbf{R\_11\_5cp} & \textbf{R\_12\_10cp} & \textbf{R\_13\_15cp} & \textbf{Average} \\hline
svo\_mono & 53.36 & 0.00 & 71.28 & 41.55 \\
============================================
============================================
DRIFT-VALID ONLY ATE LATEX TABLE (ROT DEG / TRANS M)
============================================
 & \textbf{R\_11\_5cp} & \textbf{R\_12\_10cp} & \textbf{R\_13\_15cp} & \textbf{Average} \\hline
svo\_mono & 6.566 / 16.008 & nan / nan & 9.621 / 21.290 & 8.093 / 18.649 \\
============================================
============================================
DRIFT-VALID ONLY DISTANCE RPE LATEX TABLE (ROT DEG / TRANS M)
============================================
 & \textbf{8m} & \textbf{16m} & \textbf{24m} & \textbf{32m} & \textbf{40m} & \textbf{48m} \\hline
svo\_mono & 1.577 / 1.315 & 1.653 / 1.746 & 1.342 / 2.041 & 1.393 / 2.239 & 1.718 / 2.467 & 1.422 / 2.755 \\
============================================
============================================
DRIFT-VALID ONLY DISTANCE DRIFT RATE LATEX TABLE (% TRANS / DIST)
============================================
 & \textbf{8m} & \textbf{16m} & \textbf{24m} & \textbf{32m} & \textbf{40m} & \textbf{48m} \\hline
svo\_mono & 16.440 & 10.914 & 8.504 & 6.996 & 6.167 & 5.739 \\
============================================
============================================
DRIFT-VALID ONLY 1S TIME RPE LATEX TABLE (ROT DEG / TRANS M)
============================================
 & \textbf{R\_11\_5cp} & \textbf{R\_12\_10cp} & \textbf{R\_13\_15cp} & \textbf{Average} \\hline
svo\_mono & 2.627 / 1.085 & nan / nan & 1.365 / 1.092 & 1.996 / 1.089 \\
============================================
```

Key observations from this example:

- `TOOL SOURCE: epa=3, ov_eval=0` means all three sequences were evaluated by
  EPA final alignment; none fell back to ov_eval-style SE3.
- `R_12_10cp` has full-trajectory ATE translation error `1737.041 m` and
  one-second time RPE translation error `122.646 m`, so it is a severe failure
  case.
- `DRIFT-VALID SUCCESS RATE` is `0.00%` for `R_12_10cp`, so its drift-valid-only
  metrics are `nan / nan`.
- `R_11_5cp` and `R_13_15cp` retain `53.36%` and `71.28%` of their path length,
  respectively, for drift-valid-only metrics.
- Full-trajectory tables show overall sequence quality, including fail segments.
  Drift-valid-only tables show local performance after detected fail segments
  are excluded.
- Distance drift rate normalizes translation RPE by segment length. For example,
  full-trajectory `8m` drift rate `23.334%` means the average 8-meter translation
  RPE is about `23.334%` of the segment length.
