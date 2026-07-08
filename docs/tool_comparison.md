# Tool Comparison

This page summarizes a direct benchmark study in which `epica` and `evo` were evaluated separately on 139 cases from the `euroc_mav`, `grand_tour`, `lamaria`, and `uzh_fpv` datasets.


## Overall Comparison

| Metric | epica | evo |
|---|---:|---:|
| Successful cases | 139 | 139 |
| Non-critical comparable cases | 103 | 103 |
| Mean aligned RMSE (m) | 2.134 | 5.339 |
| Median aligned RMSE (m) | 0.809 | 4.482 |
| Case wins | 69 | 3 |
| Win rate | 66.99% | 2.91% |
| Ties | 31 | 31 |

Interpretation:

- The mean and median RMSE values above are computed only on non-critical comparable cases.
- The win counts above follow the same composite winner rule used in the per-case table below.
- `epica` has a much lower median aligned RMSE.
- Excluding critical cases removes the heavy-tailed failure regime and gives a clearer view of normal-case performance.

<img src="../images/benchmark_study_aligned_rmse_scatter.png" alt="Aligned RMSE scatter" style="width:72%; display:block; margin:0 auto;">

*Aligned RMSE scatter on non-critical comparable cases. Points below the diagonal favor `epica`.*

<img src="../images/benchmark_study_aligned_rmse_ecdf.png" alt="Aligned RMSE ECDF" style="width:72%; display:block; margin:0 auto;">

*Aligned RMSE ECDF on non-critical comparable cases. A left-shifted curve indicates better overall error distribution.*

Notes:

- `critical`: cases whose EPICA runs are flagged as unreliable in the current reporting logic, typically due to severe scale mismatch, very high final-alignment error, or other strong failure indicators.
- `non-critical`: comparable cases that are not labeled `critical`; this includes both `ok` and `warning` cases and is used here to focus on normal-case performance rather than extreme failure tails.

Winner rule:

- If the aligned RMSE gap is larger than 5%, the lower aligned RMSE wins.
- If aligned RMSE is close, the higher improvement percentage wins when the gap is larger than 5 percentage points.
- If both are still close, the side with more matched pairs wins when the gap is meaningful.
- Otherwise the result is marked as `tie`.

## Per-Dataset Comparison

| Dataset | Non-critical cases | Critical cases | Mean RMSE<br>`epica / evo` (m, ↓) | Median RMSE<br>`epica / evo` (m, ↓) | Wins<br>`epica / evo` (↑) |
|---|---:|---:|---:|---:|---:|
| `euroc_mav` | 21 | 1 | 0.197 / 1.971 | 0.156 / 0.374 | 14 / 0 |
| `grand_tour` | 52 | 16 | 3.537 / 5.073 | 2.576 / 3.937 | 26 / 2 |
| `uzh_fpv` | 30 | 0 | 1.058 / 8.159 | 0.640 / 6.712 | 29 / 1 |
| `lamaria` | 0 | 19 | N/A | N/A | N/A |

Interpretation:

- Our toolkit `epica` is stronger on all three non-critical dataset slices: `uzh_fpv`, `grand_tour`, and `euroc_mav`.
- Dataset `lamaria` is excluded from the non-critical comparison because all comparable cases in the current run are classified as critical.
- Ties are omitted from the dataset win column; in the current run they are `7` for `euroc_mav`, `24` for `grand_tour`, and `0` for `uzh_fpv`.

<img src="../images/benchmark_study_dataset_boxplot.png" alt="Per-dataset boxplot" style="width:72%; display:block; margin:0 auto;">

*Per-dataset aligned RMSE distribution for non-critical comparable cases.*

<img src="../images/benchmark_study_win_tie_loss_stacked.png" alt="Win tie loss stacked bar" style="width:72%; display:block; margin:0 auto;">

*Win / tie / loss counts under the benchmark-study winner rule.*

## Per-Case Comparison

<img src="../images/benchmark_study_improvement_box.png" alt="Improvement boxplot" style="width:72%; display:block; margin:0 auto;">

*Improvement percentage distribution, added as an extra view for quick comparison.*

The following table shows per-case benchmark comparison results for non-critical cases only.

| case (103 total) | Alert Signal (EPA) | Aligned RMSE `epica / evo` (m, ↓) | Improvement `epica / evo` (%, ↑) | Time Offset `epica / evo` (s, N/A) | Matched Pairs `epica / evo` (↑) | Winner |
|---|---|---:|---:|---:|---:|---|
| euroc_mav_MH_01_easy_rovio | ok | 0.266 / 0.268 | 95.52 / 95.48 | -0.002 / 0.050 | 3638 / 3639 | tie |
| euroc_mav_MH_03_medium_rovio | ok | 0.337 / 0.374 | 94.23 / 93.60 | -0.001 / 0.150 | 2631 / 2631 | epica |
| euroc_mav_MH_04_difficult_rovio | warning | 0.608 / 6.565 | 96.25 / 55.34 | -0.003 / -20.000 | 1976 / 1597 | epica |
| euroc_mav_MH_05_difficult_rovio | warning | 0.568 / 7.280 | 96.63 / 52.10 | -0.001 / 20.000 | 2222 / 1849 | epica |
| euroc_mav_V1_01_easy_rovio | ok | 0.137 / 2.192 | 96.64 / 30.03 | -0.002 / -15.300 | 2894 / 2604 | epica |
| euroc_mav_V1_02_medium_rovio | ok | 0.156 / 1.885 | 95.51 / 16.16 | -0.002 / -12.500 | 1671 / 1438 | epica |
| euroc_mav_V1_03_difficult_rovio | ok | 0.136 / 1.767 | 95.91 / 28.63 | -0.002 / -20.000 | 2094 / 1712 | epica |
| euroc_mav_V2_01_easy_rovio | ok | 0.259 / 0.260 | 86.65 / 86.62 | -0.001 / 0.050 | 2240 / 2240 | tie |
| euroc_mav_V2_02_medium_rovio | ok | 0.272 / 0.275 | 85.74 / 85.62 | -0.001 / 0.000 | 2310 / 2310 | tie |
| euroc_mav_V2_03_difficult_rovio | ok | 0.164 / 0.172 | 90.51 / 90.04 | -0.004 / 0.050 | 1890 / 1890 | tie |
| euroc_mav_MH_01_easy_svo_stereo | ok | 0.129 / 0.152 | 97.91 / 97.54 | 0.000 / -0.150 | 3621 / 3624 | epica |
| euroc_mav_MH_02_easy_svo_stereo | ok | 0.064 / 0.077 | 98.89 / 98.68 | 0.000 / -0.050 | 2999 / 3000 | epica |
| euroc_mav_MH_03_medium_svo_stereo | ok | 0.173 / 0.245 | 97.05 / 95.81 | 0.002 / 0.150 | 2624 / 2621 | epica |
| euroc_mav_MH_04_difficult_svo_stereo | ok | 0.152 / 6.688 | 99.02 / 51.72 | 0.000 / -20.000 | 1953 / 1599 | epica |
| euroc_mav_MH_05_difficult_svo_stereo | ok | 0.172 / 6.960 | 98.95 / 52.60 | 0.002 / -20.000 | 2192 / 1845 | epica |
| euroc_mav_V1_01_easy_svo_stereo | ok | 0.041 / 2.226 | 98.97 / 27.18 | 0.000 / -15.200 | 2874 / 2608 | epica |
| euroc_mav_V1_02_medium_svo_stereo | ok | 0.077 / 1.873 | 97.79 / 13.17 | 0.000 / -12.450 | 1671 / 1441 | epica |
| euroc_mav_V1_03_difficult_svo_stereo | ok | 0.085 / 1.776 | 97.40 / 24.99 | 0.000 / -20.000 | 2076 / 1713 | epica |
| euroc_mav_V2_01_easy_svo_stereo | ok | 0.055 / 0.052 | 96.88 / 97.02 | 0.001 / 0.000 | 2208 / 2208 | tie |
| euroc_mav_V2_02_medium_svo_stereo | ok | 0.157 / 0.158 | 90.70 / 90.61 | 0.001 / 0.000 | 2277 / 2277 | tie |
| euroc_mav_V2_03_difficult_svo_stereo | ok | 0.134 / 0.140 | 91.95 / 91.55 | -0.001 / 0.050 | 1857 / 1856 | tie |
| grand_tour_2024-10-01-11-29-55_rovio | warning | 7.572 / 12.375 | 92.01 / 86.88 | 0.000 / 20.000 | 3470 / 3280 | epica |
| grand_tour_2024-10-01-12-00-49_rovio | warning | 6.540 / 7.399 | 58.70 / 51.49 | 0.003 / 10.350 | 4761 / 4665 | epica |
| grand_tour_2024-11-02-17-10-25_rovio | warning | 1.766 / 1.757 | 78.29 / 78.41 | -0.005 / 0.400 | 1785 / 1789 | tie |
| grand_tour_2024-11-02-17-43-10_rovio | warning | 2.821 / 2.928 | 74.43 / 73.48 | 0.001 / -1.850 | 2267 / 2249 | tie |
| grand_tour_2024-11-03-07-52-45_rovio | warning | 2.972 / 3.029 | 88.30 / 88.07 | 0.004 / 0.000 | 1848 / 1848 | tie |
| grand_tour_2024-11-03-13-59-54_rovio | warning | 6.955 / 8.055 | 52.85 / 42.31 | -0.005 / -9.150 | 3614 / 3523 | epica |
| grand_tour_2024-11-04-13-07-13_rovio | warning | 3.602 / 4.542 | 85.41 / 81.38 | -0.004 / 18.400 | 819 / 636 | epica |
| grand_tour_2024-11-04-16-05-00_rovio | warning | 9.322 / 11.945 | 66.93 / 53.74 | -0.001 / 20.000 | 2060 / 1860 | epica |
| grand_tour_2024-11-11-14-29-44_rovio | warning | 8.898 / 9.004 | 88.86 / 88.74 | 0.000 / 0.750 | 3532 / 3532 | tie |
| grand_tour_2024-11-14-12-01-26_rovio | warning | 9.156 / 9.851 | 56.31 / 50.84 | 0.000 / 19.100 | 4588 / 4399 | epica |
| grand_tour_2024-11-14-14-36-02_rovio | warning | 9.137 / 9.137 | 69.68 / 69.71 | 0.001 / -0.050 | 4787 / 4787 | tie |
| grand_tour_2024-11-14-15-22-43_rovio | warning | 6.132 / 6.127 | 90.45 / 90.47 | 0.000 / -0.100 | 3970 / 3970 | tie |
| grand_tour_2024-11-15-10-16-35_rovio | warning | 9.803 / 9.811 | 73.52 / 73.54 | -0.003 / 1.400 | 4761 / 4761 | tie |
| grand_tour_2024-11-15-12-06-03_rovio | warning | 2.299 / 2.455 | 29.49 / 20.36 | 0.000 / -3.600 | 2225 / 2189 | epica |
| grand_tour_2024-11-15-14-43-52_rovio | warning | 3.240 / 3.230 | 94.76 / 94.77 | -0.004 / -0.050 | 2487 / 2487 | tie |
| grand_tour_2024-11-18-12-05-01_rovio | warning | 9.308 / 9.436 | 87.38 / 87.21 | -0.003 / 1.000 | 4461 / 4461 | tie |
| grand_tour_2024-11-18-15-46-05_rovio | warning | 3.927 / 3.925 | 36.38 / 36.39 | 0.000 / 0.000 | 4103 / 4103 | tie |
| grand_tour_2024-11-18-16-59-23_rovio | warning | 5.795 / 8.764 | 79.87 / 66.90 | 0.003 / 20.000 | 4042 / 3850 | epica |
| grand_tour_2024-11-25-14-57-08_rovio | warning | 3.339 / 3.343 | 48.74 / 48.72 | 0.003 / -0.200 | 3238 / 3236 | tie |
| grand_tour_2024-11-25-16-36-19_rovio | warning | 7.862 / 7.867 | 73.61 / 73.60 | 0.007 / 0.000 | 4354 / 4354 | tie |
| grand_tour_2024-12-03-13-26-40_rovio | warning | 5.873 / 7.648 | 75.99 / 68.03 | 0.006 / -11.750 | 2845 / 2728 | epica |
| grand_tour_2024-10-01-11-29-55_svo_stereo | warning | 0.809 / 10.454 | 99.17 / 89.18 | 0.002 / 20.000 | 3451 / 3260 | epica |
| grand_tour_2024-11-02-17-10-25_svo_stereo | ok | 0.129 / 0.450 | 98.28 / 93.97 | -0.002 / -1.350 | 1786 / 1772 | epica |
| grand_tour_2024-11-02-17-43-10_svo_stereo | warning | 0.745 / 5.678 | 95.43 / 64.67 | 0.002 / -17.650 | 2250 / 2092 | epica |
| grand_tour_2024-11-02-21-12-51_svo_stereo | warning | 3.556 / 3.634 | 47.65 / 46.47 | -0.002 / -1.250 | 2633 / 2633 | tie |
| grand_tour_2024-11-03-07-52-45_svo_stereo | ok | 0.150 / 0.285 | 94.10 / 88.77 | 0.003 / 0.850 | 1829 / 1829 | epica |
| grand_tour_2024-11-03-08-17-23_svo_stereo | warning | 1.334 / 6.811 | 91.62 / 51.51 | 0.005 / -15.100 | 2301 / 2167 | epica |
| grand_tour_2024-11-03-13-59-54_svo_stereo | ok | 0.300 / 0.905 | 92.90 / 78.10 | -0.001 / -1.550 | 3606 / 3600 | epica |
| grand_tour_2024-11-04-10-57-34_svo_stereo | warning | 0.401 / 0.487 | 92.51 / 90.92 | 0.001 / 0.400 | 4686 / 4686 | epica |
| grand_tour_2024-11-04-12-55-59_svo_stereo | warning | 1.203 / 1.205 | 98.28 / 98.27 | 0.005 / -0.450 | 5195 / 5195 | tie |
| grand_tour_2024-11-04-13-07-13_svo_stereo | warning | 0.555 / 4.699 | 98.18 / 83.94 | -0.002 / 20.000 | 805 / 608 | epica |
| grand_tour_2024-11-04-16-05-00_svo_stereo | warning | 0.429 / 13.201 | 99.04 / 69.58 | 0.002 / -20.000 | 2056 / 2060 | epica |
| grand_tour_2024-11-11-12-07-40_svo_stereo | warning | 2.444 / 2.500 | 97.19 / 97.13 | -0.003 / 0.700 | 9593 / 9593 | tie |
| grand_tour_2024-11-11-12-42-47_svo_stereo | warning | 2.308 / 2.527 | 96.04 / 95.67 | 0.001 / -0.900 | 4092 / 4092 | epica |
| grand_tour_2024-11-11-14-29-44_svo_stereo | warning | 1.509 / 1.533 | 98.25 / 98.23 | 0.002 / -1.050 | 3523 / 3523 | tie |
| grand_tour_2024-11-14-12-01-26_svo_stereo | warning | 3.929 / 3.948 | 89.17 / 89.12 | 0.001 / 0.150 | 4585 / 4585 | tie |
| grand_tour_2024-11-14-14-36-02_svo_stereo | warning | 2.707 / 2.728 | 91.11 / 91.05 | 0.003 / -0.250 | 4786 / 4786 | tie |
| grand_tour_2024-11-14-15-22-43_svo_stereo | warning | 1.357 / 1.436 | 98.17 / 98.07 | 0.003 / -0.700 | 3965 / 3965 | epica |
| grand_tour_2024-11-14-16-04-09_svo_stereo | warning | 1.217 / 14.176 | 98.54 / 82.87 | 0.009 / -19.950 | 4882 / 4687 | epica |
| grand_tour_2024-11-15-10-16-35_svo_stereo | warning | 0.842 / 0.839 | 98.09 / 98.10 | 0.000 / -0.200 | 4731 / 4731 | tie |
| grand_tour_2024-11-15-11-18-14_svo_stereo | warning | 0.794 / 0.839 | 97.93 / 97.81 | 0.000 / 0.550 | 2942 / 2942 | epica |
| grand_tour_2024-11-15-11-37-15_svo_stereo | warning | 1.152 / 1.077 | 95.46 / 95.76 | 0.001 / 0.300 | 5217 / 5217 | evo |
| grand_tour_2024-11-15-12-06-03_svo_stereo | warning | 1.001 / 1.313 | 68.78 / 56.90 | 0.002 / -2.900 | 2209 / 2197 | epica |
| grand_tour_2024-11-15-14-14-12_svo_stereo | warning | 0.675 / 7.734 | 98.80 / 86.16 | 0.006 / -20.000 | 5415 / 5215 | epica |
| grand_tour_2024-11-15-14-43-52_svo_stereo | warning | 6.038 / 5.757 | 90.24 / 90.70 | -0.002 / -1.650 | 2472 / 2472 | tie |
| grand_tour_2024-11-15-16-41-14_svo_stereo | warning | 3.130 / 3.103 | 91.37 / 91.45 | 0.025 / 0.050 | 5426 / 5427 | tie |
| grand_tour_2024-11-18-12-05-01_svo_stereo | warning | 6.518 / 6.630 | 88.56 / 88.37 | -0.009 / 0.900 | 4456 / 4456 | tie |
| grand_tour_2024-11-18-16-59-23_svo_stereo | warning | 7.309 / 7.406 | 62.50 / 62.00 | 0.006 / 1.450 | 4038 / 4030 | tie |
| grand_tour_2024-11-25-14-57-08_svo_stereo | warning | 1.300 / 1.293 | 17.37 / 17.16 | 0.004 / -0.350 | 3227 / 3227 | tie |
| grand_tour_2024-11-25-16-36-19_svo_stereo | warning | 1.077 / 1.134 | 97.67 / 97.55 | 0.009 / -0.600 | 4349 / 4349 | epica |
| grand_tour_2024-12-03-13-15-38_svo_stereo | warning | 2.051 / 1.878 | 93.90 / 94.42 | 0.006 / 0.750 | 5152 / 5152 | evo |
| grand_tour_2024-12-03-13-26-40_svo_stereo | warning | 0.641 / 5.499 | 97.35 / 76.83 | 0.008 / -9.850 | 2834 / 2748 | epica |
| uzh_fpv_indoor_forward_10_snapdragon_with_gt_rovio | warning | 0.365 / 7.661 | 98.34 / 50.27 | 0.000 / 20.000 | 820 / 836 | epica |
| uzh_fpv_indoor_forward_3_snapdragon_with_gt_rovio | warning | 0.629 / 6.989 | 97.20 / 60.41 | -0.015 / 18.700 | 1333 / 1382 | epica |
| uzh_fpv_indoor_forward_5_snapdragon_with_gt_rovio | warning | 0.407 / 6.114 | 98.08 / 54.38 | -0.004 / 8.650 | 530 / 542 | epica |
| uzh_fpv_indoor_forward_6_snapdragon_with_gt_rovio | warning | 0.440 / 8.153 | 98.13 / 53.23 | -0.015 / 20.000 | 817 / 881 | epica |
| uzh_fpv_indoor_forward_7_snapdragon_with_gt_rovio | warning | 1.175 / 8.096 | 94.89 / 59.57 | -0.015 / 20.000 | 1735 / 1752 | epica |
| uzh_fpv_indoor_forward_9_snapdragon_with_gt_rovio | warning | 0.372 / 6.841 | 97.87 / 49.45 | -0.008 / 20.000 | 767 / 802 | epica |
| uzh_fpv_indoor_forward_10_snapdragon_with_gt_svo_stereo | warning | 1.692 / 7.384 | 91.45 / 50.03 | 0.000 / 20.000 | 820 / 836 | epica |
| uzh_fpv_indoor_forward_3_snapdragon_with_gt_svo_stereo | ok | 0.316 / 7.270 | 98.60 / 59.20 | 0.000 / 18.550 | 1333 / 1382 | epica |
| uzh_fpv_indoor_forward_5_snapdragon_with_gt_svo_stereo | warning | 1.298 / 6.031 | 93.27 / 55.03 | 0.000 / 8.150 | 530 / 542 | epica |
| uzh_fpv_indoor_forward_6_snapdragon_with_gt_svo_stereo | warning | 0.705 / 7.741 | 96.83 / 54.61 | -0.002 / 15.750 | 817 / 874 | epica |
| uzh_fpv_indoor_forward_7_snapdragon_with_gt_svo_stereo | warning | 0.540 / 7.686 | 97.55 / 59.92 | -0.001 / 20.000 | 1736 / 1752 | epica |
| uzh_fpv_indoor_forward_9_snapdragon_with_gt_svo_stereo | warning | 1.696 / 4.920 | 89.04 / 63.37 | -0.002 / 5.000 | 767 / 783 | epica |
| uzh_fpv_indoor_45_12_snapdragon_with_gt_rovio | warning | 9.237 / 6.582 | 50.64 / 51.75 | -0.012 / 15.150 | 1077 / 1112 | evo |
| uzh_fpv_indoor_45_13_snapdragon_with_gt_rovio | warning | 0.329 / 2.467 | 96.87 / 75.47 | -0.015 / -0.650 | 1037 / 1038 | epica |
| uzh_fpv_indoor_45_14_snapdragon_with_gt_rovio | warning | 0.529 / 2.830 | 95.34 / 73.56 | -0.016 / -0.550 | 1026 / 1028 | epica |
| uzh_fpv_indoor_45_2_snapdragon_with_gt_rovio | warning | 0.532 / 5.726 | 96.52 / 58.21 | -0.016 / -1.950 | 1265 / 1258 | epica |
| uzh_fpv_indoor_45_4_snapdragon_with_gt_rovio | warning | 0.452 / 5.348 | 96.91 / 59.91 | -0.007 / -1.800 | 1154 / 1155 | epica |
| uzh_fpv_indoor_45_12_snapdragon_with_gt_svo_stereo | warning | 0.732 / 2.253 | 92.86 / 77.03 | 0.000 / -1.000 | 1077 / 1076 | epica |
| uzh_fpv_indoor_45_13_snapdragon_with_gt_svo_stereo | warning | 0.651 / 2.296 | 93.65 / 76.79 | 0.000 / -0.600 | 1037 / 1038 | epica |
| uzh_fpv_indoor_45_14_snapdragon_with_gt_svo_stereo | warning | 1.086 / 2.559 | 89.86 / 75.15 | 0.000 / -0.450 | 1026 / 1028 | epica |
| uzh_fpv_indoor_45_2_snapdragon_with_gt_svo_stereo | warning | 0.510 / 4.482 | 96.36 / 65.49 | -0.001 / -1.400 | 1265 / 1260 | epica |
| uzh_fpv_indoor_45_4_snapdragon_with_gt_svo_stereo | warning | 0.453 / 4.785 | 96.85 / 64.20 | 0.002 / -1.600 | 1154 / 1155 | epica |
| uzh_fpv_outdoor_forward_1_snapdragon_with_gt_rovio | warning | 0.726 / 16.105 | 98.45 / 54.50 | 0.004 / 14.450 | 1165 / 1188 | epica |
| uzh_fpv_outdoor_forward_3_snapdragon_with_gt_rovio | warning | 0.732 / 18.317 | 98.75 / 64.48 | -0.003 / 15.550 | 2224 / 2275 | epica |
| uzh_fpv_outdoor_forward_5_snapdragon_with_gt_rovio | warning | 0.905 / 23.067 | 98.68 / 25.80 | 0.000 / 8.350 | 446 / 464 | epica |
| uzh_fpv_outdoor_forward_1_snapdragon_with_gt_svo_stereo | warning | 1.235 / 15.678 | 97.21 / 53.79 | 0.010 / 13.650 | 1166 / 1185 | epica |
| uzh_fpv_outdoor_forward_3_snapdragon_with_gt_svo_stereo | warning | 1.856 / 18.603 | 96.81 / 63.50 | 0.000 / 15.550 | 2223 / 2274 | epica |
| uzh_fpv_outdoor_forward_5_snapdragon_with_gt_svo_stereo | warning | 1.366 / 22.841 | 97.92 / 25.57 | 0.001 / 8.100 | 446 / 464 | epica |
| uzh_fpv_outdoor_45_1_snapdragon_with_gt_rovio | warning | 0.494 / 2.487 | 96.10 / 80.30 | 0.000 / -0.350 | 507 / 504 | epica |
| uzh_fpv_outdoor_45_1_snapdragon_with_gt_svo_stereo | ok | 0.273 / 3.456 | 98.13 / 75.37 | 0.000 / -0.500 | 498 / 493 | epica |

## Critical Cases

The following table lists cases that are classified as `critical` by the EPICA alert logic. These cases are kept separate from the main per-case comparison because they belong to the heavy-tail failure regime. Their winner field is marked as `n/a`.

| case (36 total) | Alert Signal (EPA) | Aligned RMSE `epica / evo` (m, ↓) | Improvement `epica / evo` (%, ↑) | Time Offset `epica / evo` (s, N/A) | Matched Pairs `epica / evo` (↑) | Winner |
|---|---|---:|---:|---:|---:|---|
| euroc_mav_MH_02_easy_rovio | critical | 38952.190 / 28047.708 | 27.74 / 27.18 | -0.002 / 20.000 | 3000 / 2618 | n/a |
| grand_tour_2024-11-02-21-12-51_rovio | critical | 11.818 / 12.885 | 68.46 / 65.39 | -0.002 / 19.950 | 2647 / 2460 | n/a |
| grand_tour_2024-11-03-08-17-23_rovio | critical | 33.473 / 33.564 | 31.18 / 31.19 | 0.004 / 4.300 | 2316 / 2316 | n/a |
| grand_tour_2024-11-04-10-57-34_rovio | critical | 10.659 / 10.709 | 48.69 / 48.37 | -0.001 / 2.000 | 4689 / 4680 | n/a |
| grand_tour_2024-11-04-12-55-59_rovio | critical | 13.508 / 13.652 | 73.47 / 73.18 | 0.002 / -0.450 | 5205 / 5205 | n/a |
| grand_tour_2024-11-11-12-07-40_rovio | critical | 231646.600 / 231366.822 | 21.40 / 20.68 | -0.006 / 20.000 | 9593 / 9401 | n/a |
| grand_tour_2024-11-11-12-42-47_rovio | critical | 9670.280 / 8149.411 | 17.04 / 16.02 | -0.001 / 19.950 | 4099 / 3920 | n/a |
| grand_tour_2024-11-14-11-17-02_rovio | critical | 17.071 / 20.977 | 79.70 / 75.02 | 0.001 / 18.200 | 5284 / 5116 | n/a |
| grand_tour_2024-11-14-16-04-09_rovio | critical | 15.951 / 15.940 | 80.72 / 80.74 | 0.006 / -0.050 | 4886 / 4886 | n/a |
| grand_tour_2024-11-15-11-18-14_rovio | critical | 15.451 / 14.573 | 61.88 / 63.76 | -0.002 / 20.000 | 2954 / 2760 | n/a |
| grand_tour_2024-11-15-11-37-15_rovio | critical | 10.540 / 10.522 | 57.19 / 57.26 | -0.002 / -0.050 | 5216 / 5216 | n/a |
| grand_tour_2024-11-15-14-14-12_rovio | critical | 21652.802 / 20538.061 | 26.77 / 26.13 | -0.002 / 20.000 | 5415 / 5220 | n/a |
| grand_tour_2024-11-15-16-41-14_rovio | critical | 11.095 / 11.109 | 73.84 / 73.81 | 0.007 / 0.550 | 5425 / 5425 | n/a |
| grand_tour_2024-12-03-13-15-38_rovio | critical | 16.387 / 15.959 | 61.82 / 62.83 | 0.003 / -2.250 | 5152 / 5130 | n/a |
| grand_tour_2024-10-01-12-00-49_svo_stereo | critical | 36.644 / 36.653 | 76.50 / 76.49 | 0.003 / -0.150 | 4761 / 4761 | n/a |
| grand_tour_2024-11-14-11-17-02_svo_stereo | critical | 12.095 / 14.430 | 86.72 / 84.13 | 0.003 / 9.100 | 5285 / 5207 | n/a |
| grand_tour_2024-11-18-15-46-05_svo_stereo | critical | 35314.373 / 73993.197 | 1.03 / 0.05 | -0.003 / 11.300 | 1331 / 1331 | n/a |
| lamaria_sequence_1_19_svo_mono | critical | 11.572 / 20.788 | 97.62 / 95.68 | 0.009 / -19.450 | 2906 / 2915 | n/a |
| lamaria_sequence_1_20_svo_mono | critical | 92.767 / 92.618 | 82.43 / 82.47 | 0.005 / 17.550 | 3037 / 3036 | n/a |
| lamaria_sequence_2_11_svo_mono | critical | 4358.922 / 1053.053 | 2.15 / 15.35 | 0.012 / 14.500 | 3848 / 3863 | n/a |
| lamaria_sequence_2_12_svo_mono | critical | 36890.059 / 868.297 | -0.38 / 15.48 | -0.051 / -17.000 | 2553 / 2877 | n/a |
| lamaria_sequence_3_17_svo_mono | critical | 6268.453 / 36.749 | 0.02 / 92.42 | 0.011 / -19.100 | 5893 / 5927 | n/a |
| lamaria_sequence_3_18_svo_mono | critical | 37.234 / 34.760 | 92.41 / 92.82 | 0.008 / 19.500 | 5550 / 5471 | n/a |
| lamaria_sequence_4_10_svo_mono | critical | 133747.694 / 21425.686 | 0.38 / 3.81 | 0.013 / 19.550 | 3883 / 3852 | n/a |
| lamaria_sequence_4_11_svo_mono | critical | 1197499.986 / 915750.508 | 19.27 / 16.70 | 0.025 / 6.750 | 0 / 2209 | n/a |
| lamaria_sequence_1_19_svo_mono | critical | 11.572 / 20.788 | 97.62 / 95.68 | 0.009 / -19.450 | 2906 / 2915 | n/a |
| lamaria_sequence_1_20_svo_mono | critical | 92.767 / 92.618 | 82.43 / 82.47 | 0.005 / 17.550 | 3037 / 3036 | n/a |
| lamaria_sequence_2_11_svo_mono | critical | 4358.922 / 1053.053 | 2.15 / 15.35 | 0.012 / 14.500 | 3848 / 3863 | n/a |
| lamaria_sequence_2_12_svo_mono | critical | 36890.059 / 868.297 | -0.38 / 15.48 | -0.051 / -17.000 | 2553 / 2877 | n/a |
| lamaria_sequence_3_17_svo_mono | critical | 6268.453 / 36.749 | 0.02 / 92.42 | 0.011 / -19.100 | 5893 / 5927 | n/a |
| lamaria_sequence_3_18_svo_mono | critical | 37.234 / 34.760 | 92.41 / 92.82 | 0.008 / 19.500 | 5550 / 5471 | n/a |
| lamaria_sequence_4_10_svo_mono | critical | 133747.694 / 21425.686 | 0.38 / 3.81 | 0.013 / 19.550 | 3883 / 3852 | n/a |
| lamaria_sequence_4_11_svo_mono | critical | 1197499.986 / 915750.508 | 19.27 / 16.70 | 0.025 / 6.750 | 0 / 2209 | n/a |
| lamaria_R_11_5cp_svo_mono | critical | 21.163 / 23.872 | 95.49 / 94.89 | 0.021 / -18.000 | 0 / 1548 | n/a |
| lamaria_R_12_10cp_svo_mono | critical | 1737.041 / 1028.480 | 5.53 / 13.49 | 0.013 / -4.000 | 3224 / 3235 | n/a |
| lamaria_R_13_15cp_svo_mono | critical | 29.527 / 25.591 | 82.74 / 84.90 | 0.014 / -19.000 | 4286 / 4266 | n/a |

## Notes

- This study compares the current `epica` pipeline against `evo` using the same discovered cases.
- `epica` and `evo` do not share time offsets; each method uses its own time-alignment logic.
