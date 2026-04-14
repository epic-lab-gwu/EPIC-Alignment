# AlignAnything Independent Benchmark

- total cases: 139
- both_ok: 138
- offset policy: `vicon_ws` internal xcorr estimate + `evo` independent sweep (no sharing)

## Table 1: Common Metrics (vicon / evo)

| case | status (vicon/evo) | raw_rmse_m (v/e, ↓) | aligned_rmse_m (v/e, ↓) | improve_pct (v/e, ↑) | offset_s (v/e, N/A) | matches (v/e, ↑) |
|---|---|---:|---:|---:|---:|---:|
| euroc_mav_MH_01_easy_rovio | ok / ok | 5.929 / 5.929 | 0.266 / 0.268 | 95.52 / 95.48 | 1.073 / 0.050 | 29104 / 3639 |
| euroc_mav_MH_02_easy_rovio | ok / ok | 53906.481 / 38517.367 | 38951.445 / 28047.708 | 27.74 / 27.18 | 0.882 / 20.000 | 23995 / 2618 |
| euroc_mav_MH_03_medium_rovio | ok / ok | 5.838 / 5.835 | 0.337 / 0.374 | 94.23 / 93.60 | 2.347 / 0.150 | 21042 / 2631 |
| euroc_mav_MH_04_difficult_rovio | ok / ok | 16.227 / 14.702 | 0.608 / 6.565 | 96.26 / 55.34 | 1.594 / -20.000 | 15803 / 1597 |
| euroc_mav_MH_05_difficult_rovio | ok / ok | 16.879 / 15.196 | 0.568 / 7.280 | 96.63 / 52.10 | 1.362 / 20.000 | 17770 / 1849 |
| euroc_mav_V1_01_easy_rovio | ok / ok | 4.076 / 3.133 | 0.137 / 2.192 | 96.64 / 30.03 | -0.052 / -15.300 | 2894 / 2604 |
| euroc_mav_V1_02_medium_rovio | ok / ok | 3.478 / 2.249 | 0.156 / 1.885 | 95.51 / 16.16 | 0.941 / -12.500 | 13362 / 1438 |
| euroc_mav_V1_03_difficult_rovio | ok / ok | 3.330 / 2.475 | 0.136 / 1.767 | 95.91 / 28.63 | 1.792 / -20.000 | 16746 / 1712 |
| euroc_mav_V2_01_easy_rovio | ok / ok | 1.943 / 1.943 | 0.259 / 0.260 | 86.65 / 86.62 | 1.223 / 0.050 | 17920 / 2240 |
| euroc_mav_V2_02_medium_rovio | ok / ok | 1.909 / 1.909 | 0.272 / 0.275 | 85.75 / 85.62 | 1.218 / 0.000 | 18473 / 2310 |
| euroc_mav_V2_03_difficult_rovio | ok / ok | 1.733 / 1.723 | 0.164 / 0.172 | 90.51 / 90.04 | 1.181 / 0.050 | 15120 / 1890 |
| euroc_mav_MH_01_easy_svo_stereo | ok / ok | 6.185 / 6.189 | 0.129 / 0.152 | 97.91 / 97.54 | -0.925 / -0.150 | 28236 / 3624 |
| euroc_mav_MH_02_easy_svo_stereo | ok / ok | 5.817 / 5.818 | 0.065 / 0.077 | 98.89 / 98.68 | -0.064 / -0.050 | 23988 / 3000 |
| euroc_mav_MH_03_medium_svo_stereo | ok / ok | 5.853 / 5.851 | 0.173 / 0.245 | 97.05 / 95.81 | -0.348 / 0.150 | 20990 / 2621 |
| euroc_mav_MH_04_difficult_svo_stereo | ok / ok | 15.579 / 13.852 | 0.152 / 6.688 | 99.02 / 51.72 | -1.155 / -20.000 | 15425 / 1599 |
| euroc_mav_MH_05_difficult_svo_stereo | ok / ok | 16.385 / 14.684 | 0.171 / 6.960 | 98.96 / 52.60 | -1.485 / -20.000 | 17317 / 1845 |
| euroc_mav_V1_01_easy_svo_stereo | ok / ok | 4.011 / 3.057 | 0.041 / 2.226 | 98.97 / 27.18 | -1.050 / -15.200 | 2874 / 2608 |
| euroc_mav_V1_02_medium_svo_stereo | ok / ok | 3.488 / 2.157 | 0.077 / 1.873 | 97.79 / 13.17 | 0.994 / -12.450 | 13362 / 1441 |
| euroc_mav_V1_03_difficult_svo_stereo | ok / ok | 3.265 / 2.368 | 0.085 / 1.776 | 97.39 / 24.99 | -0.904 / -20.000 | 16605 / 1713 |
| euroc_mav_V2_01_easy_svo_stereo | ok / ok | 1.757 / 1.757 | 0.055 / 0.052 | 96.88 / 97.02 | -1.624 / 0.000 | 17664 / 2208 |
| euroc_mav_V2_02_medium_svo_stereo | ok / ok | 1.687 / 1.687 | 0.157 / 0.158 | 90.70 / 90.61 | -1.628 / 0.000 | 18216 / 2277 |
| euroc_mav_V2_03_difficult_svo_stereo | ok / ok | 1.667 / 1.661 | 0.134 / 0.140 | 91.95 / 91.55 | -1.666 / 0.050 | 14856 / 1856 |
| grand_tour_2024-10-01-11-29-55_rovio | ok / ok | 94.819 / 94.320 | 7.572 / 12.375 | 92.01 / 86.88 | -0.056 / 20.000 | 27760 / 3280 |
| grand_tour_2024-10-01-12-00-49_rovio | ok / ok | 15.836 / 15.252 | 6.540 / 7.399 | 58.70 / 51.49 | -0.043 / 10.350 | 38088 / 4665 |
| grand_tour_2024-11-02-17-10-25_rovio | ok / ok | 13.458 / 8.136 | 9.277 / 1.757 | 31.07 / 78.41 | 65.740 / 0.400 | 14320 / 1789 |
| grand_tour_2024-11-02-17-43-10_rovio | ok / ok | 11.033 / 11.044 | 2.821 / 2.928 | 74.43 / 73.48 | -0.072 / -1.850 | 18136 / 2249 |
| grand_tour_2024-11-02-21-12-51_rovio | ok / ok | 37.470 / 37.232 | 11.818 / 12.885 | 68.46 / 65.39 | -0.088 / 19.950 | 21176 / 2460 |
| grand_tour_2024-11-03-07-52-45_rovio | ok / ok | 25.410 / 25.383 | 2.972 / 3.029 | 88.30 / 88.07 | -0.001 / 0.000 | 14781 / 1848 |
| grand_tour_2024-11-03-08-17-23_rovio | ok / ok | 48.637 / 48.779 | 33.473 / 33.564 | 31.18 / 31.19 | -0.090 / 4.300 | 18528 / 2316 |
| grand_tour_2024-11-03-13-59-54_rovio | ok / ok | 14.752 / 13.964 | 6.955 / 8.055 | 52.85 / 42.31 | -0.068 / -9.150 | 28912 / 3523 |
| grand_tour_2024-11-04-10-57-34_rovio | ok / ok | 20.772 / 20.743 | 10.659 / 10.709 | 48.69 / 48.37 | -0.053 / 2.000 | 37512 / 4680 |
| grand_tour_2024-11-04-12-55-59_rovio | ok / ok | 50.915 / 50.906 | 13.508 / 13.652 | 73.47 / 73.18 | -0.466 / -0.450 | 41640 / 5205 |
| grand_tour_2024-11-04-13-07-13_rovio | ok / ok | 24.687 / 24.399 | 3.602 / 4.542 | 85.41 / 81.38 | -0.065 / 18.400 | 6552 / 636 |
| grand_tour_2024-11-04-16-05-00_rovio | ok / ok | 28.188 / 25.821 | 9.322 / 11.945 | 66.93 / 53.74 | -0.044 / 20.000 | 16480 / 1860 |
| grand_tour_2024-11-11-12-07-40_rovio | ok / ok | 294733.853 / 291704.660 | 231646.549 / 231366.822 | 21.40 / 20.68 | -0.019 / 20.000 | 76744 / 9401 |
| grand_tour_2024-11-11-12-42-47_rovio | ok / ok | 11656.854 / 9704.050 | 9670.313 / 8149.411 | 17.04 / 16.02 | -0.095 / 19.950 | 32792 / 3920 |
| grand_tour_2024-11-11-14-29-44_rovio | ok / ok | 79.904 / 79.974 | 8.898 / 9.004 | 88.86 / 88.74 | -0.043 / 0.750 | 28256 / 3532 |
| grand_tour_2024-11-14-11-17-02_rovio | ok / ok | 84.111 / 83.986 | 17.071 / 20.977 | 79.70 / 75.02 | 0.871 / 18.200 | 42272 / 5116 |
| grand_tour_2024-11-14-12-01-26_rovio | ok / ok | 20.955 / 20.039 | 9.156 / 9.851 | 56.31 / 50.84 | -0.047 / 19.100 | 36704 / 4399 |
| grand_tour_2024-11-14-14-36-02_rovio | ok / ok | 30.132 / 30.168 | 9.137 / 9.137 | 69.68 / 69.71 | -0.072 / -0.050 | 38296 / 4787 |
| grand_tour_2024-11-14-15-22-43_rovio | ok / ok | 64.214 / 64.294 | 6.132 / 6.127 | 90.45 / 90.47 | -0.088 / -0.100 | 31760 / 3970 |
| grand_tour_2024-11-14-16-04-09_rovio | ok / ok | 82.731 / 82.762 | 15.952 / 15.940 | 80.72 / 80.74 | -0.028 / -0.050 | 39088 / 4886 |
| grand_tour_2024-11-15-10-16-35_rovio | ok / ok | 37.018 / 37.082 | 9.803 / 9.811 | 73.52 / 73.54 | 0.008 / 1.400 | 38083 / 4761 |
| grand_tour_2024-11-15-11-18-14_rovio | ok / ok | 40.531 / 40.208 | 15.451 / 14.573 | 61.88 / 63.76 | -0.071 / 20.000 | 23632 / 2760 |
| grand_tour_2024-11-15-11-37-15_rovio | ok / ok | 24.621 / 24.618 | 10.540 / 10.522 | 57.19 / 57.26 | -0.068 / -0.050 | 41728 / 5216 |
| grand_tour_2024-11-15-12-06-03_rovio | ok / ok | 3.260 / 3.082 | 2.299 / 2.455 | 29.50 / 20.36 | -0.007 / -3.600 | 17798 / 2189 |
| grand_tour_2024-11-15-14-14-12_rovio | ok / ok | 29570.102 / 27804.104 | 21652.839 / 20538.061 | 26.77 / 26.13 | -0.091 / 20.000 | 43320 / 5220 |
| grand_tour_2024-11-15-14-43-52_rovio | ok / ok | 61.816 / 61.803 | 3.240 / 3.230 | 94.76 / 94.77 | -0.053 / -0.050 | 19896 / 2487 |
| grand_tour_2024-11-15-16-41-14_rovio | ok / ok | 42.404 / 42.415 | 11.095 / 11.109 | 73.84 / 73.81 | -0.067 / 0.550 | 43400 / 5425 |
| grand_tour_2024-11-18-12-05-01_rovio | ok / ok | 73.728 / 73.758 | 9.308 / 9.436 | 87.38 / 87.21 | -0.011 / 1.000 | 35687 / 4461 |
| grand_tour_2024-11-18-15-46-05_rovio | ok / ok | 6.173 / 6.170 | 3.927 / 3.925 | 36.38 / 36.39 | -0.027 / 0.000 | 32824 / 4103 |
| grand_tour_2024-11-18-16-59-23_rovio | ok / ok | 28.782 / 26.480 | 5.795 / 8.764 | 79.87 / 66.90 | -0.090 / 20.000 | 32336 / 3850 |
| grand_tour_2024-11-25-14-57-08_rovio | ok / ok | 6.515 / 6.519 | 3.339 / 3.343 | 48.74 / 48.72 | -0.015 / -0.200 | 25903 / 3236 |
| grand_tour_2024-11-25-16-36-19_rovio | ok / ok | 29.789 / 29.802 | 7.862 / 7.867 | 73.61 / 73.60 | -0.008 / 0.000 | 34830 / 4354 |
| grand_tour_2024-12-03-13-15-38_rovio | ok / ok | 42.918 / 42.935 | 16.387 / 15.959 | 61.82 / 62.83 | -0.059 / -2.250 | 41216 / 5130 |
| grand_tour_2024-12-03-13-26-40_rovio | ok / ok | 24.463 / 23.926 | 5.873 / 7.648 | 75.99 / 68.03 | -0.048 / -11.750 | 22760 / 2728 |
| grand_tour_2024-10-01-11-29-55_svo_stereo | ok / ok | 97.353 / 96.618 | 0.809 / 10.454 | 99.17 / 89.18 | -1.954 / 20.000 | 27608 / 3260 |
| grand_tour_2024-10-01-12-00-49_svo_stereo | ok / ok | 155.906 / 155.879 | 36.644 / 36.653 | 76.50 / 76.49 | -0.143 / -0.150 | 38088 / 4761 |
| grand_tour_2024-11-02-17-10-25_svo_stereo | ok / ok | 16.597 / 7.471 | 13.325 / 0.450 | 19.71 / 93.97 | 65.045 / -1.350 | 14320 / 1772 |
| grand_tour_2024-11-02-17-43-10_svo_stereo | ok / ok | 16.283 / 16.070 | 0.745 / 5.678 | 95.43 / 64.67 | -1.871 / -17.650 | 18000 / 2092 |
| grand_tour_2024-11-02-21-12-51_svo_stereo | ok / ok | 6.793 / 6.790 | 3.557 / 3.634 | 47.64 / 46.47 | -1.588 / -1.250 | 21064 / 2633 |
| grand_tour_2024-11-03-07-52-45_svo_stereo | ok / ok | 2.542 / 2.542 | 0.155 / 0.285 | 93.91 / 88.77 | -1.999 / 0.850 | 14632 / 1829 |
| grand_tour_2024-11-03-08-17-23_svo_stereo | ok / ok | 15.921 / 14.046 | 1.335 / 6.811 | 91.62 / 51.51 | -1.688 / -15.100 | 18408 / 2167 |
| grand_tour_2024-11-03-13-59-54_svo_stereo | ok / ok | 4.226 / 4.130 | 0.300 / 0.905 | 92.90 / 78.10 | -0.964 / -1.550 | 28848 / 3600 |
| grand_tour_2024-11-04-10-57-34_svo_stereo | ok / ok | 5.363 / 5.362 | 0.401 / 0.487 | 92.51 / 90.92 | -0.350 / 0.400 | 37488 / 4686 |
| grand_tour_2024-11-04-12-55-59_svo_stereo | ok / ok | 69.823 / 69.847 | 1.203 / 1.205 | 98.28 / 98.27 | -0.463 / -0.450 | 41559 / 5195 |
| grand_tour_2024-11-04-13-07-13_svo_stereo | ok / ok | 30.463 / 29.269 | 0.555 / 4.699 | 98.18 / 83.94 | -1.164 / 20.000 | 6440 / 608 |
| grand_tour_2024-11-04-16-05-00_svo_stereo | ok / ok | 44.709 / 43.397 | 0.429 / 13.201 | 99.04 / 69.58 | -0.441 / -20.000 | 16448 / 2060 |
| grand_tour_2024-11-11-12-07-40_svo_stereo | ok / ok | 86.988 / 87.019 | 2.444 / 2.500 | 97.19 / 97.13 | -0.116 / 0.700 | 76744 / 9593 |
| grand_tour_2024-11-11-12-42-47_svo_stereo | ok / ok | 58.235 / 58.396 | 2.308 / 2.527 | 96.04 / 95.67 | -0.893 / -0.900 | 32736 / 4092 |
| grand_tour_2024-11-11-14-29-44_svo_stereo | ok / ok | 86.405 / 86.610 | 1.509 / 1.533 | 98.25 / 98.23 | -1.041 / -1.050 | 28184 / 3523 |
| grand_tour_2024-11-14-11-17-02_svo_stereo | ok / ok | 91.053 / 90.949 | 12.094 / 14.430 | 86.72 / 84.13 | 0.874 / 9.100 | 42280 / 5207 |
| grand_tour_2024-11-14-12-01-26_svo_stereo | ok / ok | 36.274 / 36.274 | 3.929 / 3.948 | 89.17 / 89.12 | -0.446 / 0.150 | 36679 / 4585 |
| grand_tour_2024-11-14-14-36-02_svo_stereo | ok / ok | 30.439 / 30.473 | 2.708 / 2.728 | 91.11 / 91.05 | -0.270 / -0.250 | 38288 / 4786 |
| grand_tour_2024-11-14-15-22-43_svo_stereo | ok / ok | 74.142 / 74.277 | 1.357 / 1.436 | 98.17 / 98.07 | -0.685 / -0.700 | 31720 / 3965 |
| grand_tour_2024-11-14-16-04-09_svo_stereo | ok / ok | 83.180 / 82.770 | 1.216 / 14.176 | 98.54 / 82.87 | -0.426 / -19.950 | 39056 / 4687 |
| grand_tour_2024-11-15-10-16-35_svo_stereo | ok / ok | 44.021 / 44.103 | 0.842 / 0.839 | 98.09 / 98.10 | -0.189 / -0.200 | 37848 / 4731 |
| grand_tour_2024-11-15-11-18-14_svo_stereo | ok / ok | 38.281 / 38.388 | 0.794 / 0.839 | 97.93 / 97.81 | -1.369 / 0.550 | 23536 / 2942 |
| grand_tour_2024-11-15-11-37-15_svo_stereo | ok / ok | 25.358 / 25.384 | 1.152 / 1.077 | 95.46 / 95.76 | -0.065 / 0.300 | 41736 / 5217 |
| grand_tour_2024-11-15-12-06-03_svo_stereo | ok / ok | 3.208 / 3.046 | 1.001 / 1.313 | 68.78 / 56.90 | -1.704 / -2.900 | 17672 / 2197 |
| grand_tour_2024-11-15-14-14-12_svo_stereo | ok / ok | 56.242 / 55.871 | 0.675 / 7.734 | 98.80 / 86.16 | 0.016 / -20.000 | 43312 / 5215 |
| grand_tour_2024-11-15-14-43-52_svo_stereo | ok / ok | 61.853 / 61.890 | 6.037 / 5.757 | 90.24 / 90.70 | -1.650 / -1.650 | 19776 / 2472 |
| grand_tour_2024-11-15-16-41-14_svo_stereo | ok / ok | 36.270 / 36.280 | 3.128 / 3.103 | 91.38 / 91.45 | 0.047 / 0.050 | 43408 / 5427 |
| grand_tour_2024-11-18-12-05-01_svo_stereo | ok / ok | 56.966 / 57.032 | 6.518 / 6.630 | 88.56 / 88.37 | -0.617 / 0.900 | 35647 / 4456 |
| grand_tour_2024-11-18-15-46-05_svo_stereo | ok / ok | 35682.039 / 74028.023 | 35407.263 / 73993.197 | 0.77 / 0.05 | -0.625 / 11.300 | 10649 / 1331 |
| grand_tour_2024-11-18-16-59-23_svo_stereo | ok / ok | 19.493 / 19.491 | 7.309 / 7.406 | 62.50 / 62.00 | -0.586 / 1.450 | 32303 / 4030 |
| grand_tour_2024-11-25-14-57-08_svo_stereo | ok / ok | 1.573 / 1.560 | 1.300 / 1.293 | 17.37 / 17.16 | -1.214 / -0.350 | 25816 / 3227 |
| grand_tour_2024-11-25-16-36-19_svo_stereo | ok / ok | 46.294 / 46.340 | 1.077 / 1.134 | 97.67 / 97.55 | -0.605 / -0.600 | 34792 / 4349 |
| grand_tour_2024-12-03-13-15-38_svo_stereo | ok / ok | 33.656 / 33.659 | 2.051 / 1.878 | 93.91 / 94.42 | -0.157 / 0.750 | 41216 / 5152 |
| grand_tour_2024-12-03-13-26-40_svo_stereo | ok / ok | 24.147 / 23.737 | 0.641 / 5.499 | 97.35 / 76.83 | -1.247 / -9.850 | 22672 / 2748 |
| lamaria_sequence_1_19_svo_mono | ok / ok | 485.841 / 481.049 | 11.572 / 20.788 | 97.62 / 95.68 | 0.759 / -19.450 | 2906 / 2915 |
| lamaria_sequence_1_20_svo_mono | ok / ok | 527.859 / 528.194 | 92.767 / 92.618 | 82.43 / 82.47 | 1.105 / 17.550 | 3037 / 3036 |
| lamaria_sequence_2_11_svo_mono | ok / ok | 4454.613 / 1244.001 | 4358.922 / 1053.053 | 2.15 / 15.35 | 0.812 / 14.500 | 3848 / 3863 |
| lamaria_sequence_2_12_svo_mono | ok / ok | 65824.693 / 1027.300 | 65879.125 / 868.297 | -0.08 / 15.48 | -77.572 / -17.000 | 0 / 2877 |
| lamaria_sequence_3_17_svo_mono | ok / ok | 6269.836 / 484.927 | 6268.453 / 36.749 | 0.02 / 92.42 | 0.111 / -19.100 | 5893 / 5927 |
| lamaria_sequence_3_18_svo_mono | ok / ok | 490.838 / 484.287 | 37.234 / 34.760 | 92.41 / 92.82 | 0.908 / 19.500 | 5550 / 5471 |
| lamaria_sequence_4_10_svo_mono | ok / ok | 134258.053 / 22273.283 | 133747.694 / 21425.686 | 0.38 / 3.81 | 1.413 / 19.550 | 3883 / 3852 |
| lamaria_sequence_4_11_svo_mono | ok / ok | 1483250.613 / 1099365.994 | 1197499.986 / 915750.508 | 19.27 / 16.70 | 0.175 / 6.750 | 0 / 2209 |
| lamaria_sequence_1_19_svo_mono | ok / ok | 485.841 / 481.049 | 11.572 / 20.788 | 97.62 / 95.68 | 0.759 / -19.450 | 2906 / 2915 |
| lamaria_sequence_1_20_svo_mono | ok / ok | 527.859 / 528.194 | 92.767 / 92.618 | 82.43 / 82.47 | 1.105 / 17.550 | 3037 / 3036 |
| lamaria_sequence_2_11_svo_mono | ok / ok | 4454.613 / 1244.001 | 4358.922 / 1053.053 | 2.15 / 15.35 | 0.812 / 14.500 | 3848 / 3863 |
| lamaria_sequence_2_12_svo_mono | ok / ok | 65824.693 / 1027.300 | 65879.125 / 868.297 | -0.08 / 15.48 | -77.572 / -17.000 | 0 / 2877 |
| lamaria_sequence_3_17_svo_mono | ok / ok | 6269.836 / 484.927 | 6268.453 / 36.749 | 0.02 / 92.42 | 0.111 / -19.100 | 5893 / 5927 |
| lamaria_sequence_3_18_svo_mono | ok / ok | 490.838 / 484.287 | 37.234 / 34.760 | 92.41 / 92.82 | 0.908 / 19.500 | 5550 / 5471 |
| lamaria_sequence_4_10_svo_mono | ok / ok | 134258.053 / 22273.283 | 133747.694 / 21425.686 | 0.38 / 3.81 | 1.413 / 19.550 | 3883 / 3852 |
| lamaria_sequence_4_11_svo_mono | ok / ok | 1483250.613 / 1099365.994 | 1197499.986 / 915750.508 | 19.27 / 16.70 | 0.175 / 6.750 | 0 / 2209 |
| lamaria_R_11_5cp_svo_mono | failed / ok | nan / 467.122 | nan / 23.872 | nan / 94.89 | nan / -18.000 | nan / 1548 |
| lamaria_R_12_10cp_svo_mono | ok / ok | 1838.746 / 1188.906 | 1737.041 / 1028.480 | 5.53 / 13.49 | 0.163 / -4.000 | 3224 / 3235 |
| lamaria_R_13_15cp_svo_mono | ok / ok | 171.054 / 169.464 | 29.527 / 25.591 | 82.74 / 84.90 | 0.414 / -19.000 | 4286 / 4266 |
| uzh_fpv_indoor_forward_10_snapdragon_with_gt_rovio | ok / ok | 14.034 / 15.404 | 7.217 / 7.661 | 48.57 / 50.27 | -8.337 / 20.000 | 10262 / 836 |
| uzh_fpv_indoor_forward_3_snapdragon_with_gt_rovio | ok / ok | 22.439 / 17.654 | 0.628 / 6.989 | 97.20 / 60.41 | 29.969 / 18.700 | 22640 / 1382 |
| uzh_fpv_indoor_forward_5_snapdragon_with_gt_rovio | ok / ok | 14.927 / 13.400 | 7.602 / 6.114 | 49.07 / 54.38 | 1.811 / 8.650 | 9346 / 542 |
| uzh_fpv_indoor_forward_6_snapdragon_with_gt_rovio | ok / ok | 15.981 / 17.432 | 8.041 / 8.153 | 49.68 / 53.23 | 1.200 / 20.000 | 14982 / 881 |
| uzh_fpv_indoor_forward_7_snapdragon_with_gt_rovio | ok / ok | 23.010 / 20.026 | 1.174 / 8.096 | 94.90 / 59.57 | 30.303 / 20.000 | 29686 / 1752 |
| uzh_fpv_indoor_forward_9_snapdragon_with_gt_rovio | ok / ok | 13.460 / 13.533 | 6.933 / 6.841 | 48.49 / 49.45 | -8.790 / 20.000 | 9296 / 802 |
| uzh_fpv_indoor_forward_10_snapdragon_with_gt_svo_stereo | ok / ok | 13.987 / 14.776 | 6.736 / 7.384 | 51.84 / 50.03 | -11.099 / 20.000 | 8979 / 836 |
| uzh_fpv_indoor_forward_3_snapdragon_with_gt_svo_stereo | ok / ok | 22.557 / 17.820 | 0.313 / 7.270 | 98.61 / 59.20 | 27.132 / 18.550 | 22627 / 1382 |
| uzh_fpv_indoor_forward_5_snapdragon_with_gt_svo_stereo | ok / ok | 15.024 / 13.411 | 7.574 / 6.031 | 49.59 / 55.03 | 7.511 / 8.150 | 9307 / 542 |
| uzh_fpv_indoor_forward_6_snapdragon_with_gt_svo_stereo | ok / ok | 20.216 / 17.055 | 5.943 / 7.741 | 70.60 / 54.61 | 19.617 / 15.750 | 14256 / 874 |
| uzh_fpv_indoor_forward_7_snapdragon_with_gt_svo_stereo | ok / ok | 22.043 / 19.178 | 0.541 / 7.686 | 97.55 / 59.92 | 27.796 / 20.000 | 29682 / 1752 |
| uzh_fpv_indoor_forward_9_snapdragon_with_gt_svo_stereo | ok / ok | 13.474 / 13.431 | 6.612 / 4.920 | 50.93 / 63.37 | -11.584 / 5.000 | 8031 / 783 |
| uzh_fpv_indoor_45_12_snapdragon_with_gt_rovio | ok / ok | 18.706 / 13.643 | 9.228 / 6.582 | 50.67 / 51.75 | 21.180 / 15.150 | 18312 / 1112 |
| uzh_fpv_indoor_45_13_snapdragon_with_gt_rovio | ok / ok | 14.635 / 10.054 | 8.288 / 2.467 | 43.37 / 75.47 | -1.128 / -0.650 | 17050 / 1038 |
| uzh_fpv_indoor_45_14_snapdragon_with_gt_rovio | ok / ok | 11.360 / 10.704 | 0.528 / 2.830 | 95.35 / 73.56 | 18.984 / -0.550 | 17294 / 1028 |
| uzh_fpv_indoor_45_2_snapdragon_with_gt_rovio | ok / ok | 15.285 / 13.702 | 0.533 / 5.726 | 96.51 / 58.21 | 18.061 / -1.950 | 21700 / 1258 |
| uzh_fpv_indoor_45_4_snapdragon_with_gt_rovio | ok / ok | 14.621 / 13.339 | 0.451 / 5.348 | 96.92 / 59.91 | 18.300 / -1.800 | 19606 / 1155 |
| uzh_fpv_indoor_45_12_snapdragon_with_gt_svo_stereo | ok / ok | 10.262 / 9.810 | 0.735 / 2.253 | 92.84 / 77.03 | 18.433 / -1.000 | 18312 / 1076 |
| uzh_fpv_indoor_45_13_snapdragon_with_gt_svo_stereo | ok / ok | 10.256 / 9.892 | 0.652 / 2.296 | 93.64 / 76.79 | 16.250 / -0.600 | 17499 / 1038 |
| uzh_fpv_indoor_45_14_snapdragon_with_gt_svo_stereo | ok / ok | 10.697 / 10.297 | 1.083 / 2.559 | 89.87 / 75.15 | 16.192 / -0.450 | 17301 / 1028 |
| uzh_fpv_indoor_45_2_snapdragon_with_gt_svo_stereo | ok / ok | 14.006 / 12.987 | 0.509 / 4.482 | 96.36 / 65.49 | 15.422 / -1.400 | 21701 / 1260 |
| uzh_fpv_indoor_45_4_snapdragon_with_gt_svo_stereo | ok / ok | 14.361 / 13.366 | 0.453 / 4.785 | 96.85 / 64.20 | 15.361 / -1.600 | 19611 / 1155 |
| uzh_fpv_outdoor_forward_1_snapdragon_with_gt_rovio | ok / ok | 38.939 / 35.400 | 17.721 / 16.105 | 54.49 / 54.50 | 20.401 / 14.450 | 19835 / 1188 |
| uzh_fpv_outdoor_forward_3_snapdragon_with_gt_rovio | ok / ok | 58.568 / 51.570 | 0.732 / 18.317 | 98.75 / 64.48 | 32.267 / 15.550 | 38075 / 2275 |
| uzh_fpv_outdoor_forward_5_snapdragon_with_gt_rovio | ok / ok | 35.869 / 31.086 | 28.510 / 23.067 | 20.52 / 25.80 | 4.886 / 8.350 | 7877 / 464 |
| uzh_fpv_outdoor_forward_1_snapdragon_with_gt_svo_stereo | ok / ok | 26.138 / 33.926 | 14.150 / 15.678 | 45.87 / 53.79 | -14.291 / 13.650 | 13504 / 1185 |
| uzh_fpv_outdoor_forward_3_snapdragon_with_gt_svo_stereo | ok / ok | 58.152 / 50.965 | 1.864 / 18.603 | 96.80 / 63.50 | 29.847 / 15.550 | 38038 / 2274 |
| uzh_fpv_outdoor_forward_5_snapdragon_with_gt_svo_stereo | ok / ok | 35.874 / 30.688 | 28.527 / 22.841 | 20.48 / 25.57 | 2.485 / 8.100 | 7878 / 464 |
| uzh_fpv_outdoor_45_1_snapdragon_with_gt_rovio | ok / ok | 18.553 / 12.623 | 11.314 / 2.487 | 39.02 / 80.30 | -5.587 / -0.350 | 6980 / 504 |
| uzh_fpv_outdoor_45_1_snapdragon_with_gt_svo_stereo | ok / ok | 14.638 / 14.034 | 0.314 / 3.456 | 97.85 / 75.37 | 13.684 / -0.500 | 8587 / 493 |

## Table 2: vicon_ws-only Metrics

| case | vicon_xcorr_peak (↑) | vicon_xcorr_psr (↑) | vicon_omega_improve_pct (↑) |
|---|---:|---:|---:|
| euroc_mav_MH_01_easy_rovio | 0.795 | 14.31 | 65.67 |
| euroc_mav_MH_02_easy_rovio | 0.804 | 14.25 | 63.74 |
| euroc_mav_MH_03_medium_rovio | 0.944 | 10.12 | 37.04 |
| euroc_mav_MH_04_difficult_rovio | 0.862 | 13.53 | -543.39 |
| euroc_mav_MH_05_difficult_rovio | 0.948 | 13.96 | 47.85 |
| euroc_mav_V1_01_easy_rovio | 0.963 | 13.08 | 20.26 |
| euroc_mav_V1_02_medium_rovio | 0.975 | 7.64 | 91.13 |
| euroc_mav_V1_03_difficult_rovio | 0.964 | 8.76 | 6.89 |
| euroc_mav_V2_01_easy_rovio | 0.965 | 10.97 | 83.93 |
| euroc_mav_V2_02_medium_rovio | 0.971 | 11.60 | 90.83 |
| euroc_mav_V2_03_difficult_rovio | 0.963 | 8.05 | 84.05 |
| euroc_mav_MH_01_easy_svo_stereo | 0.967 | 14.85 | 73.90 |
| euroc_mav_MH_02_easy_svo_stereo | 0.976 | 14.67 | 57.49 |
| euroc_mav_MH_03_medium_svo_stereo | 0.981 | 10.44 | 79.82 |
| euroc_mav_MH_04_difficult_svo_stereo | 0.941 | 13.50 | 75.52 |
| euroc_mav_MH_05_difficult_svo_stereo | 0.951 | 14.09 | 61.36 |
| euroc_mav_V1_01_easy_svo_stereo | 0.985 | 13.08 | 72.99 |
| euroc_mav_V1_02_medium_svo_stereo | 0.972 | 7.65 | 89.33 |
| euroc_mav_V1_03_difficult_svo_stereo | 0.977 | 8.97 | 87.66 |
| euroc_mav_V2_01_easy_svo_stereo | 0.955 | 11.07 | -7.45 |
| euroc_mav_V2_02_medium_svo_stereo | 0.960 | 11.94 | 85.33 |
| euroc_mav_V2_03_difficult_svo_stereo | 0.945 | 8.34 | -7.07 |
| grand_tour_2024-10-01-11-29-55_rovio | 0.898 | 10.98 | 16.30 |
| grand_tour_2024-10-01-12-00-49_rovio | 0.904 | 14.30 | 10.66 |
| grand_tour_2024-11-02-17-10-25_rovio | 0.229 | 2.94 | -6706.03 |
| grand_tour_2024-11-02-17-43-10_rovio | 0.986 | 9.11 | 33.30 |
| grand_tour_2024-11-02-21-12-51_rovio | 0.916 | 12.30 | 20.51 |
| grand_tour_2024-11-03-07-52-45_rovio | 0.866 | 6.26 | 0.00 |
| grand_tour_2024-11-03-08-17-23_rovio | 0.882 | 14.59 | 29.34 |
| grand_tour_2024-11-03-13-59-54_rovio | 0.896 | 12.01 | 22.82 |
| grand_tour_2024-11-04-10-57-34_rovio | 0.946 | 16.65 | 14.20 |
| grand_tour_2024-11-04-12-55-59_rovio | 0.880 | 10.96 | 43.98 |
| grand_tour_2024-11-04-13-07-13_rovio | 0.870 | 11.74 | 22.49 |
| grand_tour_2024-11-04-16-05-00_rovio | 0.929 | 12.40 | 11.04 |
| grand_tour_2024-11-11-12-07-40_rovio | 0.778 | 21.79 | 1.91 |
| grand_tour_2024-11-11-12-42-47_rovio | 0.848 | 19.73 | 28.40 |
| grand_tour_2024-11-11-14-29-44_rovio | 0.856 | 15.54 | 12.43 |
| grand_tour_2024-11-14-11-17-02_rovio | 0.913 | 17.12 | 59.48 |
| grand_tour_2024-11-14-12-01-26_rovio | 0.956 | 15.25 | 15.69 |
| grand_tour_2024-11-14-14-36-02_rovio | 0.829 | 13.96 | 27.07 |
| grand_tour_2024-11-14-15-22-43_rovio | 0.921 | 14.13 | 29.52 |
| grand_tour_2024-11-14-16-04-09_rovio | 0.889 | 18.05 | 5.58 |
| grand_tour_2024-11-15-10-16-35_rovio | 0.920 | 14.54 | 0.51 |
| grand_tour_2024-11-15-11-18-14_rovio | 0.933 | 14.24 | 27.62 |
| grand_tour_2024-11-15-11-37-15_rovio | 0.892 | 12.73 | 25.95 |
| grand_tour_2024-11-15-12-06-03_rovio | 0.944 | 9.04 | 0.59 |
| grand_tour_2024-11-15-14-14-12_rovio | 0.875 | 13.35 | 31.14 |
| grand_tour_2024-11-15-14-43-52_rovio | 0.921 | 14.38 | 17.97 |
| grand_tour_2024-11-15-16-41-14_rovio | 0.951 | 16.24 | 19.26 |
| grand_tour_2024-11-18-12-05-01_rovio | 0.917 | 17.89 | 1.14 |
| grand_tour_2024-11-18-15-46-05_rovio | 0.954 | 13.00 | 5.12 |
| grand_tour_2024-11-18-16-59-23_rovio | 0.947 | 9.44 | 30.41 |
| grand_tour_2024-11-25-14-57-08_rovio | 0.971 | 9.93 | 1.98 |
| grand_tour_2024-11-25-16-36-19_rovio | 0.948 | 15.77 | 0.40 |
| grand_tour_2024-12-03-13-15-38_rovio | 0.924 | 13.95 | 17.81 |
| grand_tour_2024-12-03-13-26-40_rovio | 0.939 | 12.78 | 17.47 |
| grand_tour_2024-10-01-11-29-55_svo_stereo | 0.893 | 10.87 | 49.16 |
| grand_tour_2024-10-01-12-00-49_svo_stereo | 0.867 | 13.75 | 31.84 |
| grand_tour_2024-11-02-17-10-25_svo_stereo | 0.229 | 2.87 | -1051.59 |
| grand_tour_2024-11-02-17-43-10_svo_stereo | 0.978 | 9.08 | -159.15 |
| grand_tour_2024-11-02-21-12-51_svo_stereo | 0.928 | 12.34 | 59.97 |
| grand_tour_2024-11-03-07-52-45_svo_stereo | 0.935 | 6.56 | -67.51 |
| grand_tour_2024-11-03-08-17-23_svo_stereo | 0.885 | 14.62 | 56.55 |
| grand_tour_2024-11-03-13-59-54_svo_stereo | 0.897 | 11.92 | 54.85 |
| grand_tour_2024-11-04-10-57-34_svo_stereo | 0.947 | 16.66 | 51.39 |
| grand_tour_2024-11-04-12-55-59_svo_stereo | 0.883 | 10.93 | 44.53 |
| grand_tour_2024-11-04-13-07-13_svo_stereo | 0.860 | 11.69 | 57.69 |
| grand_tour_2024-11-04-16-05-00_svo_stereo | 0.929 | 12.36 | 54.89 |
| grand_tour_2024-11-11-12-07-40_svo_stereo | 0.841 | 21.12 | 31.82 |
| grand_tour_2024-11-11-12-42-47_svo_stereo | 0.851 | 19.82 | 52.44 |
| grand_tour_2024-11-11-14-29-44_svo_stereo | 0.853 | 15.47 | 51.74 |
| grand_tour_2024-11-14-11-17-02_svo_stereo | 0.917 | 17.07 | 60.31 |
| grand_tour_2024-11-14-12-01-26_svo_stereo | 0.956 | 15.14 | 58.48 |
| grand_tour_2024-11-14-14-36-02_svo_stereo | 0.831 | 13.92 | 42.80 |
| grand_tour_2024-11-14-15-22-43_svo_stereo | 0.919 | 13.99 | 53.40 |
| grand_tour_2024-11-14-16-04-09_svo_stereo | 0.890 | 18.16 | 44.26 |
| grand_tour_2024-11-15-10-16-35_svo_stereo | 0.921 | 14.56 | 43.81 |
| grand_tour_2024-11-15-11-18-14_svo_stereo | 0.936 | 14.34 | 65.46 |
| grand_tour_2024-11-15-11-37-15_svo_stereo | 0.894 | 12.73 | 25.46 |
| grand_tour_2024-11-15-12-06-03_svo_stereo | 0.938 | 9.04 | 59.96 |
| grand_tour_2024-11-15-14-14-12_svo_stereo | 0.445 | 12.47 | 18.16 |
| grand_tour_2024-11-15-14-43-52_svo_stereo | 0.917 | 14.36 | 67.30 |
| grand_tour_2024-11-15-16-41-14_svo_stereo | 0.556 | 16.04 | 49.98 |
| grand_tour_2024-11-18-12-05-01_svo_stereo | 0.501 | 15.32 | 11.00 |
| grand_tour_2024-11-18-15-46-05_svo_stereo | 0.310 | 7.54 | 82.38 |
| grand_tour_2024-11-18-16-59-23_svo_stereo | 0.926 | 9.07 | 51.05 |
| grand_tour_2024-11-25-14-57-08_svo_stereo | 0.971 | 9.94 | 72.14 |
| grand_tour_2024-11-25-16-36-19_svo_stereo | 0.945 | 15.79 | 58.40 |
| grand_tour_2024-12-03-13-15-38_svo_stereo | 0.924 | 13.91 | 40.07 |
| grand_tour_2024-12-03-13-26-40_svo_stereo | 0.935 | 12.76 | 51.62 |
| lamaria_sequence_1_19_svo_mono | 0.702 | 25.22 | 36.52 |
| lamaria_sequence_1_20_svo_mono | 0.777 | 26.54 | 43.98 |
| lamaria_sequence_2_11_svo_mono | 0.794 | 24.25 | 48.62 |
| lamaria_sequence_2_12_svo_mono | 0.082 | 4.93 | -2383.33 |
| lamaria_sequence_3_17_svo_mono | 0.779 | 26.88 | 13.69 |
| lamaria_sequence_3_18_svo_mono | 0.798 | 25.24 | 47.87 |
| lamaria_sequence_4_10_svo_mono | 0.763 | 26.09 | 42.12 |
| lamaria_sequence_4_11_svo_mono | 0.431 | 16.52 | 3.87 |
| lamaria_sequence_1_19_svo_mono | 0.702 | 25.22 | 36.52 |
| lamaria_sequence_1_20_svo_mono | 0.777 | 26.54 | 43.98 |
| lamaria_sequence_2_11_svo_mono | 0.794 | 24.25 | 48.62 |
| lamaria_sequence_2_12_svo_mono | 0.082 | 4.93 | -2383.33 |
| lamaria_sequence_3_17_svo_mono | 0.779 | 26.88 | 13.69 |
| lamaria_sequence_3_18_svo_mono | 0.798 | 25.24 | 47.87 |
| lamaria_sequence_4_10_svo_mono | 0.763 | 26.09 | 42.12 |
| lamaria_sequence_4_11_svo_mono | 0.431 | 16.52 | 3.87 |
| lamaria_R_11_5cp_svo_mono | nan | nan | nan |
| lamaria_R_12_10cp_svo_mono | 0.779 | 25.20 | 21.87 |
| lamaria_R_13_15cp_svo_mono | 0.812 | 25.21 | 42.28 |
| uzh_fpv_indoor_forward_10_snapdragon_with_gt_rovio | 0.303 | 2.57 | -4702.93 |
| uzh_fpv_indoor_forward_3_snapdragon_with_gt_rovio | 0.438 | 5.06 | -539.30 |
| uzh_fpv_indoor_forward_5_snapdragon_with_gt_rovio | 0.230 | 2.63 | -25.07 |
| uzh_fpv_indoor_forward_6_snapdragon_with_gt_rovio | 0.157 | 2.09 | 4.22 |
| uzh_fpv_indoor_forward_7_snapdragon_with_gt_rovio | 0.722 | 8.37 | -2568.50 |
| uzh_fpv_indoor_forward_9_snapdragon_with_gt_rovio | 0.250 | 2.18 | -5794.19 |
| uzh_fpv_indoor_forward_10_snapdragon_with_gt_svo_stereo | 0.363 | 3.05 | -3816.55 |
| uzh_fpv_indoor_forward_3_snapdragon_with_gt_svo_stereo | 0.471 | 5.36 | -2483.10 |
| uzh_fpv_indoor_forward_5_snapdragon_with_gt_svo_stereo | 0.226 | 2.00 | -652.51 |
| uzh_fpv_indoor_forward_6_snapdragon_with_gt_svo_stereo | 0.184 | 2.92 | -703.92 |
| uzh_fpv_indoor_forward_7_snapdragon_with_gt_svo_stereo | 0.741 | 8.66 | -843.37 |
| uzh_fpv_indoor_forward_9_snapdragon_with_gt_svo_stereo | 0.303 | 2.10 | -2166.35 |
| uzh_fpv_indoor_45_12_snapdragon_with_gt_rovio | 0.421 | 3.34 | -685.59 |
| uzh_fpv_indoor_45_13_snapdragon_with_gt_rovio | 0.459 | 3.33 | 8.90 |
| uzh_fpv_indoor_45_14_snapdragon_with_gt_rovio | 0.348 | 2.86 | -366.94 |
| uzh_fpv_indoor_45_2_snapdragon_with_gt_rovio | 0.632 | 6.00 | -1397.02 |
| uzh_fpv_indoor_45_4_snapdragon_with_gt_rovio | 0.506 | 4.00 | -361.97 |
| uzh_fpv_indoor_45_12_snapdragon_with_gt_svo_stereo | 0.545 | 3.59 | -2135.47 |
| uzh_fpv_indoor_45_13_snapdragon_with_gt_svo_stereo | 0.488 | 3.39 | -881.61 |
| uzh_fpv_indoor_45_14_snapdragon_with_gt_svo_stereo | 0.362 | 3.24 | -1153.34 |
| uzh_fpv_indoor_45_2_snapdragon_with_gt_svo_stereo | 0.712 | 6.45 | -1913.59 |
| uzh_fpv_indoor_45_4_snapdragon_with_gt_svo_stereo | 0.609 | 4.71 | -3509.06 |
| uzh_fpv_outdoor_forward_1_snapdragon_with_gt_rovio | 0.310 | 2.13 | -1148.07 |
| uzh_fpv_outdoor_forward_3_snapdragon_with_gt_rovio | 0.511 | 4.27 | -301.20 |
| uzh_fpv_outdoor_forward_5_snapdragon_with_gt_rovio | 0.427 | 3.43 | -218.41 |
| uzh_fpv_outdoor_forward_1_snapdragon_with_gt_svo_stereo | 0.322 | 2.15 | -6128.32 |
| uzh_fpv_outdoor_forward_3_snapdragon_with_gt_svo_stereo | 0.533 | 4.49 | -4192.83 |
| uzh_fpv_outdoor_forward_5_snapdragon_with_gt_svo_stereo | 0.474 | 3.17 | -261.94 |
| uzh_fpv_outdoor_45_1_snapdragon_with_gt_rovio | 0.254 | 1.92 | -463.43 |
| uzh_fpv_outdoor_45_1_snapdragon_with_gt_svo_stereo | 0.331 | 2.49 | -3711.21 |

## Table 3: evo-only Metrics

| case | evo_sweep_evals (N/A) |
|---|---:|
| euroc_mav_MH_01_easy_rovio | 102 |
| euroc_mav_MH_02_easy_rovio | 92 |
| euroc_mav_MH_03_medium_rovio | 102 |
| euroc_mav_MH_04_difficult_rovio | 92 |
| euroc_mav_MH_05_difficult_rovio | 92 |
| euroc_mav_V1_01_easy_rovio | 102 |
| euroc_mav_V1_02_medium_rovio | 102 |
| euroc_mav_V1_03_difficult_rovio | 92 |
| euroc_mav_V2_01_easy_rovio | 102 |
| euroc_mav_V2_02_medium_rovio | 102 |
| euroc_mav_V2_03_difficult_rovio | 102 |
| euroc_mav_MH_01_easy_svo_stereo | 102 |
| euroc_mav_MH_02_easy_svo_stereo | 102 |
| euroc_mav_MH_03_medium_svo_stereo | 102 |
| euroc_mav_MH_04_difficult_svo_stereo | 92 |
| euroc_mav_MH_05_difficult_svo_stereo | 92 |
| euroc_mav_V1_01_easy_svo_stereo | 102 |
| euroc_mav_V1_02_medium_svo_stereo | 102 |
| euroc_mav_V1_03_difficult_svo_stereo | 92 |
| euroc_mav_V2_01_easy_svo_stereo | 102 |
| euroc_mav_V2_02_medium_svo_stereo | 102 |
| euroc_mav_V2_03_difficult_svo_stereo | 102 |
| grand_tour_2024-10-01-11-29-55_rovio | 92 |
| grand_tour_2024-10-01-12-00-49_rovio | 102 |
| grand_tour_2024-11-02-17-10-25_rovio | 102 |
| grand_tour_2024-11-02-17-43-10_rovio | 102 |
| grand_tour_2024-11-02-21-12-51_rovio | 92 |
| grand_tour_2024-11-03-07-52-45_rovio | 102 |
| grand_tour_2024-11-03-08-17-23_rovio | 102 |
| grand_tour_2024-11-03-13-59-54_rovio | 102 |
| grand_tour_2024-11-04-10-57-34_rovio | 102 |
| grand_tour_2024-11-04-12-55-59_rovio | 102 |
| grand_tour_2024-11-04-13-07-13_rovio | 102 |
| grand_tour_2024-11-04-16-05-00_rovio | 92 |
| grand_tour_2024-11-11-12-07-40_rovio | 92 |
| grand_tour_2024-11-11-12-42-47_rovio | 92 |
| grand_tour_2024-11-11-14-29-44_rovio | 102 |
| grand_tour_2024-11-14-11-17-02_rovio | 102 |
| grand_tour_2024-11-14-12-01-26_rovio | 102 |
| grand_tour_2024-11-14-14-36-02_rovio | 102 |
| grand_tour_2024-11-14-15-22-43_rovio | 102 |
| grand_tour_2024-11-14-16-04-09_rovio | 102 |
| grand_tour_2024-11-15-10-16-35_rovio | 102 |
| grand_tour_2024-11-15-11-18-14_rovio | 92 |
| grand_tour_2024-11-15-11-37-15_rovio | 102 |
| grand_tour_2024-11-15-12-06-03_rovio | 102 |
| grand_tour_2024-11-15-14-14-12_rovio | 92 |
| grand_tour_2024-11-15-14-43-52_rovio | 102 |
| grand_tour_2024-11-15-16-41-14_rovio | 102 |
| grand_tour_2024-11-18-12-05-01_rovio | 102 |
| grand_tour_2024-11-18-15-46-05_rovio | 102 |
| grand_tour_2024-11-18-16-59-23_rovio | 92 |
| grand_tour_2024-11-25-14-57-08_rovio | 102 |
| grand_tour_2024-11-25-16-36-19_rovio | 102 |
| grand_tour_2024-12-03-13-15-38_rovio | 102 |
| grand_tour_2024-12-03-13-26-40_rovio | 102 |
| grand_tour_2024-10-01-11-29-55_svo_stereo | 92 |
| grand_tour_2024-10-01-12-00-49_svo_stereo | 102 |
| grand_tour_2024-11-02-17-10-25_svo_stereo | 102 |
| grand_tour_2024-11-02-17-43-10_svo_stereo | 102 |
| grand_tour_2024-11-02-21-12-51_svo_stereo | 102 |
| grand_tour_2024-11-03-07-52-45_svo_stereo | 102 |
| grand_tour_2024-11-03-08-17-23_svo_stereo | 102 |
| grand_tour_2024-11-03-13-59-54_svo_stereo | 102 |
| grand_tour_2024-11-04-10-57-34_svo_stereo | 102 |
| grand_tour_2024-11-04-12-55-59_svo_stereo | 102 |
| grand_tour_2024-11-04-13-07-13_svo_stereo | 92 |
| grand_tour_2024-11-04-16-05-00_svo_stereo | 92 |
| grand_tour_2024-11-11-12-07-40_svo_stereo | 102 |
| grand_tour_2024-11-11-12-42-47_svo_stereo | 102 |
| grand_tour_2024-11-11-14-29-44_svo_stereo | 102 |
| grand_tour_2024-11-14-11-17-02_svo_stereo | 102 |
| grand_tour_2024-11-14-12-01-26_svo_stereo | 102 |
| grand_tour_2024-11-14-14-36-02_svo_stereo | 102 |
| grand_tour_2024-11-14-15-22-43_svo_stereo | 102 |
| grand_tour_2024-11-14-16-04-09_svo_stereo | 92 |
| grand_tour_2024-11-15-10-16-35_svo_stereo | 102 |
| grand_tour_2024-11-15-11-18-14_svo_stereo | 102 |
| grand_tour_2024-11-15-11-37-15_svo_stereo | 102 |
| grand_tour_2024-11-15-12-06-03_svo_stereo | 102 |
| grand_tour_2024-11-15-14-14-12_svo_stereo | 92 |
| grand_tour_2024-11-15-14-43-52_svo_stereo | 102 |
| grand_tour_2024-11-15-16-41-14_svo_stereo | 102 |
| grand_tour_2024-11-18-12-05-01_svo_stereo | 102 |
| grand_tour_2024-11-18-15-46-05_svo_stereo | 102 |
| grand_tour_2024-11-18-16-59-23_svo_stereo | 102 |
| grand_tour_2024-11-25-14-57-08_svo_stereo | 102 |
| grand_tour_2024-11-25-16-36-19_svo_stereo | 102 |
| grand_tour_2024-12-03-13-15-38_svo_stereo | 102 |
| grand_tour_2024-12-03-13-26-40_svo_stereo | 102 |
| lamaria_sequence_1_19_svo_mono | 102 |
| lamaria_sequence_1_20_svo_mono | 102 |
| lamaria_sequence_2_11_svo_mono | 102 |
| lamaria_sequence_2_12_svo_mono | 102 |
| lamaria_sequence_3_17_svo_mono | 102 |
| lamaria_sequence_3_18_svo_mono | 102 |
| lamaria_sequence_4_10_svo_mono | 92 |
| lamaria_sequence_4_11_svo_mono | 102 |
| lamaria_sequence_1_19_svo_mono | 102 |
| lamaria_sequence_1_20_svo_mono | 102 |
| lamaria_sequence_2_11_svo_mono | 102 |
| lamaria_sequence_2_12_svo_mono | 102 |
| lamaria_sequence_3_17_svo_mono | 102 |
| lamaria_sequence_3_18_svo_mono | 102 |
| lamaria_sequence_4_10_svo_mono | 92 |
| lamaria_sequence_4_11_svo_mono | 102 |
| lamaria_R_11_5cp_svo_mono | 102 |
| lamaria_R_12_10cp_svo_mono | 102 |
| lamaria_R_13_15cp_svo_mono | 102 |
| uzh_fpv_indoor_forward_10_snapdragon_with_gt_rovio | 92 |
| uzh_fpv_indoor_forward_3_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_indoor_forward_5_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_indoor_forward_6_snapdragon_with_gt_rovio | 92 |
| uzh_fpv_indoor_forward_7_snapdragon_with_gt_rovio | 92 |
| uzh_fpv_indoor_forward_9_snapdragon_with_gt_rovio | 92 |
| uzh_fpv_indoor_forward_10_snapdragon_with_gt_svo_stereo | 92 |
| uzh_fpv_indoor_forward_3_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_indoor_forward_5_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_indoor_forward_6_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_indoor_forward_7_snapdragon_with_gt_svo_stereo | 92 |
| uzh_fpv_indoor_forward_9_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_indoor_45_12_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_indoor_45_13_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_indoor_45_14_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_indoor_45_2_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_indoor_45_4_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_indoor_45_12_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_indoor_45_13_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_indoor_45_14_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_indoor_45_2_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_indoor_45_4_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_outdoor_forward_1_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_outdoor_forward_3_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_outdoor_forward_5_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_outdoor_forward_1_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_outdoor_forward_3_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_outdoor_forward_5_snapdragon_with_gt_svo_stereo | 102 |
| uzh_fpv_outdoor_45_1_snapdragon_with_gt_rovio | 102 |
| uzh_fpv_outdoor_45_1_snapdragon_with_gt_svo_stereo | 102 |

注：`↑` 越大越好，`↓` 越小越好，`N/A` 为参数或过程量，不适用统一优劣方向。
