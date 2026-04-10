# AlignAnything Corner Case Eval (no rerun) - Final

- total cases: 10
- vicon_ws ok: 10
- evo ok: 10

## Table 1: Common Metrics (vicon / evo)

> 说明：除 `t_offset_used_s` 外，每个单元格格式为 `vicon / evo`。
> 说明：`t_offset_used_s` 为共享输入值（evo 本次评估使用 vicon 估计的 offset），因此该列只填一个数。
> 说明：`vicon_step3` 与 `evo_se3` 都是“对齐后误差”，但流程定义不同；可比较效果，不是同一算法。

| case | status (vicon/evo) | raw_rmse_m (vicon/evo, ↓) | aligned_rmse_m (vicon/evo, ↓) | improve_pct (vicon/evo, ↑) | t_offset_used_s (shared input, N/A) | matches (vicon/evo, ↑) |
|---|---|---:|---:|---:|---:|---:|
| euroc_MH01_rovio | ok / ok | 5.929 / 5.955 | 0.266 / 0.628 | 95.52 / 89.46 | -1.073 | 29104 / 3637 |
| euroc_MH05_rovio | ok / ok | 16.879 / 16.872 | 0.568 / 1.684 | 96.63 / 90.02 | -1.362 | 17770 / 2215 |
| euroc_V203_svo | ok / ok | 1.667 / 2.021 | 0.134 / 1.159 | 91.95 / 42.67 | 1.666 | 14856 / 1840 |
| grandtour_20241102_211251_rovio | ok / ok | 37.470 / 37.421 | 11.818 / 11.626 | 68.46 / 68.93 | 0.088 | 21176 / 2647 |
| grandtour_20241102_211251_svo | ok / ok | 6.793 / 6.851 | 3.557 / 3.505 | 47.64 / 48.84 | 1.588 | 21064 / 2629 |
| lamaria_add2_318_svo | ok / ok | 490.838 / 489.754 | 37.234 / 30.921 | 92.41 / 93.69 | -0.908 | 5550 / 5553 |
| lamaria_cp_15_svo | ok / ok | 171.054 / 170.555 | 29.527 / 27.050 | 82.74 / 84.14 | -0.414 | 4286 / 4267 |
| uzh_indoor10_svo | ok / ok | 13.987 / 16.943 | 6.736 / 6.723 | 51.84 / 60.32 | 11.099 | 8979 / 838 |
| uzh_indoor45_14_rovio | ok / ok | 11.360 / 16.290 | 0.528 / 7.978 | 95.35 / 51.02 | -18.984 | 17294 / 745 |
| uzh_outdoor45_1_svo | ok / ok | 14.638 / 22.139 | 0.314 / 9.079 | 97.85 / 58.99 | -13.684 | 8587 / 501 |

## Table 2: vicon_ws-only Metrics

| case | vicon_xcorr_peak (↑) | vicon_xcorr_psr (↑) | vicon_omega_improve_pct (↑) | vicon_offset_est_s (N/A) |
|---|---:|---:|---:|---:|
| euroc_MH01_rovio | 0.795 | 14.31 | 65.67 | 1.073 |
| euroc_MH05_rovio | 0.948 | 13.96 | 47.85 | 1.362 |
| euroc_V203_svo | 0.945 | 8.34 | -7.07 | -1.666 |
| grandtour_20241102_211251_rovio | 0.916 | 12.30 | 20.51 | -0.088 |
| grandtour_20241102_211251_svo | 0.928 | 12.34 | 59.97 | -1.588 |
| lamaria_add2_318_svo | 0.798 | 25.24 | 47.87 | 0.908 |
| lamaria_cp_15_svo | 0.812 | 25.21 | 42.28 | 0.414 |
| uzh_indoor10_svo | 0.363 | 3.05 | -3816.55 | -11.099 |
| uzh_indoor45_14_rovio | 0.348 | 2.86 | -366.94 | 18.984 |
| uzh_outdoor45_1_svo | 0.331 | 2.49 | -3711.20 | 13.684 |

## Table 3: evo-only Metrics

当前这批输出里，evo 暂无额外独有指标（已并入 Table 1）。

注：
- `↑`：数值越大越好；`↓`：数值越小越好。
- `N/A`：该值是参数/偏移量，不存在统一的“越大或越小越好”。
