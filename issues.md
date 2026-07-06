# Issue 1: Lamaria Hard 上 PosYaw 和 SE3 的 SR 差异明显

日期：2026-07-06

## 问题概述

师兄在测试 `epa_ov_error_comparison` 时发现，Lamaria hard sequences 上 `posyaw`
和 `se3` 的 drift-valid success rate 差异很大。其中 `se3` 的结果看起来比较合理，
但 `posyaw` 在若干 case 上出现了异常的 `0.00` 或接近 `0.00` 的 SR。

原始测试命令：

```bash
epa_ov_error_comparison se3 /mnt/e/benchmark_gt/lamaria/hard benchmark/lamaria/hard/pose
epa_ov_error_comparison posyaw /mnt/e/benchmark_gt/lamaria/hard benchmark/lamaria/hard/pose
```

本地复现后确认：这个 issue 确实存在。

## 修复前现象

修复前，`se3` 的 SR 结果如下：

| Mode | Algorithm | R_08_hard | R_09_hard | R_10_hard | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| se3 | rovio | 73.50 | 41.42 | 44.26 | 53.06 |
| se3 | svo_mono | 49.05 | 67.31 | 37.81 | 51.39 |

修复前，`posyaw` 的 SR 结果如下：

| Mode | Algorithm | R_08_hard | R_09_hard | R_10_hard | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| posyaw | rovio | 83.75 | 0.00 | 3.12 | 28.96 |
| posyaw | svo_mono | 51.84 | 69.93 | 0.00 | 40.59 |

异常最明显的 case：

- `rovio / R_09_hard`
- `svo_mono / R_10_hard`

这些 case 在 `posyaw` 下出现了异常的 0 SR，但 `se3` 下仍能得到合理的
drift-valid segment。

## 根因分析

旧版 `posyaw` 实现的问题在于：它更像是一个 full-trajectory PosYaw eval
alignment。也就是说，它会在整条 trajectory 上拟合一个 yaw + translation。

Lamaria hard sequences 中有些轨迹后半段发散非常严重。使用整条轨迹拟合
PosYaw 时，发散尾部会把 yaw 和 translation 的全局拟合结果拖偏，导致原本局部
合理的 trajectory segment 也无法通过 global APE gate，最后 SR 被压成 `0.00`。

这个问题不是 SR 计算本身的问题，而是 `posyaw` 的全局对齐方式被 divergent tail
污染了。

调试中还发现一个需要注意的点：`epa_ov_error_comparison se3` 里的 `se3` 是历史兼容
alias，实际走的是 EPA Step3，而不是单纯的普通 SE3 eval alignment。因此，如果把
`posyaw` 做成 “full Step3 之后再额外做 PosYaw eval alignment”，结果会几乎完全等于
`se3`。这可以消掉 0 SR，但不是正确语义。

## 师兄建议的 PosYaw 语义

师兄给出的实现建议是：

- PosYaw 可以基于现有 three-step / EPA Step3 思路。
- extrinsic alignment 仍然可以做 6DoF。
- global frame alignment 阶段只 align yaw 和 translation。
- roll / pitch 不做 align。

因此正确的 pipeline 应该是：

```text
Step1:
  time alignment

Step2:
  extrinsic alignment，仍然是完整 6DoF

Step3:
  global frame alignment，但只允许 yaw + translation
  不允许 global roll / pitch alignment
```

也就是说：

```text
se3:
  Step2: full 6DoF extrinsic
  Step3: full 3D rotation + translation global alignment

posyaw:
  Step2: full 6DoF extrinsic
  Step3: yaw-only rotation + translation global alignment
  global roll / pitch 不参与对齐
```

## 修复方案

最终修复是把 PosYaw 放进 Step3 的 global frame alignment 阶段，而不是在完整 Step3
之后再做一个 eval alignment。

具体实现：

1. 新增 yaw-only world alignment。

   在 `src/epa/core/steps.py` 中新增 PosYaw global alignment solver。它只用 XY 平面
   估计 yaw：

   ```text
   Rw = Rz(yaw)
   tw = mean(gt) - Rw @ mean(est)
   ```

   因此 global rotation matrix 永远是 Z 轴 yaw rotation，不会包含 roll / pitch。

2. PosYaw world alignment 也使用 robust trimmed 思路。

   旧问题的核心是 hard sequence 的 divergent tail 会污染全局拟合。因此 PosYaw
   Step3 也复用了 Step3 robust trimming 的思想：

   - 先拟合 yaw + translation。
   - 计算 residual。
   - trim 掉 outlier。
   - 再用 inlier 重新拟合 yaw + translation。

3. `_solve_step2_step3(...)` 增加 `global_align_mode` 参数。

   - `global_align_mode="se3"`：保留原来的 full SE3 Step3。
   - `global_align_mode="posyaw"`：使用 yaw-only + translation Step3。

4. 更新 main pipeline。

   当用户运行：

   ```bash
   epa --eval-align posyaw ...
   ```

   内部会走：

   ```text
   Step2 full 6DoF extrinsic
   Step3 yaw-only global alignment
   ```

5. 更新 compatibility command。

   当用户运行：

   ```bash
   epa_ov_error_comparison posyaw ...
   ```

   命令不变，但内部会走新的 PosYaw-constrained Step3。

## 关键修改文件

- `src/epa/core/steps.py`
  - 新增 yaw-only world alignment。
  - 新增 PosYaw robust trimmed world alignment。
  - `_solve_step2_step3(...)` 支持 `global_align_mode`。
  - PosYaw 模式下 Step3 orientation 使用 `yaw * q_step2`，不吸收 global roll/pitch。

- `src/epa/core/pipeline_modular.py`
  - `--eval-align posyaw` 时传入 `global_align_mode="posyaw"`。
  - PosYaw 不再作为 Step3 之后的额外 eval alignment。

- `src/epa/ov_eval_compat.py`
  - `epa_ov_error_comparison posyaw` 走 PosYaw-constrained Step3。
  - 输出 source 标记为 `epa_posyaw`，便于和 `epa_step3` 区分。

- `tests/unit/test_step3_mode_selection.py`
  - 增加测试，确认 PosYaw world alignment 不会吸收 roll / pitch。

- `tests/unit/test_epica_sim3.py`
  - 增加 divergent tail 相关的 PosYaw 对齐测试。

## 验证命令

本地 Lamaria hard 验证命令：

```bash
cd /home/yifu/epa_data/AlignAnything2/AlignAnything2

epa_ov_error_comparison se3 \
  /home/yifu/epa_data/lamaria-hard/hard \
  benchmark/lamaria/hard/pose

epa_ov_error_comparison posyaw \
  /home/yifu/epa_data/lamaria-hard/hard \
  benchmark/lamaria/hard/pose
```

单 case 验证命令：

```bash
epa \
  --gt /home/yifu/epa_data/lamaria-hard/hard/R_09_hard.txt \
  --gt-format tum \
  --est /home/yifu/epa_data/AlignAnything2/AlignAnything2/benchmark/lamaria/hard/pose/rovio/R_09_hard/rovio_poses.txt \
  --est-format tum \
  --eval-align posyaw
```

单 case 输出确认走到了新的 PosYaw Step3：

```text
Step3 alignment_mode=posyaw_robust_trimmed
SR_valid_dist=41.38%
```

单元测试：

```bash
pytest tests/unit
```

结果：

```text
216 passed, 2 warnings
```

## 修复后结果

修复后，`posyaw` 不再出现异常的 0 SR。同时，它也不再和 `se3` 完全相同。
两者数值接近，是因为这些 Lamaria hard case 在 Step2 之后不需要很大的 global
roll / pitch correction；但现在两条路径的语义已经不同：

```text
se3 source:    epa_step3
posyaw source: epa_posyaw
```

### Drift-Valid Success Rate

| Mode | Algorithm | R_08_hard | R_09_hard | R_10_hard | Average |
| --- | --- | ---: | ---: | ---: | ---: |
| original posyaw | rovio | 83.75 | 0.00 | 3.12 | 28.96 |
| fixed posyaw | rovio | 73.44 | 41.38 | 44.00 | 52.94 |
| se3 / Step3 | rovio | 73.50 | 41.42 | 44.26 | 53.06 |
| original posyaw | svo_mono | 51.84 | 69.93 | 0.00 | 40.59 |
| fixed posyaw | svo_mono | 49.51 | 67.24 | 37.84 | 51.53 |
| se3 / Step3 | svo_mono | 49.05 | 67.31 | 37.81 | 51.39 |

### Full Trajectory ATE

单位：rotation deg / translation m。

| Mode | Algorithm | R_08_hard | R_09_hard | R_10_hard | Average |
| --- | --- | --- | --- | --- | --- |
| original posyaw | rovio | 9.068 / 15.938 | 131.439 / 98275.529 | 9.274 / 33.606 | 49.927 / 32775.024 |
| fixed posyaw | rovio | 9.781 / 19.698 | 55.290 / 112148.982 | 9.155 / 34.157 | 24.742 / 37400.946 |
| se3 / Step3 | rovio | 9.810 / 19.657 | 57.390 / 112149.821 | 9.206 / 34.129 | 25.468 / 37401.202 |
| original posyaw | svo_mono | 10.088 / 38.293 | 5.564 / 37.752 | 70.105 / 24513.048 | 28.586 / 8196.364 |
| fixed posyaw | svo_mono | 14.296 / 86.573 | 6.146 / 37.826 | 53.747 / 27397.080 | 24.730 / 9173.826 |
| se3 / Step3 | svo_mono | 14.403 / 86.511 | 6.663 / 37.513 | 53.909 / 27396.640 | 24.992 / 9173.554 |

### Drift-Valid Only ATE

单位：rotation deg / translation m。

| Mode | Algorithm | R_08_hard | R_09_hard | R_10_hard | Average |
| --- | --- | --- | --- | --- | --- |
| original posyaw | rovio | 9.121 / 14.384 | nan / nan | 0.372 / 6.004 | 4.747 / 10.194 |
| fixed posyaw | rovio | 8.286 / 10.443 | 15.069 / 17.235 | 7.709 / 19.410 | 10.355 / 15.696 |
| se3 / Step3 | rovio | 8.318 / 10.395 | 16.021 / 17.086 | 7.784 / 19.502 | 10.708 / 15.661 |
| original posyaw | svo_mono | 9.233 / 6.932 | 4.244 / 20.234 | nan / nan | 6.738 / 13.583 |
| fixed posyaw | svo_mono | 4.521 / 5.589 | 4.109 / 17.381 | 4.499 / 11.582 | 4.376 / 11.517 |
| se3 / Step3 | svo_mono | 4.652 / 5.602 | 4.918 / 17.107 | 4.809 / 11.725 | 4.793 / 11.478 |

## 结论

这个 issue 的根因是旧版 `posyaw` 在整条 hard trajectory 上拟合 yaw + translation，
导致发散尾部污染全局对齐结果，进而让一些本应有有效局部 segment 的 case 被 global
gate 判成 0 SR。

最终修复是把 PosYaw 正确放到 EPA Step3 的 global frame alignment 阶段：

```text
extrinsic alignment: full 6DoF
global frame alignment: yaw + translation only
global roll / pitch: not aligned
```

修复后，Lamaria hard 上 `posyaw` 的异常 0 SR 被消除，并且仍保留了它和 full SE3
Step3 的语义区别。

# Issue 2: Aqualoc Sim3 低 SR case 的 drift-valid 姿态误差异常

日期：2026-07-06

## 问题概述

师兄在 Aqualoc `archaeo1` 上测试 `sim3` 时发现：Sim3 的 SR 相比之前有所改善，
但一些小 SR case 在 `DRIFT-VALID ONLY ATE` 表里仍然有很大的 orientation error，
接近 170-180 度。

师兄原始命令：

```bash
epa_ov_error_comparison sim3 /mnt/e/benchmark_gt/aqualoc/archaeo/archaeo1 benchmark/aqualoc1/pose/
```

本地对应命令：

```bash
cd /home/yifu/epa_data/AlignAnything2/AlignAnything2

epa_ov_error_comparison sim3 \
  GT/aqualoc/archaeo/archaeo1 \
  benchmark/archaeo/pose
```

## 师兄指出的问题 case

按 “small SR + drift-valid orientation error 很大” 筛选，主要是这 4 个 case：

| Case | Method | 原 SR | 原 drift-valid ATE rot/trans |
| --- | --- | ---: | ---: |
| `archaeo_sequence_1` | `rovio` | 31.08 | 179.347 / 20.670 |
| `archaeo_sequence_2` | `rovio` | 28.22 | 178.745 / 5.400 |
| `archaeo_sequence_5` | `rovio` | 8.05 | 175.756 / 7.280 |
| `archaeo_sequence_4` | `svo_stereo` | 0.36 | 169.240 / 10.404 |

这些 case 的共同特点是：轨迹开头往往和 GT 对得比较好，但后面发生明显漂移或 scale
发散。问题在于，旧的 Sim3 对齐或 fallback 选择可能让 `DRIFT-VALID ONLY ATE` 里的
orientation 仍然接近 180 度，看起来像有效片段的姿态完全错了。

## drift-valid rot 的定义确认

调试过程中确认了一个关键点：`DRIFT-VALID ONLY ATE` 里的 `rot` 不是 full trajectory
rotation RMSE，而是在 drift-valid segment 覆盖到的 sample 上计算的 APE rotation RMSE。

计算逻辑是：

1. 先对轨迹做当前 eval alignment，例如 `sim3`。
2. 对所有 timestamp 计算 APE translation 和 APE rotation。
3. 用 translation/drift 规则筛出 drift-valid segment：
   - APE translation 低于阈值；
   - 1s RPE translation 没有明显 jump；
   - APE slope / jump 不像已经漂走；
   - 一个 segment 的两端 sample 都 valid，这个 segment 才算 valid。
4. 只保留这些 valid segment 覆盖到的 sample。
5. 在这些 sample 上计算：

```text
rot_i = angle(R_est_i^-1 * R_gt_i)
drift_valid_rot = sqrt(mean(rot_i^2 over drift-valid samples))
```

因此，如果一个 case 前半段对齐好、后半段漂移，drift-valid rot 应该主要反映前半段
未漂移部分，而不是整条轨迹。

## 根因分析

这个 issue 最终不是通过“把 unreliable case 从 SR 表里 gate 掉”解决的。早期确实尝试过
使用 reliability-gated SR，但后来决定去掉 `gated SR` 这个对外 metric，只保留：

- raw SR；
- local/drift-valid SR；
- SR reliability status；
- warning explanation。

真正的问题在于主 `sim3` 的候选选择逻辑不够稳：

1. 原始 `epica_sim3` 有时会给出数学上可用、但 scale 严重越界的结果。

   例如 `archaeo_sequence_5_rovio` 和 `archaeo_sequence_7_rovio`，轨迹开头看起来正常，
   后面 scale 发散。纯 `epica_sim3` 可能会用一个极端 scale 吸收错误，让整体轨迹视觉上
   不合理。

2. 只用 “scale 是否 severe” 作为 fail 条件又太硬。

   `archaeo_sequence_4_svo_stereo` 是反例。它的 `epica_sim3` scale 很小，但视觉轨迹和
   drift-valid 指标明显比 fallback 更合理。如果单纯因为 severe scale 拒绝 `epica_sim3`，
   会错误 fallback 到更差的 `epa_sim3_v2`。

3. `epica_sim3_stable` 对部分 “前面好、后面漂” 的 case 更合适。

   stable 版本会在较稳定的窗口上找 anchor，再把该 Sim3 transform 应用到整条轨迹。
   这对 `seq1/seq2/seq5_rovio` 这类 case 能显著降低 drift-valid orientation error。

## 当前 sim3 组成

对外 mode 仍然只叫 `sim3`，命令不变。内部现在是一个三段式候选流程：

```text
sim3:
  1. 先尝试 epica_sim3
  2. 如果 epica_sim3 失败或 scale 严重越界，则尝试 epica_sim3_stable
  3. 如果 stable 也失败，则 fallback 到 epa_sim3_v2
  4. 如果 primary epica 只是 scale severe，但 stable 更差或不可用，
     则保留 primary epica，而不是盲目 fallback 到 v2
```

这里的第 4 点是后来专门为 `archaeo_sequence_4_svo_stereo` 修正的。这个 case 证明：
小 scale 不一定意味着这个 Sim3 解比 fallback 差；需要结合几何残差和实际轨迹形状判断。

## 关键修改点

- `src/epa/core/sim3.py`
  - 主 `solve_epa_sim3(...)` 改成候选式流程：
    `epica_sim3 -> epica_sim3_stable -> epa_sim3_v2`。
  - severe scale 不再简单等于丢弃 primary。
  - 如果 stable 不可用，而 primary EPICA transform 本身可用，则保留 primary。

- `src/epa/benchmark/benchmark_harness.py`
  - 修复 `--python-bin .venv/bin/python` 被 `.resolve()` 解到 pyenv 原始 Python 的问题。
    现在保留 venv symlink，benchmark 能正确使用虚拟环境。

- `src/epa/viz/interactive_html.py`
- `src/epa/core/io_utils.py`
- `src/epa/core/evaluation.py`
- `src/epa/core/pipeline_modular.py`
  - 去掉对外展示的 `gated SR` metric。
  - SR 不再被 reliability gate 改写为 0。
  - reliability 只通过 `SR reliability` 和 warning explanation 表达。

## 最终手动验证结果

用户用当前版本重新手动运行师兄命令后，输出显示原来 4 个问题 case 已明显改善：

```text
TOOL SOURCE: epa_step3=0, epa_eval=10, failed=0
```

### Drift-Valid Success Rate

| Case | Method | 师兄原 SR | 当前 SR |
| --- | --- | ---: | ---: |
| `archaeo_sequence_1` | `rovio` | 31.08 | 32.36 |
| `archaeo_sequence_2` | `rovio` | 28.22 | 42.65 |
| `archaeo_sequence_5` | `rovio` | 8.05 | 13.21 |
| `archaeo_sequence_4` | `svo_stereo` | 0.36 | 77.51 |

### Drift-Valid Only ATE

单位：rotation deg / translation m。

| Case | Method | 师兄原 drift-valid ATE | 当前 drift-valid ATE |
| --- | --- | ---: | ---: |
| `archaeo_sequence_1` | `rovio` | 179.347 / 20.670 | 1.581 / 0.774 |
| `archaeo_sequence_2` | `rovio` | 178.745 / 5.400 | 1.741 / 0.361 |
| `archaeo_sequence_5` | `rovio` | 175.756 / 7.280 | 2.401 / 0.576 |
| `archaeo_sequence_4` | `svo_stereo` | 169.240 / 10.404 | 6.886 / 3.296 |

## 结论

师兄指出的问题已经基本解决：

- 原来 drift-valid only 表中接近 170-180 度的异常 orientation error，现在降到几度量级。
- `seq1/seq2/seq5_rovio` 的 drift-valid rot 已经能正确反映前半段未明显漂移的有效片段。
- `seq4_svo_stereo` 不再被过硬的 scale guard 错误 fallback 到更差的 v2，而是保留更合理的
  EPICA Sim3 对齐。
- full trajectory 仍可能有很大的 translation error，说明后半段漂移/scale 崩溃仍存在；
  但这正是 full trajectory 表应该暴露的问题，不应混入 drift-valid only 指标。

最终保留的对外指标是 raw/local SR + reliability status，不再展示 `gated SR`。

# Issue 3: Drift-valid distance RPE 是否混入 failure sequences

日期：2026-07-06

## 问题概述

师兄在 Aqualoc `archaeo1` 上测试 `sim3` 时发现，`DRIFT-VALID ONLY DISTANCE RPE`
表里的数值和 `FULL TRAJECTORY DISTANCE RPE` 表完全一样：

```text
FULL TRAJECTORY DISTANCE RPE LATEX TABLE
rovio      20.779 / 82171.569 ... 53.671 / 300244.784
svo_stereo  4.295 / 36.153    ... 14.736 / 176.561

DRIFT-VALID ONLY DISTANCE RPE LATEX TABLE
rovio      20.779 / 82171.569 ... 53.671 / 300244.784
svo_stereo  4.295 / 36.153    ... 14.736 / 176.561
```

这说明 drift-valid distance RPE 很可能没有真正过滤掉 failure sequences。

师兄还建议：

- 去掉原来的 aggregate distance RPE 表；
- 改成 per-sequence 的 segment RPE 表；
- segment 长度改成 `10m / 20m / 50m / 100m`。

期望格式类似：

```text
DRIFT-VALID ONLY 10m SEGMENT RPE LATEX TABLE (ROT DEG / TRANS M)
 & sequence_1 & sequence_2 & ... & Average
rovio & ...
svo_stereo & ...
```

## 根因分析

根因是 `valid_segment_mask` 的长度理解错了。

在 `compute_valid_segment_metrics(...)` 里：

```text
valid_sample_mask:  长度 N，表示每个 pose/sample 是否 valid
valid_segment_mask: 长度 N-1，表示相邻 pose 之间的 edge/segment 是否 valid
```

也就是说，`valid_segment_mask[i]` 表示：

```text
pose_i -> pose_{i+1}
```

这条相邻轨迹段是否 drift-valid。

但是 distance RPE 过滤函数 `_compute_rpe_segments_ov_eval_style(...)` 之前错误地认为
传进来的 mask 必须是长度 `N` 的 sample mask：

```text
if valid_mask.size != gt_pos.shape[0]:
    valid_mask = None
```

而实际传进来的是长度 `N-1` 的 `valid_segment_mask`。因此长度不匹配，代码直接把
valid mask 置成 `None`，导致后续 distance RPE 没有做任何 drift-valid 过滤。

结果就是：

```text
DRIFT-VALID ONLY DISTANCE RPE == FULL TRAJECTORY DISTANCE RPE
```

failure sequence 自然也被混进了 drift-valid RPE。

## 修复方案

修复发生在 `src/epa/ov_eval_compat.py`。

### 1. 正确支持 N-1 的 valid segment mask

现在 `_compute_rpe_segments_ov_eval_style(...)` 同时支持两种 mask：

```text
长度 N:
  解释为 valid_sample_mask
  RPE pair start -> end 要求所有 sample 都 valid

长度 N-1:
  解释为 valid_edge_mask / valid_segment_mask
  RPE pair start -> end 要求 start 到 end 之间的所有 edge 都 valid
```

也就是说，对于一个 distance RPE pair：

```text
pose_start -> pose_end
```

只有当下面这些边全部 drift-valid 时，才会计入 drift-valid RPE：

```text
valid_segment_mask[start : end]
```

如果中间任何一段已经漂移或失效，这个 RPE pair 会被跳过。

### 2. 删除旧的 aggregate distance RPE 表

移除了最终 LaTeX 输出里的：

```text
FULL TRAJECTORY DISTANCE RPE LATEX TABLE
FULL TRAJECTORY DISTANCE DRIFT RATE LATEX TABLE
DRIFT-VALID ONLY DISTANCE RPE LATEX TABLE
DRIFT-VALID ONLY DISTANCE DRIFT RATE LATEX TABLE
```

原因是这些 aggregate 表按算法和 segment 长度汇总，不能看出具体哪个 sequence 在贡献错误，
而且旧的 drift-valid 表曾经因为 mask bug 混入 failure sequence，容易误导。

### 3. 新增 per-sequence drift-valid segment RPE 表

新增 4 张表：

```text
DRIFT-VALID ONLY 10m SEGMENT RPE LATEX TABLE
DRIFT-VALID ONLY 20m SEGMENT RPE LATEX TABLE
DRIFT-VALID ONLY 50m SEGMENT RPE LATEX TABLE
DRIFT-VALID ONLY 100m SEGMENT RPE LATEX TABLE
```

每张表的列是 sequence，行是 algorithm：

```text
 & sequence_1 & sequence_2 & sequence_3 & sequence_4 & sequence_5 & Average
rovio & ...
svo_stereo & ...
```

这样可以直接看每个 sequence 在指定距离段上的 drift-valid RPE，而不是把所有 sequence
混在一个 aggregate 数字里。

## 为什么会出现 nan / nan

修复后，有些 cell 会显示：

```text
nan / nan
```

这是合理的，不是新的 bug。

原因是：某个 sequence 在指定 segment 长度下，没有任何一对 pose 能同时满足：

1. GT 距离间隔接近指定长度，比如 `10m`；
2. 从 start 到 end 之间所有相邻 edge 都是 drift-valid；
3. 这对 pose 没有跨过漂移/失败段。

如果没有满足条件的 RPE pair，就没有样本可以计算 RMSE，于是显示 `nan / nan`。

例如 Aqualoc `archaeo1` 当前结果里：

```text
DRIFT-VALID ONLY 10m SEGMENT RPE
rovio / archaeo_sequence_2: nan / nan
rovio / archaeo_sequence_5: nan / nan
```

这表示：

```text
seq2_rovio 和 seq5_rovio 虽然有一些短的 drift-valid 片段，
但没有连续 10m 的 drift-valid trajectory segment。
```

因此它们不应该被强行纳入 10m distance RPE。显示 `nan / nan` 比把 failure 段混进去更正确。

对于更长的 segment，比如 `50m` 或 `100m`，`nan / nan` 会更常见，因为很多 sequence
本身总长度不足、或 drift-valid 连续片段没有那么长。

## 修复后验证

验证命令：

```bash
cd /home/yifu/epa_data/AlignAnything2/AlignAnything2

epa_ov_error_comparison sim3 \
  GT/aqualoc/archaeo/archaeo1 \
  benchmark/archaeo/pose
```

修复后，旧的 aggregate distance RPE 表不再出现。输出中出现新的 per-sequence 表。

### 10m Segment RPE

单位：rotation deg / translation m。

| Algorithm | seq1 | seq2 | seq3 | seq4 | seq5 | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rovio | 1.760 / 1.013 | nan / nan | 2.297 / 1.342 | 8.057 / 1.847 | nan / nan | 4.038 / 1.401 |
| svo_stereo | 2.708 / 2.911 | 2.895 / 0.451 | 2.520 / 0.595 | 6.912 / 2.470 | 2.511 / 2.044 | 3.509 / 1.694 |

### 20m Segment RPE

| Algorithm | seq1 | seq2 | seq3 | seq4 | seq5 | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rovio | 2.839 / 1.247 | nan / nan | 1.770 / 1.828 | 8.132 / 2.633 | nan / nan | 4.247 / 1.903 |
| svo_stereo | 3.466 / 3.553 | 3.014 / 0.492 | 2.083 / 0.681 | 7.875 / 2.958 | 5.232 / 1.936 | 4.334 / 1.924 |

### 50m Segment RPE

| Algorithm | seq1 | seq2 | seq3 | seq4 | seq5 | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rovio | nan / nan | nan / nan | nan / nan | 9.275 / 4.133 | nan / nan | 9.275 / 4.133 |
| svo_stereo | 6.731 / 4.129 | nan / nan | nan / nan | 9.864 / 4.923 | nan / nan | 8.298 / 4.526 |

### 100m Segment RPE

| Algorithm | seq1 | seq2 | seq3 | seq4 | seq5 | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rovio | nan / nan | nan / nan | nan / nan | 14.185 / 6.612 | nan / nan | 14.185 / 6.612 |
| svo_stereo | nan / nan | nan / nan | nan / nan | nan / nan | nan / nan | nan / nan |

## 测试

新增单元测试覆盖了这个 bug：

```text
test_valid_rpe_segments_filter_by_valid_edge_mask
```

它构造一个后半段失败的轨迹，确认：

- full distance RPE 会看到跨入失败段的 pair；
- drift-valid distance RPE 只保留所有 edge 都 valid 的 pair；
- 如果没有连续足够长的 valid segment，则 pair count 为 0。

相关测试：

```bash
pytest tests/unit/test_ov_eval_compat.py
```

结果：

```text
17 passed
```

全量单元测试：

```bash
pytest tests/unit
```

结果：

```text
225 passed, 2 warnings
```

## 结论

这个 issue 确实存在。旧版 drift-valid distance RPE 因为 mask 长度处理错误，没有真正应用
drift-valid filter，所以包含了 failure sequences。

修复后，drift-valid distance RPE 只统计完全落在连续 drift-valid 片段内的 pair，并改成
10m / 20m / 50m / 100m 的 per-sequence 表。`nan / nan` 表示该 sequence 在对应距离下没有
足够长的连续有效片段，这是正确行为。

# Issue4: `epa` 主命令比 `epa_ov_error_singlerun` 慢很多

## 师兄反馈的问题

师兄测试 Lamaria hard `R_10_hard` 时发现：

```bash
epa /mnt/e/benchmark_gt/lamaria/hard/R_10_hard.txt
```

耗时：

```text
143.21s user 23.64s system 122% cpu 2:16.19 total
```

而：

```bash
epa_ov_error_singlerun se3 /mnt/e/benchmark_gt/lamaria/hard/R_10_hard.txt
```

耗时：

```text
14.78s user 19.64s system 419% cpu 8.212 total
```

也就是 `epa` 约 2 分 16 秒，而 `epa_ov_error_singlerun` 约 8 秒，差距非常大。

## 初步确认

当前版本中，`epa` positional 模式需要两个输入：

```text
<gt_file> <est_file>
```

如果只传一个 GT 文件，当前程序会直接报错：

```text
epa: error: Positional mode requires exactly two inputs: <gt_file> <est_file>.
```

所以师兄实际运行时应该还有 est 输入，或者通过 config/default 参数补了 est。我们按师兄目录风格模拟时使用：

```bash
cd /home/yifu/epa_data/AlignAnything2/AlignAnything2

/home/yifu/EPIC-Alignment/.venv/bin/epa \
  /mnt/e/benchmark_gt/lamaria/hard/R_10_hard.txt \
  benchmark/lamaria/hard/pose/rovio/R_10_hard/rovio_poses.txt \
  --gt-format tum --est-format tum \
  --output-root /mnt/e/epa_perf_sim/outputs
```

## 原因分析

`epa` 和 `epa_ov_error_singlerun` 做的事情不一样：

- `epa_ov_error_singlerun` 是轻量指标命令，主要计算单次 alignment/evaluation 并输出 terminal metrics；
- `epa` 是完整主流程，会生成 plots、HTML interactive report、CSV、metrics JSON、reports 等输出；
- 修复前，`epa` 的 interactive HTML 还会无条件额外计算多个 trajectory view：
  - `EPA PosYaw`
  - `OV Sim3`
  - `Sim3`
  - `EPICA Sim3`
  - `EPICA Stable Sim3`

这些额外 view 对调试有帮助，但默认主命令不应该全部计算。尤其 `epica_sim3_stable` 比较贵，会让普通 `epa` 在没有请求 Sim3 的情况下也被拖慢。

另外，师兄使用的是 `/mnt/e` Windows 挂载盘。WSL 访问 Windows 盘时，IO 和大量小文件写入可能明显慢于 `/home/yifu` 下的 Linux 文件系统。这会进一步放大 `epa` 的输出成本。

## 修复方式

修改 `src/epa/core/pipeline_modular.py`：

1. 新增 `_default_interactive_view(...)` 和 `_interactive_eval_view_specs(...)`。
2. 默认 `eval_align=none` 时，只保留 `step3` interactive view。
3. 只有用户显式请求某个 eval alignment 时，才生成对应的额外 view。
4. 默认 step3 report 复用前面已经算好的 metrics，避免重复计算。

也就是说：

- 默认 `epa`：只生成 Step3 view；
- `--eval-align sim3`：只额外生成 Sim3 view；
- `--eval-align epica_sim3_stable`：只额外生成 EPICA Stable Sim3 view；
- `--eval-align posyaw`：只额外生成 PosYaw view。

## 修复后计时

在本机使用同一个 Lamaria `R_10_hard / rovio` case 测试。

### 本地 `/home/yifu` 路径

| 命令 | 修复前 | 修复后 |
| --- | ---: | ---: |
| `epa` 默认 | 26.8s | 19.2s |
| `epa --no-plot` | 17.9s | 11.9s |
| `epa_ov_error_singlerun se3` | 约 15s | 14.5s |

### 模拟师兄 `/mnt/e` 路径

按师兄目录结构模拟：

| 命令 | 耗时 |
| --- | ---: |
| `epa` 默认，GT 在 `/mnt/e`，output 在 `/mnt/e` | 20.4s |
| `epa_ov_error_singlerun se3`，GT 在 `/mnt/e` | 12.3s |
| `epa --no-downsample`，GT/output 在 `/mnt/e` | 20.1s |
| `epa --eval-align sim3`，GT/output 在 `/mnt/e` | 20.6s |

额外测试：

| 场景 | 耗时 |
| --- | ---: |
| `/mnt/e` input + `/mnt/e` output | 20.8s |
| `/mnt/e` input + `/home` output | 24.1s |
| `/mnt/e` input + `--no-plot` | 19.4s |

当前环境没有复现师兄的 `2:16`，修复后更接近：

```text
epa: 20s 左右
epa_ov_error_singlerun: 12s 左右
```

## 测试

新增测试：

```text
test_interactive_eval_views_only_include_requested_alignment
```

覆盖：

- 默认 `none / epa_step3` 不生成额外 interactive eval view；
- `sim3` 只生成 Sim3 view；
- `epa_sim3_v2` 只生成对应 Sim3 view；
- `epica_sim3_stable` 只生成 Stable Sim3 view；
- `posyaw` 只生成 PosYaw view。

相关测试：

```bash
pytest tests/unit/test_step3_mode_selection.py tests/unit/test_metric_plots.py
```

结果：

```text
22 passed
```

全量单元测试：

```bash
pytest tests/unit
```

结果：

```text
226 passed, 2 warnings
```

## 结论

这个 issue 的大方向成立：`epa` 主命令确实比 `epa_ov_error_singlerun` 更重，因为它会生成完整报告和可视化输出。

旧版还存在一个额外问题：interactive report 会默认计算多个没有被请求的 alignment view，导致普通 `epa` 被 Sim3/Stable Sim3 等额外计算拖慢。这个问题已修复。

修复后，在当前环境中没有复现师兄的 2 分 16 秒；按师兄路径模拟约为 20 秒。剩余差距主要来自 `epa` 的报告/plot/HTML/CSV 输出，以及 `/mnt/e` Windows 挂载盘的 IO 开销。快速测试时可以使用：

```bash
epa ... --no-plot
```

并尽量把输入和输出放在 `/home/yifu` 下，而不是 `/mnt/e`。

# Issue5: `epa_ov_error_singlerun se3` 在 estimated poses 已经带 extrinsics 时失败

## 师兄反馈的问题

测试数据：

```text
GT:  ~/datasets/euroc_mav/gt/machine_hall/MH_05_difficult.txt
EST: estimated_poses_mm.txt
```

命令：

```bash
epa_ov_error_singlerun se3 \
  ~/datasets/euroc_mav/gt/machine_hall/MH_05_difficult.txt \
  estimated_poses_mm.txt
```

0.1.11 输出正常：

```text
rmse_ori = 1.870 | rmse_pos = 0.295
1s time - rmse_ori = 0.876 | rmse_pos = 0.120 (31 samples)
SR@5.0m - distance = 100.00% | time = 100.00%
Aligned pairs: 116
```

0.1.12 输出异常：

```text
rmse_ori = 90.086 | rmse_pos = 0.438
1s time - rmse_ori = 20.818 | rmse_pos = 1.245 (7191 samples)
SR@5.0m - distance = 90.33% | time = 96.48%
Aligned pairs: 7253
Eval source = epa_step3
```

主要异常是 orientation error 从约 `1.87 deg` 变成约 `90 deg`，同时 aligned pairs 从 `116`
变成 `7253`。

## 复现

使用师兄提供的 `estimated_poses_mm.txt` 和本地 EuRoC GT：

```bash
epa_ov_error_singlerun se3 \
  /home/yifu/epa_data/AlignAnything2/AlignAnything2/GT/euroc_mav/MH_05_difficult.txt \
  tmp_issue5/estimated_poses_mm.txt
```

修复前可以复现 0.1.12 的异常结果：

```text
rmse_ori = 90.086 | rmse_pos = 0.438
1s time - rmse_ori = 20.818 | rmse_pos = 1.245 (7191 samples)
Aligned pairs: 7253
Eval source = epa_step3
```

同时用隔离环境安装 `epica==0.1.11` 后，确认 0.1.11 输出和师兄给的一致：

```text
rmse_ori = 1.870 | rmse_pos = 0.295
1s time - rmse_ori = 0.876 | rmse_pos = 0.120 (31 samples)
Aligned pairs: 116
```

## 原因

问题不是单纯的 quaternion 顺序，也不是 `se3` alias 本身。

真正的 regression 是 0.1.12 的 `epa_ov_error_singlerun se3` 在 EPA Step3 路径里改用了
`_prepare_solve_eval_trajectories(...)`。这个函数会把 sparse estimate 插值/同步到 dense GT
轨迹上。

对于这个 case：

```text
0.1.11: 使用 OV-style sparse association，aligned pairs = 116
0.1.12: 插值到 dense GT evaluation trajectory，aligned pairs = 7253
```

当 estimated poses 本身已经带有/应用了 extrinsics，且 est 频率明显低于 GT 时，0.1.12
这种 dense 插值会让 EPA Step2/Step3 在大量插值点上重新估外参和世界对齐，导致姿态评估被破坏，
最终出现约 `90 deg` 的 orientation error。

所以这个 issue 的本质是：

```text
OV compatibility CLI 不应该把 sparse estimate densify 到 GT 全频率；
它应该保持 0.1.11 的 sparse timestamp association 行为。
```

## 修复方式

修改 `src/epa/ov_eval_compat.py` 中 `_evaluate_pair_epa_step3(...)`：

1. Step1 仍然用于估计时间 offset；
2. 之后恢复 0.1.11 行为，使用 `_associate_est_gt(...)` 按 est 时间戳匹配 GT；
3. Step2/Step3 只在这些 matched sparse pairs 上求解和评估；
4. 不再使用 `_prepare_solve_eval_trajectories(...)` 把 sparse estimate 插值到 dense GT。

保留 `epa_downsample_hz` 和 `epa_quat_interp` 参数是为了 CLI/API 兼容，但在这个 OV-compatible
Step3 sparse association 路径里不再改变 evaluation pair 数。

## 修复后结果

修复后同一命令输出：

```text
======================================
Absolute Trajectory Error
======================================
rmse_ori = 1.870 | rmse_pos = 0.295
mean_ori = 1.777 | mean_pos = 0.279
min_ori  = 0.669 | min_pos  = 0.035
max_ori  = 3.380 | max_pos  = 0.482
std_ori  = 0.584 | std_pos  = 0.096
======================================
Relative Pose Error
======================================
seg 8 - median_ori = 0.950 | median_pos = 0.272 (91 samples)
seg 16 - median_ori = 1.034 | median_pos = 0.449 (77 samples)
seg 24 - median_ori = 1.035 | median_pos = 0.546 (71 samples)
seg 32 - median_ori = 1.156 | median_pos = 0.578 (71 samples)
seg 40 - median_ori = 0.915 | median_pos = 0.644 (58 samples)
1s time - rmse_ori = 0.876 | rmse_pos = 0.120 (31 samples)
SR@5.0m - distance = 100.00% | time = 100.00%
Aligned pairs: 116
Eval source = epa_step3
```

该结果和 0.1.11 对齐。

## 新增测试

新增测试：

```text
test_evaluate_pair_epa_step3_keeps_sparse_estimate_association
```

测试构造：

- dense GT：`100 Hz`
- sparse estimate：`1 Hz`
- 两者 pose 本身一致

断言：

```text
_evaluate_pair_epa_step3(...) 的 matched 数量等于 sparse estimate 数量，
不能膨胀到 dense GT 数量。
```

这样可以防止以后 OV-compatible Step3 再次把 sparse estimate densify 成 dense GT。

## 测试结果

相关测试：

```bash
pytest tests/unit/test_ov_eval_compat.py
```

结果：

```text
18 passed
```

全量单元测试：

```bash
pytest tests/unit
```

结果：

```text
227 passed, 2 warnings
```

## 结论

这个 issue 确实存在。0.1.12 在 OV compatibility 的 EPA Step3 路径里错误地把 sparse estimate
插值到 dense GT 轨迹，导致已经带 extrinsics 的估计轨迹被重新外参对齐并破坏 orientation。

修复后，`epa_ov_error_singlerun se3` 恢复 0.1.11 的 sparse association 行为，师兄提供的
EuRoC MH_05 case 结果恢复正常。
