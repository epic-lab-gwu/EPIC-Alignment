# EPICA 工程 TODO

> 来源：组会讨论整理。这里只记录 EPA/EPICA 工程相关任务；论文写作和 GT-free 指标研究先不放在这里。

## P0：把 OV / OpenVINS 兼容接口统一切到 EPICA 核心算法

目标：保留 OpenVINS / OV-EVO 风格的输入、输出、表格和命令习惯，但内部算法必须全部走 EPICA。

- [x] 梳理兼容入口：
  - [x] `src/epa/ov_eval_compat.py`
  - [x] `src/epa/openvins_runner.py`
  - [x] `src/epa/case_toolchain.py`
- [x] 找出仍在调用 OV-EVO-style alignment 或 metric 的路径。
- [x] 将兼容入口内部替换为 EPICA 的流程：
  - [x] EPICA time alignment
  - [x] EPICA Step2 / Step3 alignment
  - [x] EPICA APE / RPE / successful rate 计算
- [x] 保留 legacy 命令风格和 summary / latex 输出。
- [x] 禁止 silent fallback 到旧 OV-EVO-style evaluation。
- [ ] 如果短期必须 fallback：
  - [ ] 在结果里写入 `eval_source=fallback_ov_style`
  - [ ] 在 summary CSV / HTML / Markdown 中显示 warning
- [x] 验收：同一个 case 用 `epa` 直接跑和 OV/OpenVINS 兼容入口跑，核心 metrics 一致或差异可解释。
- [x] 验收：EPICA-backed run 的 summary 中明确显示 `eval_source=epa_step3`。

验收记录：`outputs/p0_compat_validation_final/compat_validation.md`。

## P0：定位并修复兼容入口变慢问题

现象：同一个 case，直接用 `epa` 跑大约 10 秒，但用 OV/EVO 兼容入口跑大约 60 秒。

- [x] 复现慢 case，例如 LaMAR / LaMARia hard 20 SVO Mono。
- [x] 给兼容入口加阶段计时或用 profiler 分析：
  - [x] 文件加载和解析
  - [x] timestamp association
  - [x] Step1 time alignment
  - [x] Step2 / Step3 solve
  - [x] APE / RPE / successful rate metrics
  - [x] plotting / report generation
- [x] 检查是否有重复计算：
  - [x] 重复读文件
  - [x] 重复 timestamp association
  - [x] 同时跑 OV-style 和 EPICA evaluation
  - [x] 重复生成 plots / reports
- [x] 修复不必要的重复流程。
- [x] 验收：兼容入口运行时间应与直接 `epa` 同量级。
- [x] 验收：如果有不可避免的额外开销，需要在日志或文档中解释。

验收记录：`outputs/p0_compat_validation_final/compat_validation.md`。`case_toolchain` 默认 EPA-only，不再重复调用 `openvins_runner`；`ov_eval_compat` 的 distance-RPE 匹配已从 O(n²) 改为近邻查找。

## P0：开发 EPICA-native Sim3

问题：旧 Sim3 可能用 scale 把 position error 拉低，从而掩盖已经失败的 trajectory，导致 successful rate 看起来正常。

- [ ] 明确 Sim3 适用场景：
  - [ ] GT 有 metric scale，estimate 没有 scale，例如 visual SLAM
  - [ ] 一个 trajectory 有 scale，另一个没有 scale
- [ ] 明确 Sim3 风险场景：
  - [ ] GT 和 VIO estimate 都有 metric scale 时，Sim3 可能吃掉真实 error
  - [ ] trajectory 已经发散时，Sim3 可能强行把 position error 拉低
- [ ] 设计 EPICA-native Sim3 流程：
  - [ ] 先跑 EPICA Step1 time alignment
  - [ ] 先解决 orientation / extrinsic consistency
  - [ ] 在 orientation 稳定后再估计 translation + scale
  - [ ] 不直接复用 OV-EVO Sim3 作为核心算法
- [ ] 增加 Sim3 可靠性诊断字段：
  - [ ] `sim3_scale`
  - [ ] `sim3_gain_ratio`
  - [ ] `sim3_reliable`
  - [ ] `sim3_failure_reason`
  - [ ] orientation gate status
  - [ ] trajectory jump / divergence status
- [ ] 增加 gate，避免 Sim3 把失败 case 包装成成功：
  - [ ] severe scale mismatch gate
  - [ ] abnormal Sim3-over-SE3 gain gate
  - [ ] orientation RMSE / RPE gate
  - [ ] trajectory jump gate
- [ ] 验收：trajectory 爆炸但被 scale 数值救回来的 case，必须被标记为 unreliable 或 failed。
- [ ] 验收：report 中解释 Sim3 是否可能 masking real failure。

## P1：整理 alignment mode 策略

- [ ] 主要保留三个 mode：
  - [ ] `se3`
  - [ ] `posyaw`
  - [ ] `sim3`
- [ ] 将 `se3single` 和 `posyawsingle` 标记为 legacy / optional。
- [ ] 说明 first-frame alignment 很脆弱，很多时候不可靠。
- [ ] 用 EPICA alignment logic 重新实现 `posyaw`，不要继续走旧 OV-EVO internals。
- [ ] 文档中写清楚推荐用法：
  - [ ] metric VIO：EPICA Step3 或 `se3`
  - [ ] visual-only no-scale SLAM：显式使用 `sim3`
  - [ ] yaw-only benchmark convention：使用 `posyaw`
- [ ] 验收：CLI help 和 docs 说明每个 mode 的语义和风险。
- [ ] 验收：每个主要 mode 都有 unit test 和至少一个 smoke / regression case。

## P1：防止 successful rate 误导

- [ ] successful rate 不能只依赖 Sim3-aligned position error。
- [ ] 同时报告：
  - [ ] raw position successful rate
  - [ ] reliability-gated successful rate
- [ ] 在以下情况增加 warning：
  - [ ] translation SR 很高但 orientation error 极大
  - [ ] Sim3 scale 严重异常
  - [ ] local RPE 看起来好但 global trajectory 不稳定
  - [ ] trajectory jump 或 divergence
- [ ] 将 warning 写入：
  - [ ] `metrics.json`
  - [ ] `metrics_summary.csv`
  - [ ] Markdown reports
  - [ ] benchmark summary HTML
- [ ] 验收：已知 translation SR 高但 orientation 很差的 case 必须被标记。
- [ ] 验收：已知 failed trajectory 不能显示为干净的 `ok` case。

## P1：建立 regression benchmark 集合

- [ ] 收集固定 regression cases：
  - [ ] 之前 low successful rate 的 regression cases
  - [ ] LaMAR / LaMARia hard 20 SVO Mono
  - [ ] Aqualoc / AquaLog 中 scale 或 orientation 可疑的 cases
  - [ ] orientation error 爆炸但 Sim3 降低 position error 的 cases
- [ ] 每个 case 固定跑：
  - [ ] direct `epa`
  - [ ] OV/OpenVINS-compatible entry
  - [ ] `se3`
  - [ ] `posyaw`
  - [ ] `sim3`
- [ ] 每次记录：
  - [ ] runtime
  - [ ] eval source
  - [ ] APE / RPE
  - [ ] successful rate
  - [ ] Sim3 reliability
  - [ ] warnings
- [ ] 做一个命令自动重新生成 regression summary。
- [ ] 验收：未来改 EPICA 时，不能悄悄退回 legacy OV-EVO behavior。

## P2：文档和命名清理

- [ ] 将 legacy interfaces 描述为 “OpenVINS-style inputs/outputs backed by EPICA”。
- [ ] 不要让文档暗示 EPICA 保留 OV-EVO 内部算法。
- [ ] 只在对已有用户有价值时保留 legacy command names。
- [ ] 考虑给 table-generation workflow 增加更清晰的 EPICA-first alias。
- [ ] 在 README / docs 中说明：即使命令风格沿用 OpenVINS legacy，核心算法也是 EPICA。
- [ ] 验收：用户能根据文档判断 VIO / visual SLAM / benchmark case 应该用哪个命令和 alignment mode。
