# vicon_ws
2026/4/14

解决了昨天大部分todo, 代码已推送, 文档已部署至[Documentation website](https://epic-lab-gwu.github.io/vicon_ws/)

---
2026/4/13

参考evo, 给vicon_ws
1. 补齐了工具链的接口
2. 做了可安装化: pyproject.toml + entry points
3. 加了各种指标的可视化和绘图
4. 配置好了rerun, 增加了一些细节(增加轨迹说明标签, 只对step3轨迹做渐变色映射)
5. 增加了一些todo:

- [ ] 起个名字, 方便代码统一
- [x] 弄得差不多了, 可以做个wiki网站, 然后完善一下readme

---
2026/4/12
1. 扩展了输入格式, 但有一些格式还没实测能否work, 比如rosbag
2. 优化了部分代码结构, 今天晚点上传
3. evo的time offset换成非vicon_ws的offset, 重新跑了师兄给的case, 结果更新至summary.md
4. 总共跑出来的有138个case, 有一些case fail了, 后面再看看原因; 有一些case没有匹配到对应的ground truth和estimation, 没跑

---
2026/4/10

## TODO:

- [x] 重写了项目结构, 使其易于扩展和复用, 也保留了上个版本的pipeline.py;
- [x] 把evo的metrics加了进来, 再看看有什么可以新增的metrics;
- [x] 用现在的vicon_ws和evo分别跑了alignanything, 结果在vicon_ws/summary_final.md;
- [x] evo支持更多的格式输入, vicon后面可以加上 # 4/12
- [x] 可考虑模仿evo, 补齐成一个完整的工具链cli # 4/13
- [x] 指标可视化支持 rerun（APE/RPE/traj/3-step） # 4/13
- [x] rerun 配置补齐（工具入口支持 `--rerun` 与 `--rerun-rec-id`） # 4/13
- [x] 现在evo的结果用了vicon_ws的offset, 后面可以改成人工sweep最优 # 4/12

---
`vicon_ws` 是一个轨迹对齐与评估工具集，包含：

- 三步对齐主流程（时间对齐 -> 外参求解 -> 世界系对齐）
- evo 风格的轨迹工具链（`traj / ape / rpe / res / config`）
- 可选的 Rerun 可视化
- AlignAnything 独立 benchmark harness

核心算法代码在 `src/vicon_ws/core`，`pipeline.py` 仅作为兼容入口（thin shim）。

## 安装

```bash
pip install -e .
```

可选依赖：

```bash
# Rerun 可视化
pip install -e .[rerun]

# bag / bag2 / mcap
pip install -e .[ros]

# 地图底图（contextily）
pip install -e .[geo]

# 开发与测试
pip install -e .[dev]

# IPython 交互入口（可选）
pip install -e .[ipython]
```

## 命令入口

安装后可直接使用：

- `vicon_ws`
- `vicon_ws_traj`
- `vicon_ws_ape`
- `vicon_ws_rpe`
- `vicon_ws_fig`
- `vicon_ws_res`
- `vicon_ws_config`
- `vicon_ws_benchmark`
- `vicon_ws_plot_summary`
- `vicon_ws_metric_res`
- `vicon_ws_ipython`

兼容旧入口：

```bash
python pipeline.py --help
```

## 快速开始

### 1) 三步主流程（modular engine）

```bash
vicon_ws \
  --engine modular \
  --gt-csv gt.csv \
  --est-path outputs/traj_estimate_v1_01.txt \
  --est-format tum \
  --t-max-diff 0.02 \
  --plot
```

启用 Rerun：

```bash
vicon_ws \
  --engine modular \
  --gt-csv gt.csv \
  --est-path outputs/traj_estimate_v1_01.txt \
  --est-format tum \
  --t-max-diff 0.02 \
  --plot \
  --rerun
```

### 2) 轨迹工具 `vicon_ws_traj`

```bash
# 轨迹对比绘图
vicon_ws_traj --format tum --plot --plot-mode xz gt.tum est.tum

# evo 风格子命令也兼容
vicon_ws_traj tum gt.tum est.tum --plot

# bag 的 evo 风格 topics 位置参数也兼容
vicon_ws_traj bag /path/run.bag /vicon/pose /odom --plot

# 交互式窗口（evo 风格）
vicon_ws_traj --format tum --plot --plot-interactive --plot-backend qtagg gt.tum est.tum

# 对齐与同步
vicon_ws_traj --format tum --sync --align --ref 1 gt.tum est.tum --plot

# 导出格式
vicon_ws_traj --format auto --save-as tum --out-dir outputs/traj_exports traj_a traj_b
```

### 3) APE / RPE 工具

```bash
# APE
vicon_ws_ape tum gt.tum est.tum \
  --pose_relation trans_part \
  --align \
  --t_max_diff 0.02 \
  --plot --plot_mode xz

# RPE
vicon_ws_rpe tum gt.tum est.tum \
  --pose_relation trans_part \
  --delta 1 --delta_unit f \
  --all_pairs \
  --align \
  --plot --plot_mode xz

# 若需要交互式窗口，可附加：
# --plot 默认在 TTY 终端会尝试弹交互窗口（evo 风格）；
# 也可显式指定：--plot-interactive --plot-backend qtagg
```

### 4) 结果对比 `vicon_ws_res`

```bash
vicon_ws_res outputs/results/run_a.zip outputs/results/run_b.zip \
  --metric all --stage step3 --plot --out-dir outputs/res_compare

# 交互式窗口
vicon_ws_res outputs/results/run_a.zip outputs/results/run_b.zip \
  --metric all --stage step3 --plot --plot-interactive --plot-backend qtagg

# 也支持 evo 原生结果包（info.json + stats.json + error_array.npz）
vicon_ws_res /home/yifu/evo/test/data/res_files/orb_ape.zip \
             /home/yifu/evo/test/data/res_files/sptam_ape.zip \
             --metric ape --ape-relation trans_part --plot
```

### 5) 图序列化与重绘 `vicon_ws_fig`

```bash
# 先在 ape/rpe 中序列化绘图规格
vicon_ws_ape tum gt.tum est.tum \
  --pose_relation trans_part \
  --serialize_plot outputs/ape_plot.json

# 后续可独立重绘（不重算指标）
vicon_ws_fig outputs/ape_plot.json --save_plot outputs/ape_rerender.png
```

### 6) 全局配置 `vicon_ws_config`

```bash
# 根级默认（所有工具都可继承）
vicon_ws_config set plot false rpe_delta 3

# 工具级默认（更细粒度，推荐）
vicon_ws_config set --tool vicon_ws_ape plot_mode xy t_max_diff 0.05
vicon_ws_config set --tool vicon_ws_traj plot_mode xyz sync_max_diff 0.01

# 查看某个工具生效配置（global + tool 合并后）
vicon_ws_config show --tool vicon_ws_ape

vicon_ws_config show
vicon_ws_config unset plot
```

## 配置系统

所有核心工具支持 `--config <file.json>`。

优先级（高 -> 低）：

1. `--config` 文件
2. CLI 参数
3. `vicon_ws_config` 全局配置

生成模板：

```bash
vicon_ws_config generate --tool vicon_ws_ape --out ape_config.json
```

也支持分层配置（类似 evo settings）：

```json
{
  "_global": {
    "plot": true
  },
  "vicon_ws_ape": {
    "plot_mode": "xy",
    "t_max_diff": 0.05
  },
  "vicon_ws_traj": {
    "plot_mode": "xyz",
    "sync_max_diff": 0.01
  }
}
```

## 支持输入格式

主流程与工具链支持：

- `auto`
- `csv` / `euroc`
- `tum`
- `kitti`
- `bag`（ROS1）
- `bag2` / `mcap`（ROS2）

bag 读取时可指定 topic：

```bash
vicon_ws \
  --gt-csv /path/to/run.bag --gt-format bag --gt-topic /vicon/pose \
  --est-path /path/to/run.bag --est-format bag --est-topic /odom
```

TF 语法也支持：

```bash
--est-topic /tf:map.base_link
```

## Rerun 说明

`vicon_ws` / `traj` / `ape` / `rpe` 都支持 `--rerun`。

常用参数：

- `--rerun`
- `--rerun-rec-id <id>`
- `--rerun-no-spawn`
- `--rerun-stride N`
- `--rerun-motion-stride N`

说明：行为与 evo 一致，默认仅 viewer logging，不自动写 `.rrd`。

## IPython 入口

```bash
# 进入预加载 vicon_ws 模块的 IPython
vicon_ws_ipython

# 先查看预加载符号
vicon_ws_ipython --list
```

## Benchmark（AlignAnything）

运行独立 benchmark harness（`vicon_ws` 与 `evo` 独立运行，offset 不共享）：

```bash
vicon_ws_benchmark \
  --alignanything-root /home/yifu/vicon_ws/AlignAnything/AlignAnything \
  --repo-root /home/yifu/vicon_ws \
  --evo-repo /home/yifu/evo
```

从 harness 的 `summary.csv` 生成图：

```bash
vicon_ws_plot_summary \
  --summary-csv outputs/alignanything_harness/run_xxx/summary.csv
```

## 输出目录

典型输出：

- 主流程：`outputs/run_YYYYmmdd_HHMMSS/`
- traj：`outputs/traj_tool/run_YYYYmmdd_HHMMSS/`
- ape：`outputs/ape/run_YYYYmmdd_HHMMSS/`
- rpe：`outputs/rpe/run_YYYYmmdd_HHMMSS/`
- res：`outputs/res/run_YYYYmmdd_HHMMSS/`
- harness：`outputs/alignanything_harness/run_YYYYmmdd_HHMMSS/`

主流程目录常见文件：

- `metrics.json`
- `metrics_summary.csv`
- `metrics_zh.md`
- `step1_cross_correlation.png`
- `step1_time_alignment.png`
- `step23_trajectory_alignment_3d.png`
- `plots/*.png`

## 代码结构

```text
vicon_ws/
├── pipeline.py                    # 兼容入口（thin shim）
├── pyproject.toml
├── src/vicon_ws/
│   ├── cli.py                     # vicon_ws 主 CLI
│   ├── runner.py                  # engine 分发
│   ├── bridge_legacy.py           # legacy 桥接
│   ├── config.py                  # PipelineOptions
│   ├── config_cli.py              # --config / 全局配置注入
│   ├── config_tool.py             # vicon_ws_config
│   ├── traj_tool.py               # vicon_ws_traj
│   ├── ape_tool.py                # vicon_ws_ape
│   ├── rpe_tool.py                # vicon_ws_rpe
│   ├── metric_cli_common.py       # APE/RPE 共享 CLI 逻辑
│   ├── core/
│   │   ├── pipeline_modular.py    # modular 三步主流程
│   │   ├── time_alignment.py
│   │   ├── calibration.py
│   │   ├── evaluation.py
│   │   ├── math_utils.py
│   │   └── io_utils.py
│   ├── viz/
│   │   ├── rerun_viz.py           # Rerun logging
│   │   └── metric_plots.py        # 指标绘图/聚合图
│   └── benchmark/
│       ├── alignanything_harness.py
│       ├── plot_summary.py
│       ├── metrics_res.py
│       └── res.py                 # vicon_ws_res
├── docs/
│   ├── architecture.md
│   └── evaluation_inputs.md
└── tests/
    ├── unit/
    └── smoke/
```

## 文档站（替代 Wiki）

本项目使用 `docs/ + MkDocs + GitHub Pages` 搭建文档站（不依赖 GitHub Wiki）。

本地预览：

```bash
pip install mkdocs
mkdocs serve
```

默认访问：`http://127.0.0.1:8000`

静态构建：

```bash
mkdocs build --strict
```

站点配置在 `mkdocs.yml`，页面内容在 `docs/`。

已新增自动发布 workflow：`.github/workflows/docs.yml`。

首次启用 GitHub Pages（仓库设置）：

1. `Settings -> Pages`
2. `Build and deployment` 选择 `GitHub Actions`
3. 推送 `main` 分支后会自动发布

## 开发与测试

```bash
ruff check src tests
mypy
pytest -q
```

最小 smoke：

```bash
bash tests/smoke/test_cli_help.sh
bash tests/smoke/test_engines.sh
```

## 相关文档

- `docs/architecture.md`
- `docs/evaluation_inputs.md`
- `track.md`
