# EPIC-Alignment (epica)

`epica` is a trajectory alignment and evaluation toolkit.

It provides:

- 3-step alignment pipeline (time offset, extrinsic, world alignment)
- a set of CLI tools (`traj`, `ape`, `rpe`, `res`, `config`)
- OpenVINS compatibility entrypoints
- optional plotting and rerun-based visualization

## Installation

Create and activate a virtual environment first (recommended):

```bash
conda create -n epa python=3.10 -y
conda activate epa
```

Then install:

```bash
pip install epica
```

Optional extras:

```bash
pip install "epica[rerun]"  # rerun visualization
pip install "epica[ros]"    # bag / bag2 / mcap support
pip install "epica[geo]"    # map-related tools
```

## Quick Start

For General Workspace:

```bash
epa --gt-csv <gt_file> --gt-format <gt_format> --est-path <est_file> --est-format <est_format> --plot
```

OpenVINS examples:

Single case:

```bash
epa_openvins /path/to/case_dir se3 --keep-output
```

Multiple cases:

```bash
epa_openvins /path/to/case_a /path/to/case_b --align-mode se3
```

Run from OpenVINS repo root:

```bash
epa_openvins ./ov_eval/example none --keep-output
```

Each case directory should contain:

- `stamped_groundtruth.txt`
- `stamped_traj_estimate.txt`

## Outputs

Each run generates a run folder with:

- `plots/`
- `metrics.json`
- `metrics_summary.csv`
- `report_en.md`
- `report_zh.md`

## Benchmark LaTeX Tables

For benchmark summary CSV files, generate paper tables with:

```bash
epa_latex_summary --summary-csv /path/to/summary.csv
```

This creates:

- `main_table.tex`
- `dataset_table.tex`
- `appendix_full_table.tex`


## Full CLI Toolchain

- `epa` / `epica` / `epic-alignment`: Run the main 3-step EPA pipeline for alignment and metric export; usage: `epa --gt-csv gt.tum --gt-format tum --est-path est.tum --est-format tum --plot`.
- `epa_config`: Manage global or per-tool default config values; usage: `epa_config set --tool epa_ape plot_mode xy`.
- `epa_ape`: Compute APE in EVO-style format with optional alignment and plots; usage: `epa_ape tum gt.tum est.tum --align --plot`.
- `epa_rpe`: Compute RPE in EVO-style format with configurable delta settings; usage: `epa_rpe tum gt.tum est.tum --delta 1 --delta_unit f --plot`.
- `epa_traj`: Compare, align, sync, and visualize trajectories in EVO-style workflow; usage: `epa_traj tum gt.tum est.tum --sync --align --plot`.
- `epa_fig`: Re-render saved plotting specs without recomputing metrics; usage: `epa_fig outputs/ape_plot.json --save_plot outputs/ape.png`.
- `epa_res`: Compare multiple result bundles or EVO zip results; usage: `epa_res run_a.zip run_b.zip --metric all --plot`.
- `epa_benchmark`: Run batch benchmark harness for EPA/EVO-style comparison over many cases; usage: `epa_benchmark --cases-csv cases.csv --out-dir outputs/bench`.
- `epa_plot_summary`: Generate benchmark summary figures from harness outputs; usage: `epa_plot_summary --summary-csv outputs/bench/run_xxx/summary.csv`.
- `epa_latex_summary`: Generate paper-ready LaTeX tables from benchmark `summary.csv`; usage: `epa_latex_summary --summary-csv outputs/bench/run_xxx/summary.csv`.
- `epa_metric_res`: Aggregate or compare one or more `metrics.json` files directly; usage: `epa_metric_res --inputs run1/metrics.json run2/metrics.json`.
- `epa_case_rerun` / `epa_rerun`: Replay one case with rerun-based visual diagnostics; usage: `epa_case_rerun --case-json outputs/bench/run_xxx/cases_json/case_001.json`.
- `epa_ipython`: Launch EPA tools in IPython-friendly entry mode; usage: `epa_ipython`.
- `epa_ov_eval`: OpenVINS `ov_eval` compatibility umbrella entrypoint; usage: `epa_ov_eval --help`.
- `epa_ov_format_converter`: Convert trajectory formats in `ov_eval`-compatible style; usage: `epa_ov_format_converter --help`.
- `epa_ov_plot_trajectories`: Plot trajectories in `ov_eval`-compatible style; usage: `epa_ov_plot_trajectories --help`.
- `epa_ov_error_singlerun`: Run single-run error evaluation in `ov_eval`-compatible style; usage: `epa_ov_error_singlerun --help`.
- `epa_ov_error_dataset`: Run dataset-level error summary in `ov_eval`-compatible style; usage: `epa_ov_error_dataset --help`.
- `epa_ov_error_comparison`: Run algorithm comparison summary in `ov_eval`-compatible style; usage: `epa_ov_error_comparison --help`.
- `epa_openvins`: Run EPA on one or multiple OpenVINS case folders with minimal integration flow; usage: `epa_openvins /path/to/case_dir se3 --keep-output`.

## Project Links

- Source and full project docs: https://github.com/epic-lab-gwu/epa
