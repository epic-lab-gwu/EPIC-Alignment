# Benchmark Workflow

This page covers the batch evaluation tools provided by `epica`, including case discovery, repeated execution, and summary analysis. The main commands on this page are `epa_bench` and `epa_benchall`.

## Benchmark Scope

`epica` includes a benchmark harness for large-scale case evaluation. The main goal is to run the same evaluation workflow over many prepared cases and collect consistent summaries.

In the current repository, the primary benchmark workflow uses a generic cases root that contains `benchmark/` and `GT/`.

## Multi-Case Harness

`epa_bench` runs an independent benchmark over cases discovered from a cases-root directory layout.

It:

- discover benchmark cases from prepared dataset and ground-truth directories
- run `epa` on each case
- write per-case JSON records, logs, and summary tables

## Required Inputs

It is recommended to keep datasets outside the repository (for example, `/home/username/epa_data`) and manage paths through environment variables:

```bash
export EPA_DATA_ROOT=/home/username/epa_data
```

`epa_bench` defaults to `$EPA_CASES_ROOT`; if it is unset, it falls back to `$EPA_ALIGNANYTHING_ROOT`, and then to `$EPA_DATA_ROOT/benchmark_cases`.

Before running the harness, make sure you have:

- a cases root with `benchmark/` and `GT/`
- the `epica` repository root
- a Python 3.10+ executable that can run `epica`

The harness exposes these main path options:

- `--cases-root`
- `--repo-root`
- `--python-bin`
- `--epa-src`

## Basic Run

If you want the main batch benchmark workflow, start here:

```bash
epa_bench \
  --cases-root /home/username/epa_data/benchmark_cases \
  --repo-root /home/username/epa \
  --python-bin /home/username/miniconda3/envs/epa/bin/python
```

This command:

1. discover benchmark cases
2. prepare temporary TUM trajectories for each case
3. run `epa` on each case
4. aggregate the per-case outputs into a summary table

## Full Multi-Case Workflow

Use `epa_benchall` if you want the full batch workflow in one command:

- run the benchmark harness
- generate summary plots
- generate LaTeX tables

Typical command:

```bash
epa_benchall \
  --cases-root /home/username/epa_data/benchmark_cases \
  --repo-root /home/username/epa \
  --python-bin /home/username/miniconda3/envs/epa/bin/python
```

## Per-Case Visualization (Raw / GT / Aligned)

For each dataset case, use the main pipeline with `--rerun` to visualize:

- GT trajectory
- raw trajectory after step-1 time sync
- aligned trajectory after step-3 world alignment

Quick command (recommended):

```bash
epa_rerun \
  --run-dir outputs/<cases_root_name>_bench \
  --case euroc_mav_MH_01_easy_rovio
```

The command resolves the latest `run_*` automatically when `--run-dir` points to the parent harness directory.

Direct `epa` command (manual paths):

```bash
epa \
  --gt-csv /path/to/gt.tum \
  --gt-format tum \
  --est-path /path/to/est.tum \
  --est-format tum \
  --plot \
  --rerun
```

## Useful Filters

Use these options for smaller or more targeted runs:

- `--case-pattern`: regex filter for case IDs
- `--methods`: comma-separated method filter such as `rovio,svo_stereo`
- `--limit`: cap the number of cases
- `--dry-run`: discover cases without executing them

Examples:

List matching cases only:

```bash
epa_bench \
  --cases-root /home/username/epa_data/benchmark_cases \
  --repo-root /home/username/epa \
  --python-bin /home/username/miniconda3/envs/epa/bin/python \
  --case-pattern euroc \
  --dry-run
```

Run only a subset of methods:

```bash
epa_bench \
  --cases-root /home/username/epa_data/benchmark_cases \
  --repo-root /home/username/epa \
  --python-bin /home/username/miniconda3/envs/epa/bin/python \
  --methods rovio,svo_stereo \
  --limit 20
```

## Output Structure

Each harness run creates a fresh run directory under the benchmark output root.

Typical layout:

```text
outputs/<cases_root_name>_bench/run_YYYYmmdd_HHMMSS/
```

Typical directory tree:

```text
outputs/<cases_root_name>_bench/run_YYYYmmdd_HHMMSS/
├── harness_config.json
├── summary.csv
├── summary.md
├── unresolved_cases.csv
├── cases/
│   └── *.json
├── logs/
│   └── ...
├── paper_tables/
│   ├── main_table.tex
│   ├── dataset_table.tex
│   └── appendix_full_table.tex
└── prepared_tum/
    └── ...
```

Common contents:

- `summary.csv`: machine-readable per-case summary
- `summary.md`: human-readable summary table
- `paper_tables/main_table.tex`: paper-ready short LaTeX table
- `paper_tables/dataset_table.tex`: dataset-level aggregate LaTeX table
- `paper_tables/appendix_full_table.tex`: full per-case LaTeX longtable for appendix
- `cases/*.json`: one JSON file per case
- `logs/`: stdout and stderr logs for executed tools
- `prepared_tum/`: prepared TUM files used by the harness
- `harness_config.json`: the run configuration snapshot
- `unresolved_cases.csv`: discovered but unresolved cases, when applicable

## Key Columns in `summary.csv`

Important fields include:

- `case`, `dataset`, `method`
- `status`, `epa_status`
- `epa_offset_est_s`
- `epa_ate_rmse_raw_m`, `epa_ate_rmse_step3_m`
- `epa_improve_pct`
- `epa_matches_equivalent`
- `epa_run_dir`
- `error`

These fields cover per-case metrics, output locations, and failure diagnosis.

## Summary Plot Generation

Use `epa_plot_summary` to generate charts from a harness `summary.csv`.

Example:

```bash
epa_plot_summary \
  --summary-csv outputs/<cases_root_name>_bench/run_xxx/summary.csv
```

By default, plots are written to:

```text
<summary_dir>/plots/
```

Typical figures:

- status count chart
- top-case aligned RMSE chart
- other benchmark summary figures under `<summary_dir>/plots/`

Useful option:

- `--top-k`: number of cases included in the top-case bar chart

## LaTeX Table Generation

Use `epa_latex_summary` to generate paper-ready LaTeX tables from a harness `summary.csv`.

```bash
epa_latex_summary \
  --summary-csv outputs/<cases_root_name>_bench/run_xxx/summary.csv
```

By default, tables are written to `<summary_dir>/paper_tables/` with:

- `main_table.tex`: compact main-paper table (automatically compressed when cases are too many)
- `dataset_table.tex`: dataset-level aggregate table
- `appendix_full_table.tex`: full longtable for appendix

`epa_bench` also generates this `paper_tables/` directory automatically at the end of each run.

### Use in Overleaf (2 ways)

1. Directly drag files into your Overleaf project

- Drag `paper_tables/*.tex` into the project root (or a subfolder).
- In your manuscript preamble, add `\usepackage{booktabs}` and `\usepackage{longtable}`.
- Insert tables where needed (if files are in a subfolder such as `tables/`, use relative paths like `\input{tables/main_table.tex}`):

```tex
\input{main_table.tex}
\input{dataset_table.tex}
\input{appendix_full_table.tex}
```

2. Copy and paste table code

- Open the target `.tex` table file and copy the table block into your manuscript body/appendix.
- Keep the same preamble requirements: `\usepackage{booktabs}` and `\usepackage{longtable}`.
- `main_table.tex` and `dataset_table.tex` are `table` environments, while `appendix_full_table.tex` is a `longtable` environment.

## `metrics.json` Aggregation

Use `epa_metric_res` to aggregate or compare one or more `metrics.json` files directly, without using the full harness summary workflow.

Typical example:

```bash
epa_metric_res \
  --metrics-json /path/to/metrics_a.json /path/to/metrics_b.json \
  --mode aggregate \
  --metric all \
  --stage step3
```

Main modes:

- `single`: stage comparison within one run
- `aggregate`: cross-run comparison
- `auto`: infer mode from the number of inputs

Useful options:

- `--metric`
- `--ape-relation`
- `--rpe-relation`
- `--stage`
- `--x-dimension`
- `--out-dir`

## Recommended Workflow

Recommended workflow:

1. Run `epa_bench`
2. Inspect `summary.csv` and `summary.md`
3. Generate charts with `epa_plot_summary`
4. Drill into failed cases using `cases/*.json` and `logs/`
5. Use `epa_metric_res` when you want additional aggregation across selected runs

## Troubleshooting Hints

If a harness run is incomplete or noisy, check these first:

- the Python executable passed with `--python-bin`
- the cases root path
- unresolved cases listed in `unresolved_cases.csv`
- per-case stderr logs under `logs/`

If many cases have poor match counts, revisit:

- `t-max-diff`
- minimum match ratio
