# Quick Start

## Requirements

- Python 3.10 or newer

## Install

Create a clean environment and install the package:

```bash
conda create -n epa python=3.10 -y
conda activate epa
python -m pip install epica
```

## Run One Pair

Use `epa` when you have one reference trajectory and one estimated trajectory:

```bash
epa <gt_file> <est_file>
```

Example with the included files:

```bash
epa example_data/example_groundtruth.csv example_data/example_estimation.txt
```

`epa` runs the full alignment pipeline, detects trajectory formats automatically, writes plots by default, and exports a compact `metrics.json`. If format detection is wrong, set `--gt-format` and `--est-format` explicitly.

## Check Outputs

Each run creates a timestamped directory under `outputs/`:

```text
outputs/run_YYYYmmdd_HHMMSS/
├── metrics.json
├── metrics_summary.csv
├── report_en.md
├── report_zh.md
└── plots/
    ├── step1_time_alignment.png
    └── step23_trajectory_alignment_3d.png
```

Start with `report_en.md` for a readable summary, `metrics_summary.csv` for table-friendly numbers, and the two plots below for the main alignment diagnostics. Add `--save-full-metrics` only if you need full per-sample arrays for custom analysis.

![Step 1 time alignment result](images/quickstart_step1_time_alignment.png)

*Step 1: rotational signals after temporal alignment.*

![Step 2/3 trajectory alignment result](images/quickstart_step23_alignment_3d.png)

*Step 2 and Step 3: aligned trajectories in 3D.*

## Run A Benchmark

Use `epa_bench` when you have many cases organized under one benchmark root:

```bash
epa_bench <cases_root>
```

Example:

```bash
epa_bench /home/username/epa_data/benchmark_cases
```

`epa_bench` runs cases in parallel by default with `--jobs auto`, writes one folder per case, and collects the results into a benchmark summary:

```text
outputs/<benchmark_name>_bench/run_YYYYmmdd_HHMMSS/
├── summary.csv
├── summary.md
├── unresolved_cases.csv
├── cases/
├── logs/
└── paper_tables/
```

Temporary prepared trajectory files are removed by default after the benchmark finishes. Add `--keep-prepared` only if you want to inspect those intermediate files later.

## Next Steps

- Go to [CLI Reference](cli.md) for command-level options.
- Go to [Benchmark Workflow](benchmark_workflow.md) for batch evaluation details.
- Go to [Troubleshooting](troubleshooting.md) if your first run fails.
