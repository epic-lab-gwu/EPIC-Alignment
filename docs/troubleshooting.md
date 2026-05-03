# Troubleshooting

This page covers the problems users are most likely to hit during installation, first runs, and common trajectory evaluation workflows.

## If You Do Not Want to Debug

No worries. If you would rather not spend time debugging, collect the following information and open an issue here:

- [EPIC-Alignment Issues](https://github.com/epic-lab-gwu/EPIC-Alignment/issues)

Please include:

- the exact command you ran
- the exact error message
- the input file paths
- the selected format and topic
- any non-default `t_max_diff`, `t_offset`, `t_start`, or `t_end` values

## If You Want to Debug

If you want the fastest sanity check, start here:

```bash
epa --help
epa_traj --help
```

Also make sure you are in the Python or conda environment where you installed `epica`.

## `epa: command not found`

Cause:

- `epica` is not installed in the current environment
- you are using a different Python or conda environment from the one where you installed it

Fix:

```bash
conda activate epa
python -m pip install epica
```

If you are working from a local clone and want an editable install instead:

```bash
conda activate epa
python -m pip install -e .
```

## `GT CSV not found` or `Estimation trajectory file not found`

Cause:

- the path is wrong
- the file exists, but not relative to your current working directory

Check:

```bash
ls example_data/example_groundtruth.csv
ls example_data/example_estimation.txt
```

Fix:

- use the correct relative path from the current directory
- or pass an absolute path

This is one of the most common first-run issues, so it is usually just a path mismatch.

## `Provide <gt_file> <est_file>, or use --gt/--est`

Cause:

- the main pipeline was started without both trajectory inputs

Fix:

```bash
epa example_data/example_groundtruth.csv example_data/example_estimation.txt
```

## `Unsupported format` or `Cannot infer trajectory format`

Cause:

- the file type does not match the selected format
- `auto` could not determine the correct parser

Fix:

- specify the format explicitly
- check that the file contents actually match the format you selected

If format auto-detection feels unclear, being explicit is usually the quickest fix.

Examples:

```bash
epa_traj --format tum gt.tum est.tum --plot
epa example_data/example_groundtruth.csv est.tum --gt-format csv --est-format tum
```

## `Topic not found in bag` or `No trajectory messages found for topic`

Cause:

- the topic name is wrong
- the topic exists, but does not contain a supported pose-like message type
- the topic has no usable trajectory messages

Check:

- `--gt-topic`
- `--est-topic`
- whether the bag actually contains pose, odometry, or TF-style trajectory data

This usually means the bag opened fine, but the selected topic was not the one you wanted.

## `Bag trajectory loading requires a non-empty topic`

Cause:

- `bag`, `bag2`, or `mcap` was used without specifying a topic

Fix:

```bash
epa /path/to/run.bag /path/to/run.bag \
  --gt-format bag --gt-topic /vicon/pose \
  --est-format bag --est-topic /odom
```

## `Trajectory association produced only X matches`

Cause:

- the reference and estimation timestamps do not align closely enough
- `t_max_diff` is too small
- `t_offset` is wrong
- the two trajectories do not overlap enough in time

Try:

1. increase `t_max_diff` slightly
2. check whether a nonzero `t_offset` is needed
3. verify that the two files actually cover the same time span

Example:

```bash
epa_ape tum gt.tum est.tum \
  --t_max_diff 0.05 \
  --t_offset 0.0 \
  --plot
```

For a quick inspection before metric computation:

```bash
epa_traj --format tum --sync --ref 1 gt.tum est.tum --plot
```

## `Not enough overlap between GT and estimation for robust time alignment`

Cause:

- after cropping and offset handling, the two trajectories share too little overlap
- the selected files may come from different segments or different runs

Check:

- `t_start`
- `t_end`
- `t_offset`
- whether the selected GT and estimation files actually belong together

## `Time window kept fewer than 2 trajectory samples`

Cause:

- `t_start` and `t_end` define a window that is too narrow
- the selected file contains too few valid samples after filtering

Fix:

- widen the time window
- remove `t_start` and `t_end` temporarily and retry

## `--align and --align_origin cannot be used together`

Cause:

- these alignment modes are mutually exclusive

Fix:

- choose only one of them for a given run

## `Umeyama alignment requires at least 3 paired 3D points`

Cause:

- too few matched trajectory samples remain after synchronization or filtering

Try:

- increase `t_max_diff`
- reduce filtering
- use a longer trajectory segment

## `mkdocs serve` fails with `No module named 'pymdownx'`

Cause:

- the local docs environment is missing `pymdown-extensions`

Fix:

```bash
python -m pip install mkdocs pymdown-extensions
```

Then retry:

```bash
mkdocs serve
```

## Markdown Preview Does Not Show Math Formulas

Cause:

- the IDE's plain Markdown preview does not read `mkdocs.yml`
- MathJax rendering is configured for the generated docs site, not raw `.md` preview

Use this instead:

```bash
mkdocs serve
```

Then open the local docs site in a browser.

For runtime debugging, `epa_traj` is usually the fastest way to check whether trajectories load and synchronize correctly before running the full pipeline.
