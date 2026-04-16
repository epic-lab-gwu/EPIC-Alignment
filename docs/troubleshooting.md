# Troubleshooting

This page covers the most common local `epa` problems and the fastest ways to fix them.

## Before You Debug

Start with:

```bash
epa --help
epa_traj --help
```

For docs issues:

```bash
mkdocs build --strict
```

Also confirm that you are using the intended Python environment.

## `epa: command not found`

Cause:

- the package is not installed in the current environment
- you are using the wrong Python or conda environment

Fix:

```bash
pip install -e .
```

If you use the local conda environment:

```bash
conda activate epa
pip install -e .
```

## `mkdocs serve` fails with `No module named 'pymdownx'`

Cause:

- the local docs environment is missing `pymdown-extensions`

Fix:

```bash
pip install mkdocs pymdown-extensions
```

Then retry:

```bash
mkdocs serve
```

## Markdown Preview Does Not Show Math Formulas

Cause:

- your IDE's plain Markdown preview usually does not read `mkdocs.yml`
- MathJax rendering is configured for the generated documentation site, not raw `.md` preview

Use this instead:

```bash
mkdocs serve
```

Then open the local docs site in a browser and inspect the rendered page there.

## `GT CSV not found` or `Estimation trajectory file not found`

Cause:

- the path is wrong
- the file exists, but not relative to the current repository root

Check:

```bash
ls gt.csv
ls outputs/traj_estimate_v1_01.txt
```

Fix:

- use a correct relative path from the repository root
- or pass an absolute path

## `Real mode requires --est-path`

Cause:

- you ran the main pipeline without providing an estimation trajectory

Fix:

```bash
epa \
  --engine modular \
  --gt-csv gt.csv \
  --est-path outputs/traj_estimate_v1_01.txt \
  --est-format tum
```

## `Trajectory association produced only X matches`

Cause:

- the reference and estimation timestamps do not align closely enough
- `t_max_diff` is too small
- `t_offset` is wrong
- the trajectories do not overlap enough in time

Try this:

1. increase `t_max_diff` slightly
2. check whether a nonzero `t_offset` is needed
3. verify the two files actually cover the same time span

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
- the trajectories may come from different segments or different runs

Check:

- `t_start`
- `t_end`
- `t_offset`
- whether the selected GT and estimation files actually belong together

## `Time window kept fewer than 2 trajectory samples`

Cause:

- your `t_start` and `t_end` window is too narrow
- the selected file contains too few valid samples after filtering

Fix:

- widen the time window
- remove `t_start` and `t_end` temporarily and retry

## `Not enough rotational excitation to estimate extrinsic rotation`

Cause:

- the motion sequence does not contain enough rotation to constrain the extrinsic solve

In practice:

- the trajectory may be too short
- the sensor motion may be nearly straight-line or too static

Try:

- use a longer or more dynamic sequence
- verify that the input trajectory is not heavily downsampled

## `No valid translation constraints for extrinsic translation solve`

Cause:

- the motion is too weak or too degenerate for the translation solve
- consecutive motion increments do not provide enough usable constraints

Try:

- use a more informative sequence
- verify that the input trajectory is valid and not nearly constant

## `Topic not found in bag` or `No trajectory messages found for topic`

Cause:

- the topic name is wrong
- the topic exists, but does not carry a supported pose-like message type
- the topic has no usable trajectory messages

Check:

- `--gt-topic`
- `--est-topic`
- whether the bag actually contains pose, odometry, or TF-style trajectory data

For bag inputs, provide the topic explicitly unless your workflow already discovers it.

## `Bag trajectory loading requires a non-empty topic`

Cause:

- you passed `bag`, `bag2`, or `mcap` without a topic

Fix:

```bash
epa \
  --gt-csv /path/to/run.bag \
  --gt-format bag \
  --gt-topic /vicon/pose \
  --est-path /path/to/run.bag \
  --est-format bag \
  --est-topic /odom
```

## `Unsupported format` or `Cannot infer trajectory format`

Cause:

- the file type does not match the selected format
- `--format auto` could not determine the correct parser

Fix:

- specify the format explicitly
- confirm that the file contents actually match the format you selected

Examples:

```bash
epa_traj --format tum gt.tum est.tum --plot
epa --gt-format csv --est-format tum --gt-csv gt.csv --est-path est.tum
```

## `--align and --align_origin cannot be used together`

Cause:

- these two alignment modes are mutually exclusive

Fix:

- choose only one of them for a given run

## `Umeyama alignment requires at least 3 paired 3D points`

Cause:

- too few matched trajectory samples remain after synchronization or filtering

Try:

- increase `t_max_diff`
- reduce filtering
- use a longer trajectory segment

## `--ros_map_yaml currently requires --plot_mode xy`

Cause:

- map overlays currently work only for XY plots

Fix:

```bash
epa_ape tum gt.tum est.tum \
  --plot \
  --plot_mode xy \
  --ros_map_yaml /path/to/map.yaml
```

The same restriction applies to `--map_tile`.

## `Config file not found`, `Config JSON must be an object`, or `Unknown config key`

Cause:

- the config file path is wrong
- the JSON structure is invalid
- the config contains keys not recognized by the selected tool

Fix:

- confirm the path exists
- ensure the file is a JSON object at the top level
- generate a template first if needed

Useful command:

```bash
epa_config generate --tool epa_ape --out ape_config.json
```

## Docs Build Fails with `--strict`

Common causes:

- broken Markdown links
- nav entries pointing to missing files
- syntax problems after page renames

Check with:

```bash
mkdocs build --strict
```

## GitHub Pages Build Fails but Local Build Works

Common causes:

- local environment has docs dependencies that CI does not
- a recent `mkdocs.yml` change was not mirrored in `.github/workflows/docs.yml`

Check:

- `mkdocs.yml`
- `.github/workflows/docs.yml`
- whether any new Markdown extension needs to be installed in CI

## Still Stuck

If the failure is still unclear, collect:

- the exact command you ran
- the exact error message
- the input file paths
- the selected format and topic
- any non-default `t_max_diff`, `t_offset`, `t_start`, or `t_end` values

For runtime debugging, `epa_traj` is often the fastest way to inspect whether the trajectories load and synchronize correctly before running the full pipeline.
