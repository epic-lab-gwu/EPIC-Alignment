# Evaluation Inputs

This page describes the stored trajectory files used as evaluation inputs in this repository.

`epa` evaluates trajectory outputs, but the estimators that produce those trajectories are maintained outside this repository.

## What This Page Covers

The files listed here are typically:

- estimated trajectories exported from an external estimator
- reused inputs for examples or reproducible evaluation runs
- stored files that keep benchmarks and documentation examples repeatable

This page does not document estimator source code. It tracks the evaluation-side files consumed by `epa`.

## Example: `sqrtVINS`

`sqrtVINS` is a good example of the intended workflow.

The estimator itself is maintained outside `epa`, but one of its exported trajectory results is stored here:

- `outputs/traj_estimate_v1_01.txt`

Typical workflow:

1. run `sqrtVINS` in its own workspace
2. export the estimated trajectory in a format such as `tum`
3. copy the trajectory file into this repository only if it is needed for reproducible evaluation
4. evaluate it with `epa` against the matching ground-truth trajectory

For example:

```bash
epa \
  --engine modular \
  --gt-csv gt.csv \
  --est-path outputs/traj_estimate_v1_01.txt \
  --est-format tum \
  --t-max-diff 0.02 \
  --plot
```

In this setup:

- `sqrtVINS` is the external producer of the estimated trajectory
- `epa` is the evaluation and analysis tool
- the stored trajectory file is a stable, reusable evaluation input

## Tracked Input Files

| file | format | source estimator | dataset hint | notes |
|---|---|---|---|---|
| `outputs/traj_estimate_v1_01.txt` | `tum` | `sqrtVINS` (external workspace) | EuRoC `V1_01_easy` | Used by repository examples and quick-start style runs |

## Usage in This Repository

These files are commonly used for:

- quick examples in the documentation
- manual local testing
- repeatable comparisons across code changes

In most cases, the matching reference trajectory is stored separately and passed with the estimation file into `epa`, `epa_ape`, `epa_rpe`, or benchmark tools.

## Update Policy

- Keep estimator source code outside this repository
- Store only the trajectory outputs needed for reproducible evaluation
- Prefer stable, documented input files over temporary experiment dumps
- When replacing or adding a tracked input, update this page in the same commit

## What Not to Store

Avoid using this repository as a general dump for:

- full estimator workspaces
- raw training outputs
- unrelated experiment logs
- large temporary files that are not part of reproducible evaluation
