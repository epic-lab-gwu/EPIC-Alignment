# Estimation Artifacts

This repo stores trajectory artifacts used for evaluation.
Estimator codebases (e.g. sqrtVINS) are maintained externally.

## Tracked Artifacts

| file | format | source estimator | dataset hint | notes |
|---|---|---|---|---|
| `outputs/traj_estimate_v1_01.txt` | `tum` | `sqrtVINS` (external workspace) | EuRoC `V1_01_easy` | Used by README real-mode examples |

## Update Policy

- Keep estimator source code outside this repository.
- Store only trajectory outputs needed for reproducible evaluation.
- When replacing an artifact, update this table in the same commit.
