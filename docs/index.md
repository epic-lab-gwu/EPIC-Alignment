# vicon_ws Documentation

`vicon_ws` is a trajectory alignment and evaluation toolkit with:

- 3-step alignment pipeline (time alignment -> extrinsic solve -> world alignment)
- evo-style CLI tools (`vicon_ws_traj`, `vicon_ws_ape`, `vicon_ws_rpe`, `vicon_ws_res`)
- optional rerun visualization
- AlignAnything benchmark harness

## Start Here

- local docs first: `mkdocs serve`
- local docs check: `mkdocs build --strict`
- see [Local Docs](local_docs.md) for full setup

## Who This Is For

- team members who need to run alignment and evaluation quickly
- contributors extending formats, metrics, or visualization
- benchmark users comparing `vicon_ws` and `evo`

## Documentation Map

- `Quick Start`: local install and first successful run
- `CLI Reference`: commands and common options
- `Benchmark`: how to run harness and summary plotting
- `Architecture`: module layout and execution flow
- `Estimation Artifacts`: tracked trajectory artifacts policy

## Conventions

- input/output paths in examples are repo-relative unless stated
- generated outputs are under `outputs/`
- `pipeline.py` is a compatibility shim; core logic is under `src/vicon_ws/core`
