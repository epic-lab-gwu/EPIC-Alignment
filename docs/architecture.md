# vicon_ws Architecture

## Round 1

- Added wrapper CLI and legacy bridge:
  - `vicon_ws.cli -> vicon_ws.runner -> bridge_legacy -> pipeline.py`
- Goal: preserve existing behavior while introducing project skeleton.

## Round 2

- Added modular execution engine:
  - `vicon_ws.cli --engine modular`
  - `vicon_ws.runner -> core.pipeline_modular`
- Core algorithm code is reorganized by responsibility:
  - `core/time_alignment.py`
  - `core/calibration.py`
  - `core/world` functionality merged in `core/calibration.py`
  - `core/evaluation.py`
  - `core/io_utils.py`
  - `core/math_utils.py`

## Compatibility

- Legacy path is still available:
  - `--engine legacy` calls original `pipeline.py`.
- Original `pipeline.py` file and methods are preserved.

## Next Steps

1. Add unit tests for each `core/*` module.
2. Make `pipeline.py` a thin shim that calls modular core.
3. Split plotting and metrics serialization into dedicated components.

