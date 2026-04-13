# CLI Reference

## Core Entry Points

- `vicon_ws`: main 3-step pipeline wrapper
- `vicon_ws_traj`: trajectory utility
- `vicon_ws_ape`: absolute pose error tool
- `vicon_ws_rpe`: relative pose error tool
- `vicon_ws_res`: compare result bundles / metrics files
- `vicon_ws_config`: global config manager

## Help Commands

```bash
vicon_ws --help
vicon_ws_traj --help
vicon_ws_ape --help
vicon_ws_rpe --help
vicon_ws_res --help
vicon_ws_config --help
```

## Common Formats

Supported trajectory formats:

- `auto`
- `csv` / `euroc`
- `tum`
- `kitti`
- `bag` / `bag2` / `mcap`

Bag examples:

```bash
vicon_ws \
  --gt-csv /path/to/run.bag --gt-format bag --gt-topic /vicon/pose \
  --est-path /path/to/run.bag --est-format bag --est-topic /odom
```

TF topic syntax:

```bash
--est-topic /tf:map.base_link
```

## Config Priority

All core tools support `--config <file.json>`.

Priority order (high -> low):

1. config file values
2. CLI values
3. global config from `vicon_ws_config`
