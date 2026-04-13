# Benchmark

## AlignAnything Harness

`vicon_ws_benchmark` runs an independent comparison between `vicon_ws` and `evo`.

Key property:

- offsets are not shared
- `vicon_ws` uses internal step-1 estimate
- `evo` uses independent sweep inside harness

Run example:

```bash
vicon_ws_benchmark \
  --alignanything-root /home/yifu/vicon_ws/AlignAnything/AlignAnything \
  --repo-root /home/yifu/vicon_ws \
  --evo-repo /home/yifu/evo
```

Typical outputs:

- `outputs/alignanything_harness/run_xxx/summary.csv`
- `outputs/alignanything_harness/run_xxx/summary.md`
- `outputs/alignanything_harness/run_xxx/cases/*.json`

## Plot Harness Summary

```bash
vicon_ws_plot_summary --summary-csv outputs/alignanything_harness/run_xxx/summary.csv
```

## Metric Result Aggregation

```bash
vicon_ws_metric_res --metrics-json /path/to/metrics_a.json /path/to/metrics_b.json --mode aggregate
```
