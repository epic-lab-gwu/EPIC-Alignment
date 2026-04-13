from __future__ import annotations

import argparse
import json
from pathlib import Path

from vicon_ws.config_cli import parse_args_with_config
from vicon_ws.core.evaluation import normalize_pose_relation
from vicon_ws.viz.metric_plots import aggregate_metric_results, generate_metric_plots


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Metric plotting and result aggregation for vicon_ws metrics.json files."
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to JSON config file. If set, config values override CLI flags.",
    )
    parser.add_argument(
        "--metrics-json",
        nargs="+",
        required=True,
        help="One or multiple metrics.json paths.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "single", "aggregate"],
        default="auto",
        help="single: per-run stage comparison; aggregate: cross-run comparison; auto: infer from file count.",
    )
    parser.add_argument(
        "--metric",
        choices=["all", "ape", "rpe"],
        default="all",
        help="Metric family to aggregate in aggregate mode.",
    )
    parser.add_argument(
        "--ape-relation",
        default="trans_part",
        help="APE pose relation (alias or internal name).",
    )
    parser.add_argument(
        "--rpe-relation",
        default="trans_part",
        help="RPE pose relation (alias or internal name).",
    )
    parser.add_argument(
        "--stage",
        choices=["raw", "step2", "step3"],
        default="step3",
        help="Stage used in aggregate mode.",
    )
    parser.add_argument(
        "--x-dimension",
        choices=["index", "seconds", "distances"],
        default="seconds",
        help="X-axis used for raw plots in single mode.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to a sibling folder of the first metrics.json.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    paths = [Path(p).expanduser().resolve() for p in args.metrics_json]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"metrics.json not found: {p}")

    mode = args.mode
    if mode == "auto":
        mode = "single" if len(paths) == 1 else "aggregate"

    ape_rel = normalize_pose_relation("ape", args.ape_relation)
    rpe_rel = normalize_pose_relation("rpe", args.rpe_relation)

    if args.out_dir:
        out_root = Path(args.out_dir).expanduser().resolve()
    else:
        suffix = "metric_plots" if mode == "single" else "metric_aggregate"
        out_root = paths[0].parent / suffix

    payloads = [_load_payload(p) for p in paths]

    produced: list[Path] = []
    if mode == "single":
        produced = generate_metric_plots(
            metrics_payload=payloads[0],
            out_dir=out_root,
            ape_relation=ape_rel,
            rpe_relation=rpe_rel,
            x_dimension=args.x_dimension,
        )
    else:
        labels = [p.parent.name for p in paths]
        selected_metrics = ["ape", "rpe"] if args.metric == "all" else [args.metric]
        for metric in selected_metrics:
            relation = ape_rel if metric == "ape" else rpe_rel
            metric_out = out_root / f"{metric}_{relation}_{args.stage}"
            produced.extend(
                aggregate_metric_results(
                    payloads=payloads,
                    labels=labels,
                    out_dir=metric_out,
                    metric_kind=metric,
                    relation=relation,
                    stage=args.stage,
                )
            )

    print(f"Mode: {mode}")
    print(f"Output dir: {out_root}")
    print(f"Generated {len(produced)} file(s).")
    return 0


def main() -> int:
    parser = build_parser()
    args = parse_args_with_config(parser, config_dest="config")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
