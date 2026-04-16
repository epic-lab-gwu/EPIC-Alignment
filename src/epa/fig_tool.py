from __future__ import annotations

import argparse
from pathlib import Path

from epa.viz.plot_bundle import iter_figure_specs, load_plot_bundle, render_figure_spec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render serialized plot bundles (epa figure specs)."
    )
    p.add_argument(
        "bundles",
        nargs="+",
        help="Path(s) to serialized plot bundle JSON files.",
    )
    p.add_argument(
        "--out_dir",
        default="",
        help="Directory to write rendered plots. Defaults to <bundle>_rerender.",
    )
    p.add_argument(
        "--save_plot",
        default="",
        help="Path stem for exported plots (e.g. figs.png -> figs_raw.png, figs_map.png).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output dpi for rendered figures.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List figure names in bundle(s) without rendering.",
    )
    return p


def _resolve_save_path(
    *,
    save_plot: str,
    out_dir: str,
    bundle_path: Path,
    figure_name: str,
    multi_bundle: bool,
) -> Path:
    if str(save_plot).strip():
        base = Path(save_plot).expanduser().resolve()
        suffix = base.suffix if base.suffix else ".png"
        stem = base.stem if base.suffix else base.name
        if multi_bundle:
            tag = f"{bundle_path.stem}_{figure_name}"
        else:
            tag = figure_name
        return base.with_name(f"{stem}_{tag}{suffix}")

    if str(out_dir).strip():
        target_dir = Path(out_dir).expanduser().resolve()
    else:
        target_dir = bundle_path.parent / f"{bundle_path.stem}_rerender"
    target_dir.mkdir(parents=True, exist_ok=True)
    if multi_bundle:
        return target_dir / f"{bundle_path.stem}_{figure_name}.png"
    return target_dir / f"{figure_name}.png"


def run(args: argparse.Namespace) -> int:
    bundle_paths = [Path(p).expanduser().resolve() for p in args.bundles]
    multi_bundle = len(bundle_paths) > 1

    for bundle_path in bundle_paths:
        bundle = load_plot_bundle(bundle_path)
        figure_specs = iter_figure_specs(bundle)

        if bool(getattr(args, "list", False)):
            names = [str(spec.get("name", f"figure_{i:02d}")) for i, spec in enumerate(figure_specs)]
            print(f"{bundle_path}: {', '.join(names)}")
            continue

        for idx, spec in enumerate(figure_specs):
            figure_name = str(spec.get("name", f"figure_{idx:02d}"))
            out_path = _resolve_save_path(
                save_plot=str(getattr(args, "save_plot", "")),
                out_dir=str(getattr(args, "out_dir", "")),
                bundle_path=bundle_path,
                figure_name=figure_name,
                multi_bundle=multi_bundle,
            )
            render_figure_spec(spec, out_path=out_path, dpi=int(getattr(args, "dpi", 180)))
            print(f"Saved figure: {out_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

