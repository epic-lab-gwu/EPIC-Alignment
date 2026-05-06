#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Install ov_eval-compatible command shims that forward to EPA.

Usage:
  bash scripts/openvins/install_ov_eval_epa_shims.sh [target_dir]

Arguments:
  target_dir  Directory where shim commands will be created.
              Default: ./tools/epa_ov_eval_shims

After install:
  export PATH="<target_dir>:$PATH"
  conda activate epa

Supported forms:
  error_comparison se3 <gt_dir> <alg_dir>
  error_comparison_se3 <gt_dir> <alg_dir>
EOF
  exit 0
fi

target_dir="${1:-$(pwd)/tools/epa_ov_eval_shims}"
mkdir -p "$target_dir"

dispatch="$target_dir/ov_eval_epa_dispatch"
cat >"$dispatch" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

prog="$(basename "$0")"
python_bin="${EPA_PYTHON:-python}"

case "$prog" in
  format_converter|error_singlerun|error_dataset|error_comparison|plot_trajectories)
    exec "$python_bin" -m epa.ov_eval_compat "$prog" "$@"
    ;;
  error_singlerun_*|error_dataset_*|error_comparison_*|plot_trajectories_*)
    cmd="${prog%_*}"
    mode="${prog##*_}"
    exec "$python_bin" -m epa.ov_eval_compat "$cmd" "$mode" "$@"
    ;;
  epa_ov_eval_shim)
    if [[ $# -lt 1 ]]; then
      echo "Usage: epa_ov_eval_shim <command> [args...]" >&2
      echo "Commands: format_converter error_singlerun error_dataset error_comparison plot_trajectories" >&2
      exit 2
    fi
    cmd="$1"
    shift
    exec "$python_bin" -m epa.ov_eval_compat "$cmd" "$@"
    ;;
  *)
    echo "Unsupported shim name: $prog" >&2
    exit 2
    ;;
esac
EOF

chmod +x "$dispatch"

base_commands=(format_converter error_singlerun error_dataset error_comparison plot_trajectories epa_ov_eval_shim)
align_modes=(se3 se3single sim3 posyaw posyawsingle none)

for name in "${base_commands[@]}"; do
  ln -sfn "ov_eval_epa_dispatch" "$target_dir/$name"
done

for cmd in error_singlerun error_dataset error_comparison plot_trajectories; do
  for mode in "${align_modes[@]}"; do
    ln -sfn "ov_eval_epa_dispatch" "$target_dir/${cmd}_${mode}"
  done
done

echo "Installed EPA ov_eval shims in: $target_dir"
echo "Next:"
echo "  1) conda activate epa"
echo "  2) export PATH=\"$target_dir:\$PATH\""
echo "  3) error_singlerun se3 <gt.txt> <est.txt>"
echo "     or: error_singlerun_se3 <gt.txt> <est.txt>"
