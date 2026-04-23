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

for name in format_converter error_singlerun error_dataset error_comparison plot_trajectories epa_ov_eval_shim; do
  ln -sfn "ov_eval_epa_dispatch" "$target_dir/$name"
done

echo "Installed EPA ov_eval shims in: $target_dir"
echo "Next:"
echo "  1) conda activate epa"
echo "  2) export PATH=\"$target_dir:\$PATH\""
echo "  3) error_singlerun se3 <gt.txt> <est.txt>"
