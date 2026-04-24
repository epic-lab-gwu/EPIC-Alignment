#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Run EPA full toolchain in one command.

Required:
  --gt PATH                 Ground-truth trajectory file
  --est PATH                Estimated trajectory file

Optional:
  --format FMT              tum|kitti|euroc (default: tum)
  --align-mode MODE         none|se3|sim3 (default: se3)
  --t-max-diff SEC          Time match threshold (default: 0.02)
  --out-root DIR            Output root (default: ./outputs/epa_all/run_YYYYMMDD_HHMMSS)
  --python-bin PATH         Python used for module fallback and benchmark (default: python)
  --evo-repo DIR            Local evo repo (default: /home/yifu/evo)
  --openvins-case DIR       Optional OpenVINS case dir; repeatable
  --alignanything-root DIR  Optional AlignAnything root; enables epa_benchmark
  --no-plot                 Disable plotting for all runnable tools
  -h, --help                Show this help

Examples:
  bash scripts/epa_all.sh \
    --gt ./ov_eval/example/stamped_groundtruth.txt \
    --est ./ov_eval/example/stamped_traj_estimate.txt \
    --format tum

  bash scripts/epa_all.sh \
    --gt ./ov_eval/example/stamped_groundtruth.txt \
    --est ./ov_eval/example/stamped_traj_estimate.txt \
    --format tum \
    --openvins-case ./ov_eval/example \
    --align-mode se3
EOF
}

abs_path() {
  local p="$1"
  "$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$p"
}

run_tool() {
  local bin="$1"
  local module="$2"
  shift 2
  if command -v "$bin" >/dev/null 2>&1; then
    echo "+ $bin $*"
    "$bin" "$@"
  else
    export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    echo "+ $PYTHON_BIN -m $module $*"
    "$PYTHON_BIN" -m "$module" "$@"
  fi
}

GT=""
EST=""
FORMAT="tum"
ALIGN_MODE="se3"
T_MAX_DIFF="0.02"
OUT_ROOT=""
PYTHON_BIN="python"
EVO_REPO="/home/yifu/evo"
ALIGNANYTHING_ROOT=""
NO_PLOT=0
OPENVINS_CASES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gt)
      GT="${2:-}"
      shift 2
      ;;
    --est)
      EST="${2:-}"
      shift 2
      ;;
    --format)
      FORMAT="${2:-}"
      shift 2
      ;;
    --align-mode)
      ALIGN_MODE="${2:-}"
      shift 2
      ;;
    --t-max-diff)
      T_MAX_DIFF="${2:-}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --evo-repo)
      EVO_REPO="${2:-}"
      shift 2
      ;;
    --openvins-case)
      OPENVINS_CASES+=("${2:-}")
      shift 2
      ;;
    --alignanything-root)
      ALIGNANYTHING_ROOT="${2:-}"
      shift 2
      ;;
    --no-plot)
      NO_PLOT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$GT" || -z "$EST" ]]; then
  echo "--gt and --est are required." >&2
  usage
  exit 2
fi

case "$FORMAT" in
  tum|kitti|euroc)
    ;;
  *)
    echo "--format must be one of: tum|kitti|euroc (got: $FORMAT)" >&2
    exit 2
    ;;
esac

GT_ABS="$(abs_path "$GT")"
EST_ABS="$(abs_path "$EST")"
if [[ ! -f "$GT_ABS" ]]; then
  echo "GT file not found: $GT_ABS" >&2
  exit 2
fi
if [[ ! -f "$EST_ABS" ]]; then
  echo "EST file not found: $EST_ABS" >&2
  exit 2
fi

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="$(pwd)/outputs/epa_all/run_$(date +%Y%m%d_%H%M%S)"
fi
OUT_ROOT="$(abs_path "$OUT_ROOT")"
mkdir -p "$OUT_ROOT"

if [[ "$NO_PLOT" -eq 1 ]]; then
  MAIN_PLOT_FLAG="--no-plot"
  METRIC_PLOT_FLAG=""
  TRAJ_PLOT_FLAG=""
  OPENVINS_PLOT_FLAG="--no-plot"
else
  MAIN_PLOT_FLAG="--plot"
  METRIC_PLOT_FLAG="--plot"
  TRAJ_PLOT_FLAG="--plot"
  OPENVINS_PLOT_FLAG=""
fi

echo "== EPA ALL TOOLCHAIN =="
echo "GT:             $GT_ABS"
echo "EST:            $EST_ABS"
echo "FORMAT:         $FORMAT"
echo "OUT_ROOT:       $OUT_ROOT"
echo "ALIGN_MODE:     $ALIGN_MODE"
echo "T_MAX_DIFF:     $T_MAX_DIFF"
echo "PYTHON_BIN:     $PYTHON_BIN"
echo

echo "[1/6] epa main pipeline"
mkdir -p "$OUT_ROOT/main_workspace"
(
  cd "$OUT_ROOT/main_workspace"
  run_tool epa epa.cli \
    --gt-csv "$GT_ABS" --gt-format "$FORMAT" \
    --est-path "$EST_ABS" --est-format "$FORMAT" \
    --t-max-diff "$T_MAX_DIFF" \
    "$MAIN_PLOT_FLAG"
)

echo "[2/6] epa_ape"
mkdir -p "$OUT_ROOT/ape"
run_tool epa_ape epa.ape_tool \
  "$FORMAT" "$GT_ABS" "$EST_ABS" \
  --align --t_max_diff "$T_MAX_DIFF" \
  ${METRIC_PLOT_FLAG:+$METRIC_PLOT_FLAG} \
  --out_dir "$OUT_ROOT/ape"

echo "[3/6] epa_rpe"
mkdir -p "$OUT_ROOT/rpe"
run_tool epa_rpe epa.rpe_tool \
  "$FORMAT" "$GT_ABS" "$EST_ABS" \
  --align --delta 1 --delta_unit f --t_max_diff "$T_MAX_DIFF" \
  ${METRIC_PLOT_FLAG:+$METRIC_PLOT_FLAG} \
  --out_dir "$OUT_ROOT/rpe"

echo "[4/6] epa_traj"
mkdir -p "$OUT_ROOT/traj"
if [[ "$FORMAT" == "kitti" ]]; then
  run_tool epa_traj epa.traj_tool \
    --format "$FORMAT" \
    --align --ref 1 \
    ${TRAJ_PLOT_FLAG:+$TRAJ_PLOT_FLAG} \
    --out-dir "$OUT_ROOT/traj" \
    "$GT_ABS" "$EST_ABS"
else
  run_tool epa_traj epa.traj_tool \
    --format "$FORMAT" \
    --sync --sync-max-diff "$T_MAX_DIFF" \
    --align --ref 1 \
    ${TRAJ_PLOT_FLAG:+$TRAJ_PLOT_FLAG} \
    --out-dir "$OUT_ROOT/traj" \
    "$GT_ABS" "$EST_ABS"
fi

echo "[5/6] epa_openvins (optional)"
if [[ "${#OPENVINS_CASES[@]}" -gt 0 ]]; then
  mkdir -p "$OUT_ROOT/openvins"
  run_tool epa_openvins epa.openvins_runner \
    "${OPENVINS_CASES[@]}" \
    --align-mode "$ALIGN_MODE" \
    ${OPENVINS_PLOT_FLAG:+$OPENVINS_PLOT_FLAG} \
    --keep-output \
    --save-root "$OUT_ROOT/openvins"
else
  echo "skip (no --openvins-case provided)"
fi

echo "[6/6] epa_benchmark (optional)"
if [[ -n "$ALIGNANYTHING_ROOT" ]]; then
  mkdir -p "$OUT_ROOT/benchmark"
  run_tool epa_benchmark epa.benchmark.alignanything_harness \
    --alignanything-root "$ALIGNANYTHING_ROOT" \
    --output-root "$OUT_ROOT/benchmark" \
    --repo-root "$REPO_ROOT" \
    --python-bin "$PYTHON_BIN" \
    --epa-src "$REPO_ROOT/src" \
    --evo-repo "$EVO_REPO" \
    --t-max-diff "$T_MAX_DIFF"
else
  echo "skip (no --alignanything-root provided)"
fi

echo
echo "Done. Outputs saved under: $OUT_ROOT"
