#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$(mktemp -d /tmp/epa_mpl_XXXXXX)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
trap 'rm -rf "${MPLCONFIGDIR}"' EXIT

"${PYTHON_BIN}" -m epa.cli --engine modular --help >/dev/null
"${PYTHON_BIN}" -m epa.cli "${ROOT_DIR}/example_data/example_groundtruth.csv" "${ROOT_DIR}/example_data/example_estimation.txt" --dry-run >/dev/null
"${PYTHON_BIN}" -m epa.cli --gt "${ROOT_DIR}/example_data/example_groundtruth.csv" --est "${ROOT_DIR}/example_data/example_estimation.txt" --dry-run >/dev/null
"${PYTHON_BIN}" -m epa.cli --gt-csv "${ROOT_DIR}/example_data/example_groundtruth.csv" --est-path "${ROOT_DIR}/example_data/example_estimation.txt" --dry-run >/dev/null

echo "smoke ok: modular engine"
