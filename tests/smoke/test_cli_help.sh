#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$(mktemp -d /tmp/epa_mpl_XXXXXX)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
trap 'rm -rf "${MPLCONFIGDIR}"' EXIT

"${PYTHON_BIN}" -m epa.cli --help >/dev/null
echo "smoke ok: epa.cli --help"
"${PYTHON_BIN}" -m epa.ov_eval_compat --help >/dev/null
echo "smoke ok: epa.ov_eval_compat --help"
"${PYTHON_BIN}" -m epa.ov_eval_compat error_comparison --help | grep -q "folder_groundtruth"
echo "smoke ok: epa.ov_eval_compat error_comparison --help"
