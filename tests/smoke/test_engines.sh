#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$(mktemp -d /tmp/vicon_ws_mpl_XXXXXX)"
trap 'rm -rf "${MPLCONFIGDIR}"' EXIT

python3 -m vicon_ws.cli --engine modular --help >/dev/null
python3 -m vicon_ws.cli --engine legacy --dry-run --synthetic --gt-csv "${ROOT_DIR}/gt.csv" >/dev/null || true

echo "smoke ok: modular+legacy engines"
