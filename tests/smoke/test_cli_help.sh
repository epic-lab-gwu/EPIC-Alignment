#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$(mktemp -d /tmp/epa_mpl_XXXXXX)"
trap 'rm -rf "${MPLCONFIGDIR}"' EXIT

python3 -m epa.cli --help >/dev/null
echo "smoke ok: epa.cli --help"
