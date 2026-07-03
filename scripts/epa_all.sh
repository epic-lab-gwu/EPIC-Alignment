#!/usr/bin/env bash
set -euo pipefail

if command -v epa_all >/dev/null 2>&1; then
  exec epa_all "$@"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m epa.case_toolchain "$@"
