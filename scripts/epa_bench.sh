#!/usr/bin/env bash
set -euo pipefail

if command -v epa_bench >/dev/null 2>&1; then
  exec epa_bench "$@"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m epa.benchmark.benchmark_harness "$@"
