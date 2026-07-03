#!/usr/bin/env bash
set -euo pipefail

# Repo-local wrapper for the pip-installable command.
if command -v epa_openvins >/dev/null 2>&1; then
  exec epa_openvins "$@"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m epa.openvins_runner "$@"
