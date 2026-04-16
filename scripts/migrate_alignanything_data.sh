#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/AlignAnything"
DATA_ROOT="${EPA_DATA_ROOT:-$HOME/epa_data}"
DST_DIR="${DATA_ROOT}/AlignAnything"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "No local AlignAnything directory at: ${SRC_DIR}"
  echo "Nothing to migrate."
  exit 0
fi

if [[ -e "${DST_DIR}" ]]; then
  echo "Target already exists: ${DST_DIR}"
  echo "Please move it away or set EPA_DATA_ROOT to another path."
  exit 1
fi

mkdir -p "${DATA_ROOT}"
mv "${SRC_DIR}" "${DST_DIR}"

echo "Moved dataset directory:" 
echo "  from: ${SRC_DIR}"
echo "  to:   ${DST_DIR}"
echo
echo "Add this to your shell profile (recommended):"
echo "  export EPA_DATA_ROOT=${DATA_ROOT}"
echo
echo "Benchmark will then default to:"
echo "  ${DST_DIR}/AlignAnything"
