#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python_bin="${PYTHON:-python}"
"$python_bin" tools/data/prepare_positives.py "$@"
