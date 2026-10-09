#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ "$#" != 0 ]]; then
    echo "Usage: $0 (set PYTHON to the notebook kernel's Python)" >&2
    exit 1
fi

python_bin="${PYTHON:-python}"
"$python_bin" tools/data/select_samples.py
"$python_bin" tools/data/prepare_review.py
