#!/usr/bin/env bash
# Reuse the FAFA environment; both poolings are fixed before test.
set -euo pipefail

cd "$(dirname "$0")/../.."
split="${1:-val}"
if [[ "$#" -gt 1 || ( "$split" != val && "$split" != test ) ]]; then
    echo "Usage: $0 [val|test]" >&2
    exit 1
fi
python_bin="${PYTHON:-.venv-fafa/bin/python}"
if [[ "$split" == val ]]; then
    "$python_bin" tools/run.py prepare --config configs/fafa_pooling.yaml
fi
"$python_bin" tools/run.py run --config configs/fafa_pooling.yaml --split "$split"
