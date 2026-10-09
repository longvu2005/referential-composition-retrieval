#!/usr/bin/env bash
# Validation freezes adapter settings; test requires the same protocol lock.
set -euo pipefail

cd "$(dirname "$0")/../.."
split="${1:-val}"
if [[ "$#" -gt 1 || ( "$split" != val && "$split" != test ) ]]; then
    echo "Usage: $0 [val|test]" >&2
    exit 1
fi
python_bin="${PYTHON:-python}"
if [[ "$split" == val ]]; then
    "$python_bin" tools/run.py prepare --config configs/fafa.yaml
fi
"$python_bin" tools/run.py run --config configs/fafa.yaml --splits "$split"
