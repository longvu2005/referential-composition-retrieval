#!/usr/bin/env bash
# Build/reuse all three caches, select best.pt on val, freeze config for test.
set -euo pipefail

cd "$(dirname "$0")/../.."
split="${1:-val}"
if [[ "$#" -gt 1 || ( "$split" != val && "$split" != test ) ]]; then
    echo "Usage: $0 [val|test]" >&2
    exit 1
fi
python_bin="${PYTHON:-python}"
if [[ "$split" == val ]]; then
    "$python_bin" tools/run.py run --config configs/proposed.yaml \
        --prepare --build-cache --train --splits val
else
    "$python_bin" tools/run.py run --config runs/proposed-v2/run_config.yaml --splits test
fi
