#!/usr/bin/env bash
# Build/reuse caches, train on train, select coefficients on val, freeze for test.
set -euo pipefail

cd "$(dirname "$0")/../.."
split="${1:-val}"
if [[ "$#" -gt 1 || ( "$split" != val && "$split" != test ) ]]; then
    echo "Usage: $0 [val|test]" >&2
    exit 1
fi
python_bin="${PYTHON:-.venv-proposed/bin/python}"
if [[ "$split" == val ]]; then
    "$python_bin" tools/run.py run --config configs/proposed.yaml \
        --prepare --build-cache --train --splits val
    "$python_bin" tools/run.py ablate --config configs/calibration.yaml --splits val
    # Publish the chosen model's validation result under the same root as test.
    "$python_bin" tools/run.py run --config runs/calibration/selected.yaml --splits val
else
    "$python_bin" tools/run.py run --config runs/calibration/selected.yaml --splits test
fi
