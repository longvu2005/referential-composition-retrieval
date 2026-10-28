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
    "$python_bin" tools/run.py prepare --config configs/proposed.yaml
    "$python_bin" tools/run.py build-cache --config configs/proposed.yaml
    "$python_bin" tools/run.py train --config configs/proposed.yaml
    "$python_bin" tools/run.py run --config runs/proposed/config.yaml --split val
    "$python_bin" tools/run.py calibrate --config configs/calibration.yaml
else
    "$python_bin" tools/run.py run --config runs/proposed-calibration/selected.yaml --split test
fi
