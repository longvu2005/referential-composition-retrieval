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
    run_dir="$("$python_bin" -c '
from rcr.common.config import load_config
print(load_config("configs/proposed.yaml")["output"]["dir"])
')"
    "$python_bin" tools/run.py prepare --config configs/proposed.yaml
    "$python_bin" tools/run.py build-cache --config configs/proposed.yaml
    "$python_bin" tools/run.py train --config configs/proposed.yaml
    "$python_bin" tools/run.py run --config "$run_dir/config.yaml" --split val
    "$python_bin" tools/run.py calibrate --config configs/calibration.yaml \
        --set "rankings=$run_dir/val/rankings.pt"
else
    calibration_dir="$("$python_bin" -c '
from rcr.common.config import load_config
print(load_config("configs/calibration.yaml")["output_dir"])
')"
    "$python_bin" tools/run.py run --config "$calibration_dir/selected.yaml" --split test
fi
