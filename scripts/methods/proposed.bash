#!/usr/bin/env bash
# Build/reuse caches and select calibrated validation checkpoints during training.
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
else
    selected_config="$("$python_bin" -c '
import sys
from pathlib import Path
sys.path.insert(0, "src")
from rcr.common.config import load_config
print(Path(load_config("configs/proposed.yaml")["output"]["dir"]) / "selected.yaml")
')"
    "$python_bin" tools/run.py run --config "$selected_config" --split test
fi
