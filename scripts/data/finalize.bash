#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Canonicalize and validate saved Full Positive decisions before building the
# final export.  Stop the positive-label UI before running this launcher: this
# step may rewrite positive_sets.jsonl in candidate-catalog order.
python_bin="${PYTHON:-.venv-dataset/bin/python}"
"$python_bin" tools/data/normalize_positives.py --write
"$python_bin" tools/data/build_final.py "$@"
