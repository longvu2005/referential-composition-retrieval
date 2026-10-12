#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python_bin="${PYTHON:-.venv-dataset/bin/python}"
"$python_bin" tools/data/prepare_positives.py "$@"
