#!/usr/bin/env bash
# Match the baseline wrapper's environment selection and argument forwarding.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python="${PROPOSED_PYTHON:-.venv-proposed/bin/python}"
"$python" tools/methods/run_proposed.py "$@"
