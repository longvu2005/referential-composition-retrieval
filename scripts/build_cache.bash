#!/usr/bin/env bash
# Explicit cache build; training/evaluation never rebuild it implicitly.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python="${PROPOSED_PYTHON:-.venv-proposed/bin/python}"
"$python" tools/methods/build_cache.py "$@"
