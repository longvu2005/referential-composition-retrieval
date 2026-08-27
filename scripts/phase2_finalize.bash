#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python tools/dataset/build_final.py "$@"
