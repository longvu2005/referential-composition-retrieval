#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" != "prepare" || "$#" != 1 ]]; then
    echo "Usage: $0 prepare" >&2
    exit 1
fi

python tools/dataset/select_samples.py
python tools/dataset/prepare_rewrite.py
python tools/dataset/prepare_handoffs.py review
