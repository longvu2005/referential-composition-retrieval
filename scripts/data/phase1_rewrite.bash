#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ "${1:-}" != "prepare" || "$#" != 1 ]]; then
    echo "Usage: $0 prepare" >&2
    exit 1
fi

python scripts/data/select_samples.py
python scripts/data/prepare_rewrite.py
python scripts/data/prepare_handoffs.py review
