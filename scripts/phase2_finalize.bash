#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Canonicalize and validate saved Full Positive decisions before building the
# final export.  Stop the positive-label UI before running this launcher: this
# step may rewrite positive_sets.jsonl in candidate-catalog order.
python tools/dataset/normalize_positives.py --write
python tools/dataset/build_final.py "$@"
