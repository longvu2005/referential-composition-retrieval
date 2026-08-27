#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

action="${1:-}"
shift || true

case "$action" in
    prepare)
        python tools/dataset/audit_dataset.py
        python tools/dataset/select_samples.py
        python tools/dataset/prepare_rewrite.py
        python tools/dataset/rewrite_gemini.py split "$@"
        ;;
    submit|status|collect)
        python tools/dataset/rewrite_gemini.py "$action" "$@"
        ;;
    merge)
        python tools/dataset/rewrite_gemini.py merge "$@"
        python tools/dataset/prepare_handoffs.py review
        ;;
    *)
        echo "Usage: $0 {prepare|submit|status|collect|merge} [options]" >&2
        exit 1
        ;;
esac
