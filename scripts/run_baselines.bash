#!/usr/bin/env bash
# Run from any directory. Forward CLI arguments to the selected environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

method="${1:-clip}"
if (( $# )); then shift; fi
case "$method" in
  clip|fafa) ;;
  *) echo "Usage: bash scripts/run_baselines.bash [clip|fafa] [--prepare] [runner args]" >&2; exit 2 ;;
esac
python="${BASELINE_PYTHON:-.venv-$method/bin/python}"
config="configs/methods/baselines/$method.yaml"
args=()
prepare=false
while (( $# )); do
  case "$1" in
    --prepare) prepare=true; shift ;;
    --config)
      if (( $# < 2 )); then echo "--config needs a path" >&2; exit 2; fi
      config="$2"; shift 2 ;;
    --config=*) config="${1#*=}"; shift ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ "$prepare" == true ]]; then
  "$python" tools/methods/prepare_baseline.py --config "$config"
fi
"$python" tools/methods/run_baselines.py --config "$config" "${args[@]}"
