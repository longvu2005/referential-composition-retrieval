#!/usr/bin/env bash
# Install one method/tool profile into the notebook's current Python.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "$#" != 1 ]]; then
    echo "Usage: $0 {proposed|clip|fafa|dataset|dataset-clip|evaluation|dev}" >&2
    exit 1
fi
case "$1" in
    proposed|clip|fafa|dataset|dataset-clip|evaluation|dev) ;;
    *) echo "Unknown profile: $1" >&2; exit 1 ;;
esac

python_bin="${PYTHON:-python}"
"$python_bin" -m pip install -r requirements/bootstrap.txt
"$python_bin" -m pip install --no-build-isolation -r "requirements/$1.txt"
echo "Ready: $python_bin ($1)"
