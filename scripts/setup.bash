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
# Check before changing pip/packages or resolving unsupported source builds.
"$python_bin" - <<'PY'
import sys

if not (3, 11) <= sys.version_info[:2] < (3, 14):
    raise SystemExit(
        f"Unsupported Python {sys.version.split()[0]}; use Python 3.11, 3.12 or 3.13 "
        "and set PYTHON to that notebook interpreter."
    )
print(f"Installing into Python {sys.version.split()[0]}: {sys.executable}")
PY
# Notebook disks need room for features, not another copy of large Torch wheels.
"$python_bin" -m pip install --no-cache-dir -r requirements/bootstrap.txt
"$python_bin" -m pip install --no-cache-dir --no-build-isolation --only-binary=numpy \
    -r "requirements/$1.txt"
"$python_bin" -c 'import rcr; print("Installed rcr:", rcr.__file__)'
echo "Ready: $python_bin ($1)"
