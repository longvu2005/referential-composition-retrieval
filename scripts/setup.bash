#!/usr/bin/env bash
# Install exactly one method/tool environment from its requirements file.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "$#" != 1 ]]; then
    echo "Usage: $0 {proposed|clip|fafa|dataset|dataset-clip|evaluation|dev}" >&2
    exit 1
fi
case "$1" in
    proposed|clip|fafa|dataset|dataset-clip|evaluation|dev) ;;
    *) echo "Unknown environment: $1" >&2; exit 1 ;;
esac

env_dir=".venv-$1"
python_bin="${PYTHON:-python3}"
# --without-pip also works on notebook images without ensurepip. Bootstrap
# through the host pip, then use the environment's pip for the actual install.
"$python_bin" -m venv --without-pip "$env_dir"
"$python_bin" -m pip --python "$env_dir/bin/python" install -r requirements/bootstrap.txt
"$env_dir/bin/python" -m pip install --no-build-isolation -r "requirements/$1.txt"
echo "Ready: $env_dir/bin/python"
