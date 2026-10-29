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
# Notebook startup hooks must not leak into the method environment or pip's
# build subprocesses. In particular, Kaggle's hooks can require host-only wrapt.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

python_version() {
    "$1" -I -c 'import sys; print("%s.%s" % sys.version_info[:2])'
}

# Honor an explicit interpreter; otherwise prefer the supported minor versions
# over a notebook's python3, which may already be Python 3.13 or newer.
python_bin=""
candidates=(python3.11 python3.12 python3)
if [[ -n "${PYTHON:-}" ]]; then
    candidates=("$PYTHON")
fi
for candidate in "${candidates[@]}"; do
    if version="$(python_version "$candidate" 2>/dev/null)"; then
        case "$version" in
            3.11|3.12) python_bin="$candidate"; break ;;
        esac
    fi
done
if [[ -z "$python_bin" ]]; then
    echo "Python 3.11 or 3.12 is required (PYTHON=${PYTHON:-auto})." >&2
    echo "Install one, then run: PYTHON=python3.11 bash scripts/setup.bash $1" >&2
    exit 1
fi

# Do not overwrite an incompatible or broken environment in place. A failed
# install may still leave bin/python behind; rerun setup to finish installing.
if [[ -e "$env_dir" || -L "$env_dir" ]]; then
    env_version="$(python_version "$env_dir/bin/python" 2>/dev/null || true)"
    if [[ "$env_version" != "$version" ]]; then
        echo "$env_dir uses Python ${env_version:-unknown}; selected Python $version." >&2
        echo "Move $env_dir aside, then rerun this setup command." >&2
        exit 1
    fi
else
    "$python_bin" -I -m venv --without-pip "$env_dir"
fi

# --without-pip also works on notebook images without ensurepip. Bootstrap
# through the host pip, then use the environment's pip for the actual install.
# CUDA wheels can be several GiB. Keep only the installed packages on notebook
# disks, rather than retaining another compressed copy in pip's download cache.
"$python_bin" -I -m pip --python "$env_dir/bin/python" install --no-cache-dir -r requirements/bootstrap.txt
"$env_dir/bin/python" -I -m pip install --no-cache-dir --no-build-isolation -r "requirements/$1.txt"
echo "Ready: $env_dir/bin/python"
