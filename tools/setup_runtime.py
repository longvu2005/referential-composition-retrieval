"""Reuse a compatible installed Torch stack without merging method requirements.

Invoked by setup.bash after bootstrapping an isolated target environment. Plain
.pth entries append donor package directories after the target's own packages;
pip installs incompatible method dependencies locally and cannot uninstall the
donor distributions outside sys.prefix. Never link individual package files.
"""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import json, sys, sysconfig
from importlib import metadata
names = json.loads(sys.argv[1])
versions, paths = {}, set()
for name in names:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    versions[name] = dist.version
    paths.add(str(dist.locate_file('')))
for dist in metadata.distributions():
    name = dist.metadata.get('Name', '').lower().replace('_', '-')
    if name.startswith('nvidia-') or name == 'triton':
        paths.add(str(dist.locate_file('')))
print(json.dumps({'python': list(sys.version_info[:2]),
                 'abi': sysconfig.get_config_var('SOABI'),
                 'versions': versions, 'paths': sorted(paths)}))
"""


def torch_requirements(method):
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    groups = ("dataset", "clip") if method == "dataset-clip" else (method,)
    return [
        req
        for group in groups
        for text in extras[group]
        if (req := Requirement(text)).name in ("torch", "torchvision")
    ]


def probe(python, requirements):
    try:
        result = subprocess.run(
            [str(python), "-c", PROBE, json.dumps([r.name for r in requirements])],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout)
        if info["python"] != list(sys.version_info[:2]) or info[
            "abi"
        ] != sysconfig.get_config_var("SOABI"):
            return None
        if not all(
            r.name in info["versions"] and info["versions"][r.name] in r.specifier
            for r in requirements
        ):
            return None
        return info
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        return None


def main(argv=None):
    method, host_python = argv or sys.argv[1:]
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run setup_runtime.py inside the target virtual environment")
    requirements = torch_requirements(method)
    if not requirements:
        return
    pth = Path(sysconfig.get_paths()["purelib"]) / "rcr_shared_runtime.pth"
    mode = os.environ.get("RCR_REUSE_PACKAGES", "auto")
    if mode not in ("auto", "off"):
        raise ValueError("RCR_REUSE_PACKAGES must be auto or off")
    if mode == "off":
        inherited = set(pth.read_text().splitlines()) if pth.is_file() else set()
        pth.unlink(missing_ok=True)
        sys.path[:] = [path for path in sys.path if path not in inherited]
    elif probe(sys.executable, requirements):
        print("Torch stack already satisfies this environment's requirements")
        return
    else:
        explicit = os.environ.get("RCR_REUSE_PYTHON")
        candidates = (
            [explicit]
            if explicit
            else [
                ROOT / ".venv-fafa/bin/python",
                ROOT / ".venv-proposed/bin/python",
                shutil.which(host_python) or host_python,
            ]
        )
        for candidate in candidates:
            info = probe(candidate, requirements)
            if info is None:
                continue
            paths = [Path(path).resolve() for path in info["paths"]]
            # Reject a donor that resolves back into this target (a sharing cycle).
            if any(path.is_relative_to(Path(sys.prefix).resolve()) for path in paths):
                continue
            if not paths or any("\n" in str(path) for path in paths):
                continue
            try:
                subprocess.run(
                    [
                        str(candidate),
                        "-c",
                        "; ".join(f"import {r.name}" for r in requirements),
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                continue
            pth.write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
            print(
                f"Reusing compatible Torch stack from {candidate}: {info['versions']}"
            )
            print("Keep that source environment; method-specific packages stay local.")
            return
        if explicit:
            raise RuntimeError(
                f"Cannot reuse {explicit}: Python/Torch/Torchvision requirements "
                "must match and the donor cannot depend on this environment"
            )
    # An incompatible/missing Torch needs another large wheel + installed CUDA
    # stack. This estimate is configurable; pip is still the dependency resolver.
    torch_req = next(r for r in requirements if r.name == "torch")
    try:
        existing = metadata.version("torch") in torch_req.specifier
    except metadata.PackageNotFoundError:
        existing = False
    if not existing:
        reserve = float(os.environ.get("RCR_INSTALL_RESERVE_GIB", "8"))
        if not 0 <= reserve < float("inf"):
            raise ValueError("RCR_INSTALL_RESERVE_GIB must be finite and nonnegative")
        free = shutil.disk_usage(sys.prefix).free / 1024**3
        if free < reserve:
            raise OSError(
                f"Fresh Torch/CUDA installation needs about {reserve:.2f} GiB free; "
                f"found {free:.2f} GiB at {sys.prefix}. Reuse a compatible environment "
                "with RCR_REUSE_PYTHON or free disk space before setup. "
                "Checkpoint/runtime assets need additional space."
            )
        print(
            "No compatible shared Torch stack; "
            f"installing locally ({free:.2f} GiB free)"
        )


if __name__ == "__main__":
    main()
