"""Exercise actual venv path precedence without downloading GPU packages."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import packaging
import pytest

from tools.setup_runtime import ROOT


def package(site, name, version):
    module = site / name
    module.mkdir(exist_ok=True)
    (module / "__init__.py").write_text(f"__version__ = {version!r}\n")
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )


@pytest.fixture
def environments(tmp_path):
    paths = []
    for name in ("donor", "target"):
        root = tmp_path / name
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", root], check=True
        )
        python = root / "bin/python"
        site = Path(
            subprocess.check_output(
                [
                    python,
                    "-c",
                    "import sysconfig; print(sysconfig.get_paths()['purelib'])",
                ],
                text=True,
            ).strip()
        )
        paths.append((python, site))
    (donor, donor_site), (target, target_site) = paths
    package(donor_site, "torch", "2.5.1+cpu")
    package(donor_site, "torchvision", "0.20.1+cpu")
    package(donor_site, "transformers", "4.56.0")
    package(target_site, "transformers", "4.39.3")
    shutil.copytree(Path(packaging.__file__).parent, target_site / "packaging")
    env = dict(os.environ, PYTHONNOUSERSITE="1", RCR_REUSE_PYTHON=str(donor))
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("RCR_REUSE_PACKAGES", None)
    return donor, donor_site, target, target_site, env


def setup(target, env):
    return subprocess.run(
        [target, ROOT / "tools/setup_runtime.py", "fafa", sys.executable],
        env=env,
        text=True,
        capture_output=True,
    )


def test_shared_torch_keeps_transformers_local_and_is_idempotent(environments):
    donor, donor_site, target, target_site, env = environments
    before = {
        p.relative_to(donor_site): p.read_bytes()
        for p in donor_site.rglob("*")
        if p.is_file()
    }
    first = setup(target, env)
    assert first.returncode == 0, first.stderr
    assert "Reusing compatible Torch" in first.stdout
    result = subprocess.check_output(
        [
            target,
            "-c",
            "import json, torch, transformers; "
            "print(json.dumps([torch.__file__, transformers.__version__]))",
        ],
        env=env,
        text=True,
    )
    torch_path, transformers = json.loads(result)
    assert Path(torch_path).is_relative_to(donor_site)
    assert transformers == "4.39.3"
    pth = target_site / "rcr_shared_runtime.pth"
    saved = pth.read_bytes()
    second = setup(target, env)
    assert second.returncode == 0, second.stderr
    assert pth.read_bytes() == saved
    assert all(
        (donor_site / p).read_bytes() == content for p, content in before.items()
    )


def test_incompatible_pinned_torch_is_never_reused(environments):
    _, donor_site, target, target_site, env = environments
    shutil.rmtree(donor_site / "torch-2.5.1+cpu.dist-info")
    package(donor_site, "torch", "2.6.0")
    result = setup(target, env)
    assert result.returncode != 0
    assert "Python/Torch/Torchvision requirements must match" in result.stderr
    assert not (target_site / "rcr_shared_runtime.pth").exists()


def test_disabling_sharing_removes_inherited_paths(environments):
    _, _, target, target_site, env = environments
    assert setup(target, env).returncode == 0
    env.update(RCR_REUSE_PACKAGES="off", RCR_INSTALL_RESERVE_GIB="0")
    result = setup(target, env)
    assert result.returncode == 0, result.stderr
    assert not (target_site / "rcr_shared_runtime.pth").exists()
    assert "installing locally" in result.stdout


def test_donor_cannot_depend_back_on_target(environments):
    _, donor_site, target, target_site, env = environments
    shutil.rmtree(donor_site / "torch")
    shutil.rmtree(donor_site / "torch-2.5.1+cpu.dist-info")
    package(target_site, "torch", "2.5.1+cpu")
    (donor_site / "target.pth").write_text(str(target_site) + "\n")
    result = setup(target, env)
    assert result.returncode != 0
    assert "cannot depend on this environment" in result.stderr
    assert not (target_site / "rcr_shared_runtime.pth").exists()
