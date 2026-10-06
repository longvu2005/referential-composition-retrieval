"""Run frozen FAFA extraction in its own dependency environment."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


def load_fafa_config(cfg: dict) -> dict:
    path = Path(cfg["person_encoder"]["fafa_config"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def fafa_spec(cfg: dict) -> dict:
    """Feature provenance independent of baseline detector/selector settings."""
    return {
        "source_commit": cfg["source"]["commit"],
        "checkpoint_source": cfg["checkpoint"].get("source_url"),
        "model_name": cfg["checkpoint"]["model_name"],
        "model_type": cfg["checkpoint"]["model_type"],
        "image_size": cfg["checkpoint"]["image_size"],
        "test_resize_hw": cfg["checkpoint"]["test_resize_hw"],
        "feature": "mean_image_embeds_before_vision_proj",
    }


def worker_python(cfg: dict) -> str:
    executable = cfg["person_encoder"]["python"]
    path = shutil.which(executable)
    if path is None and Path(executable).is_file():
        path = str(Path(executable).resolve())
    if path is None:
        raise FileNotFoundError(
            f"FAFA Python missing: {executable}; create .venv-fafa with "
            "requirements/fafa.txt or set person_encoder.python"
        )
    return path


def check_fafa_assets(cfg: dict) -> None:
    """Validate the native environment/assets before expensive scene extraction."""
    from rcr.methods.baselines.fafa import assets_ready, official_api, official_source

    batch_size = cfg["person_encoder"]["batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("person_encoder.batch_size must be positive")
    native = load_fafa_config(cfg)
    directory = official_source(native)
    checkpoint = Path(native["checkpoint"]["path"])
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise FileNotFoundError(f"FAFA checkpoint missing: {checkpoint}; run prepare")
    if not assets_ready(native):
        raise RuntimeError("FAFA runtime assets missing/stale; run prepare")
    official_api(directory)  # Check imports without constructing/downloading weights.


def run_person_worker(
    cfg: dict, *, prepare: bool = False, check: bool = False, force: bool = False
) -> None:
    """Share files only; never import native FAFA in the proposed process."""
    if cfg.get("person_encoder", {}).get("backend", "dino") == "dino":
        return
    executable = worker_python(cfg)
    worker = Path(__file__).resolve().parents[4] / "tools/methods/cache_fafa.py"
    with tempfile.TemporaryDirectory(prefix="rcr-person-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = [executable, str(worker), "--config", str(path)]
        if prepare:
            command.append("--prepare")
        if check:
            command.append("--check")
        if force:
            command.append("--force")
        subprocess.run(command, check=True)
