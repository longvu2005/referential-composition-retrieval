from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "configs" / "methods" / "proposed"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def test_proposed_configs_are_consistent() -> None:
    build = _load("build_cache.yaml")
    train = _load("train.yaml")
    retrieve = _load("retrieve.yaml")
    evaluate = _load("evaluate.yaml")

    assert build["data"]["final_dir"] == train["data"]["final_dir"]
    assert train["data"]["final_dir"] == retrieve["data"]["final_dir"]
    assert retrieve["data"]["final_dir"] == evaluate["data"]["final_dir"]

    assert build["data"]["cache"] == train["data"]["cache"]
    assert train["data"]["cache"] == retrieve["data"]["cache"]

    train_dir = Path(train["output"]["dir"])
    assert Path(retrieve["checkpoint"]) == train_dir / "best.pt"

    retrieve_dir = Path(retrieve["output"]["dir"])
    assert Path(evaluate["rankings"]) == retrieve_dir / "rankings.pt"
    assert retrieve["split"] == evaluate["split"]
    assert max(evaluate["candidate_ks"]) <= retrieve["retrieval"]["top_m"]

    periodic = train["evaluation"]
    assert periodic["every_epochs"] > 0
    assert periodic["train_max_queries"] >= 0
    assert max(periodic["candidate_ks"]) <= periodic["top_m"]

    assert train["wandb"]["log_every_steps"] > 0
    assert train["wandb"]["mode"] in {"online", "offline", "disabled"}


def test_state_defaults_match_train_and_smoke_configs() -> None:
    train, smoke = _load("train.yaml"), _load("train_smoke.yaml")
    for cfg in (train, smoke):
        assert cfg["model"]["state_dim"] == 256
        assert cfg["model"]["coarse_beta"] == 0.3
        assert cfg["loss"]["state_weight"] == 1.0
        assert cfg["loss"]["state_temperature"] > 0
