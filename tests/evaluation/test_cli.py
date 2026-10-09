"""Evaluate existing rankings without loading any model dependency."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


@pytest.mark.parametrize("templated", [False, True])
def test_evaluate_cli(tmp_path: Path, templated) -> None:
    final_dir = tmp_path / "final"
    split_dir = final_dir / "splits"
    split_dir.mkdir(parents=True)

    samples = [
        {
            "sample_id": "s1",
            "case_type": "INDIVIDUAL",
            "query_image_id": "q",
            "target_image_id": "a",
            "positive_image_ids": ["a", "left"],
            "subjects": [{"subject_id": 1, "identity_ids": ["p1"]}],
            "final_desc": "Identify Subject 1 as the person",
            "final_change": "then retrieve target images where Subject 1 is standing",
            "final_instruction": "x",
        }
    ]
    images = [
        {"image_id": x, "path": f"{split}/{x}.jpg"}
        for x, split in [
            ("q", "test"),
            ("a", "test"),
            ("b", "test"),
            ("left", "leftover"),
        ]
    ]
    gallery = [{"image_id": x["image_id"]} for x in images]
    heads = [
        {
            "box_id": "q::p1",
            "image_id": "q",
            "identity_id": "p1",
            "x": 0,
            "y": 0,
            "width": 1,
            "height": 1,
        },
        {
            "box_id": "a::p1",
            "image_id": "a",
            "identity_id": "p1",
            "x": 0,
            "y": 0,
            "width": 1,
            "height": 1,
        },
    ]

    for name, rows in [
        ("samples.jsonl", samples),
        ("images.jsonl", images),
        ("gallery.jsonl", gallery),
        ("head_boxes.jsonl", heads),
    ]:
        (final_dir / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    (final_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (split_dir / "train.txt").write_text("", encoding="utf-8")
    (split_dir / "val.txt").write_text("", encoding="utf-8")
    (split_dir / "test.txt").write_text("s1\n", encoding="utf-8")

    rankings_path = tmp_path / "test" / "rankings.pt"
    rankings_path.parent.mkdir()
    torch.save(
        {
            "sample_ids": ["s1"],
            "gallery_ids": ["q", "a", "b"],
            "rankings": torch.tensor([[1, 2]], dtype=torch.int32),
            "coarse_topm": torch.tensor([[1, 2]], dtype=torch.int32),
        },
        rankings_path,
    )

    output = tmp_path / "test" / "metrics.json"
    config = tmp_path / "evaluate.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "method": "proposed",
                "data": {
                    "final_dir": str(final_dir),
                    "image_root": str(tmp_path),
                },
                "split": "val" if templated else "test",
                "candidate_ks": [1, 2],
                "output": {"dir": str(tmp_path)},
            }
        ),
        encoding="utf-8",
    )

    repo = Path(__file__).parents[2]
    env = os.environ.copy()
    pythonpath = str(repo / "src")
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    subprocess.run(
        [
            sys.executable,
            "tools/run.py",
            "evaluate",
            "--config",
            str(config),
            *(["--splits", "test"] if templated else []),
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["overall"]["full_map"] == 1.0
    assert result["overall"]["full_r1"] == 1.0
    assert result["overall"]["id_map"] == 1.0
    assert result["overall"]["candidate_recall_1"] == 1.0
