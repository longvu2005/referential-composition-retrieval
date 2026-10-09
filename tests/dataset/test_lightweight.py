"""Annotation commands and JSON reporting work without model dependencies."""

import json
import os
import subprocess
import sys
from pathlib import Path

from rcr.common.io import write_jsonl
from tests.dataset.test_incremental_pipeline import _source


def test_prepare_review_and_ui_without_model_libraries(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = _source("1_1", "1_2", "7", "TRAIN")
    write_jsonl(tmp_path / "source.jsonl", [source])
    boxes = [
        {
            "image_id": source[f"{side}_image_id"],
            "label": "7",
            **{
                key: value
                for key, value in source[f"{side}_boxes"][0].items()
                if key != "identity_id"
            },
        }
        for side in ("query", "target")
    ]
    pairs = {
        "pairs": [
            {
                "pair_id": source["sample_id"],
                "split": "TRAIN",
                "query_image_id": "1_1",
                "target_image_id": "1_2",
            }
        ],
        "boxes": boxes,
    }
    (tmp_path / "pairs.json").write_text(json.dumps(pairs))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
from pathlib import Path

heavy = {'torch', 'torchvision', 'transformers', 'clip', 'open_clip', 'scipy', 'wandb'}
class MissingModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in heavy:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, MissingModels())

from rcr.common.io import load_jsonl
from tools.data import build_final, prepare_positives, prepare_review, select_samples
from tools import report
from labelstudio.review.app import build_task_payload
from labelstudio.positives import app

select_samples.main(['--annotations', 'source.jsonl', '--pair-data', 'pairs.json',
                     '--output', 'selected.jsonl'])
args = ['--selected', 'selected.jsonl', '--output', 'review.jsonl',
        '--reviewed', 'reviewed.jsonl']
prepare_review.main(args)
before = Path('review.jsonl').read_bytes()
prepare_review.main(args)
assert before == Path('review.jsonl').read_bytes()
assert not Path('reviewed.jsonl').exists()
assert not Path('dataset/data/work/rewrite').exists()
row = load_jsonl('review.jsonl')[0]
assert build_task_payload(row)['final_desc'] == row['final_desc']
try:
    report.main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not heavy & sys.modules.keys()
""",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
