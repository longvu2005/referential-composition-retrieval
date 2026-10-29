"""Failed annotation writes must leave both saved labels and UI state intact."""

import errno
import importlib
import threading

import pytest

from rcr.common.io import load_jsonl, write_jsonl


@pytest.mark.parametrize(
    "module_name,class_name,attribute",
    [
        ("review", "ReviewState", "reviews"),
        ("positives", "PositiveState", "positives"),
    ],
)
@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_save_does_not_mark_annotation_complete(
    tmp_path, monkeypatch, module_name, class_name, attribute, failure
):
    app = importlib.import_module(f"labelstudio.{module_name}.app")
    source = {"sample_id": "q"}
    original = {"sample_id": "q", "decision": "saved"}
    replacement = {"sample_id": "q", "decision": "edited"}
    state = getattr(app, class_name).__new__(getattr(app, class_name))
    state.rows = [source]
    state.rows_by_id = {"q": source}
    state.lock = threading.Lock()
    setattr(state, attribute, {"q": original})
    output = tmp_path / "annotations.jsonl"
    write_jsonl(output, [original])
    monkeypatch.setattr(app, "OUTPUT", output)
    monkeypatch.setattr(app, "validate_submission", lambda *_: replacement)

    def fail(*_):
        raise OSError(errno.ENOSPC, "simulated write failure")

    with monkeypatch.context() as patch:
        if failure == "write":
            patch.setattr(app, "write_jsonl", fail)
        else:
            patch.setattr(app.os, "replace", fail)
        with pytest.raises(OSError, match="simulated write failure"):
            state.save(source)
    assert getattr(state, attribute) == {"q": original}
    assert load_jsonl(output) == [original]

    assert state.save(source) == replacement
    assert getattr(state, attribute) == {"q": replacement}
    assert load_jsonl(output) == [replacement]
