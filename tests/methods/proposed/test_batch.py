import pytest
import torch
from torch import nn

from rcr.methods.proposed.batch import build_batch, build_supervision
from rcr.methods.proposed.cache import GalleryCache
from rcr.methods.proposed.model import RCRModel
from rcr.methods.proposed.training import compute_loss


class _Tokenizer:
    def __init__(self) -> None:
        self.vocab = {"[PAD]": 0, "[S1]": 1, "[S2]": 2}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def __call__(self, texts, padding=True, return_tensors="pt"):
        rows = [[self._id(x) for x in text.split()] for text in texts]
        length = max(map(len, rows))
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        mask = torch.zeros(len(rows), length, dtype=torch.long)
        for n, row in enumerate(rows):
            ids[n, : len(row)] = torch.tensor(row)
            mask[n, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]


class _TextEncoder(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, dim)

    def forward(self, ids, mask):
        return self.embedding(ids), mask.bool()


class _Cache:
    def __init__(self) -> None:
        self.image_ids = ["q1", "q2", "p1", "n1", "p2", "n2"]
        self.patch_hw = (1, 2)

    def load(self, indices):
        n = len(indices)
        scene = torch.randn(n, 2, 4)
        persons = torch.randn(n, 3, 4)
        boxes = torch.tensor(
            [[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0], [0, 0, 0, 0]]] * n
        )
        ids = []
        for index in indices.tolist():
            image_id = self.image_ids[index]
            if image_id == "q1":
                ids.append(["10", "20", None])
            elif image_id == "q2":
                ids.append(["10", "30", None])
            elif image_id == "p1":
                ids.append(["20", "10", None])
            elif image_id == "p2":
                ids.append(["30", "40", None])
            else:
                ids.append([None, None, None])
        mask = torch.tensor([[True, True, False]] * n)
        return scene, persons, boxes, ids, mask


def _samples():
    return [
        {
            "query_image_id": "q1",
            "positive_image_ids": ["p1"],
            "subjects": [
                {"subject_id": 1, "identity_ids": ["10"]},
                {"subject_id": 2, "identity_ids": ["20"]},
            ],
            "final_desc": (
                "Identify Subject 1 as the man in white and "
                "Subject 2 as the woman in black"
            ),
            "final_change": "then Subject 1 stands behind Subject 2",
        },
        {
            "query_image_id": "q2",
            "positive_image_ids": ["p2"],
            "subjects": [
                {"subject_id": 1, "identity_ids": ["10"]},
                {"subject_id": 2, "identity_ids": ["30"]},
            ],
            "final_desc": (
                "Identify Subject 1 as the man in blue and "
                "Subject 2 as the woman in red"
            ),
            "final_change": "then Subject 2 stands beside Subject 1",
        },
    ]


def test_build_supervision() -> None:
    identity_ids = [
        ["10", "20", None],
        ["10", "30", None],
    ]
    subjects = [x["subjects"] for x in _samples()]

    targets, labels = build_supervision(identity_ids, subjects)

    torch.testing.assert_close(
        targets,
        torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        ),
    )
    torch.testing.assert_close(labels, torch.tensor([[0, 1, -1], [0, 2, -1]]))


def test_group_subject_marks_all_identities() -> None:
    targets, labels = build_supervision(
        [["10", "20", "30"]],
        [[{"subject_id": 1, "identity_ids": ["10", "30"]}]],
    )

    torch.testing.assert_close(targets, torch.tensor([[[1.0, 0.0, 1.0]]]))
    torch.testing.assert_close(labels, torch.tensor([[0, 1, 2]]))


def test_build_batch() -> None:
    tokenizer = _Tokenizer()
    text_encoder = _TextEncoder()
    candidates = [["p1", "n1"], ["n2", "p2"]]

    batch = build_batch(
        _samples(),
        candidates,
        _Cache(),
        tokenizer,
        text_encoder,
        "cpu",
    )

    assert batch["query_scene"].shape == (2, 2, 4)
    assert batch["query_persons"].shape == (2, 3, 4)
    assert batch["selections"].shape[:2] == (2, 2)
    assert batch["change"].shape[0] == 2
    assert batch["target_scene"].shape == (2, 2, 2, 4)
    assert batch["target_persons"].shape == (2, 2, 3, 4)

    torch.testing.assert_close(
        batch["positive_mask"],
        torch.tensor([[True, False], [False, True]]),
    )
    torch.testing.assert_close(
        batch["grounding_targets"],
        torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        ),
    )

    change_ids = tokenizer(
        [
            "then [S1] stands behind [S2]",
            "then [S2] stands beside [S1]",
        ],
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    for n in range(2):
        for s, marker in enumerate(("[S1]", "[S2]")):
            pos = batch["subject_pos"][n, s]
            assert change_ids[n, pos] == tokenizer.convert_tokens_to_ids(marker)


def test_build_batch_tracks_repeated_subject_mentions() -> None:
    tokenizer = _Tokenizer()
    text_encoder = _TextEncoder()
    sample = _samples()[0]
    sample = {**sample}
    sample["final_change"] = (
        "then Subject 1 is beside Subject 2 and Subject 1 is looking at Subject 2"
    )

    batch = build_batch(
        [sample],
        [["p1", "n1"]],
        _Cache(),
        tokenizer,
        text_encoder,
        "cpu",
    )

    mentions = batch["subject_token_mask"][0]
    assert mentions[0].sum() == 2
    assert mentions[1].sum() == 2
    assert mentions[0, batch["subject_pos"][0, 0]]
    assert mentions[1, batch["subject_pos"][0, 1]]


def test_query_and_target_identities_share_labels() -> None:
    batch = build_batch(
        _samples(),
        [["p1", "n1"], ["n2", "p2"]],
        _Cache(),
        _Tokenizer(),
        _TextEncoder(),
        "cpu",
    )
    assert batch["query_identity_labels"].tolist() == [[0, 1, -1], [0, 2, -1]]
    assert batch["target_identity_labels"].tolist() == [
        [[1, 0, -1], [-1, -1, -1]],
        [[-1, -1, -1], [2, 3, -1]],
    ]
    assert batch["query_identity_mask"].tolist() == [[True, True, False]] * 2
    assert batch["target_identity_mask"].tolist() == [
        [[True, True, False], [False, False, False]],
        [[False, False, False], [True, True, False]],
    ]


def test_identity_masks_deduplicate_images_across_queries_and_targets() -> None:
    sample = {**_samples()[0], "positive_image_ids": ["q1", "p1"]}
    batch = build_batch(
        [sample, sample],
        [["q1", "p1", "p2"], ["p1", "p1", "p2"]],
        _Cache(),
        _Tokenizer(),
        _TextEncoder(),
        "cpu",
    )
    assert batch["query_identity_mask"].tolist() == [
        [True, True, False],
        [False, False, False],
    ]
    # q1 is already a query; p1 is used once; p2 has known IDs but is negative.
    assert batch["target_identity_mask"].tolist() == [
        [[False, False, False], [True, True, False], [False, False, False]],
        [[False, False, False]] * 3,
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cached_precision_supports_batch_training(tmp_path, dtype) -> None:
    torch.manual_seed(3)
    (tmp_path / "features").mkdir()
    persons = torch.randn(3, 2, 4).to(dtype)
    torch.save(
        {
            "image_ids": ["q1", "p1", "n1"],
            "persons": persons,
            "mask": torch.ones(3, 2, dtype=torch.bool),
            "patch_hw": (1, 2),
        },
        tmp_path / "index.pt",
    )
    for i, ids in enumerate((["10", "20"], ["20", "10"], ["30", None])):
        torch.save(
            {
                "scene": torch.randn(2, 4).to(dtype),
                "persons": persons[i].clone(),
                "boxes_scene": torch.tensor(
                    [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]
                ),
                "identity_ids": ids,
            },
            tmp_path / "features" / f"{i}.pt",
        )
    cache = GalleryCache(tmp_path)
    batch = build_batch(
        [_samples()[0]],
        [["p1", "n1"]],
        cache,
        _Tokenizer(),
        _TextEncoder(),
        "cpu",
        state_image_ids=[{"p1"}],
    )
    model = RCRModel(dim=4, identity_dim=4, num_heads=2)
    loss, parts = compute_loss(model, batch, cache.patch_hw)
    assert torch.isfinite(loss)
    assert parts["identity"] > 0
    loss.backward()
    assert model.identity_head.proj.weight.grad.norm() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_state_mask_uses_gt_image_pool_not_detected_person_labels():
    batch = build_batch(
        _samples(),
        [["p1", "n1"], ["n2", "p2"]],
        _Cache(),
        _Tokenizer(),
        _TextEncoder(),
        "cpu",
        state_image_ids=[{"p1", "n1"}, {"p2"}],
    )
    # n1's detector labels are unknown, but its image-level GT has the required IDs.
    assert (batch["target_identity_labels"][0, 1] == -1).all()
    assert batch["state_mask"].tolist() == [[True, True], [False, True]]
    assert batch["grounding_complete"].tolist() == [[True, True], [True, True]]
    with pytest.raises(ValueError, match="Full Positive"):
        build_batch(
            [_samples()[0]],
            [["p1", "n1"]],
            _Cache(),
            _Tokenizer(),
            _TextEncoder(),
            "cpu",
            state_image_ids=[{"n1"}],
        )
