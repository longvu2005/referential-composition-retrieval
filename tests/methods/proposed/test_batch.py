import torch
from torch import nn

from rcr.methods.proposed.batch import build_batch, build_supervision


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
