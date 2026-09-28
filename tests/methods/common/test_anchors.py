"""Tests for head-to-person association."""

import torch

from rcr.methods.common.anchors import match_heads_to_persons


def test_match_heads_to_persons() -> None:
    heads = torch.tensor(
        [
            [0.10, 0.05, 0.20, 0.15],
            [0.65, 0.05, 0.75, 0.15],
        ]
    )
    persons = torch.tensor(
        [
            [0.05, 0.00, 0.40, 0.95],
            [0.55, 0.00, 0.95, 0.95],
        ]
    )

    assert torch.equal(match_heads_to_persons(heads, persons), torch.tensor([0, 1]))


def test_matching_is_one_to_one() -> None:
    heads = torch.tensor(
        [
            [0.10, 0.05, 0.20, 0.15],
            [0.30, 0.05, 0.40, 0.15],
        ]
    )
    persons = torch.tensor([[0.00, 0.00, 0.50, 1.00]])

    match = match_heads_to_persons(heads, persons)
    assert match.shape == (1,)
    assert match.item() in {0, 1}


def test_unmatched_person_returns_minus_one() -> None:
    heads = torch.tensor([[0.05, 0.05, 0.15, 0.15]])
    persons = torch.tensor(
        [
            [0.00, 0.00, 0.30, 0.90],
            [0.60, 0.00, 1.00, 0.90],
        ]
    )

    assert torch.equal(match_heads_to_persons(heads, persons), torch.tensor([0, -1]))
