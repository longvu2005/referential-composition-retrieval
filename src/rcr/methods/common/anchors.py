"""Match head boxes to full-person boxes."""

import torch
from torch import Tensor


def match_heads_to_persons(heads: Tensor, persons: Tensor) -> Tensor:
    """Return one head index per person, or -1 when no head is matched.

    Boxes are xyxy in the same coordinate system. A head can match only a
    person that contains its center. Matching is one-to-one and prefers the
    geometrically closest head-to-person pair.
    """

    num_heads = heads.shape[0]
    num_persons = persons.shape[0]
    match = torch.full((num_persons,), -1, device=persons.device, dtype=torch.long)
    if num_heads == 0 or num_persons == 0:
        return match

    hx = (heads[:, 0] + heads[:, 2]) / 2
    hy = (heads[:, 1] + heads[:, 3]) / 2

    x1, y1, x2, y2 = persons.unbind(-1)
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    cx = (x1 + x2) / 2

    inside = (
        (hx[:, None] >= x1)
        & (hx[:, None] <= x2)
        & (hy[:, None] >= y1)
        & (hy[:, None] <= y2)
    )

    dx = (hx[:, None] - cx) / w
    dy = (hy[:, None] - y1) / h
    cost = (dx.square() + dy.square()).masked_fill(~inside, torch.inf)

    used_heads = torch.zeros(num_heads, device=heads.device, dtype=torch.bool)
    for flat in cost.flatten().argsort().tolist():
        head = flat // num_persons
        person = flat % num_persons
        if not torch.isfinite(cost[head, person]):
            break
        if match[person] < 0 and not used_heads[head]:
            match[person] = head
            used_heads[head] = True

    return match
