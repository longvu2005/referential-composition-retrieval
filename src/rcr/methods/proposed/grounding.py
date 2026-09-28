"""Ground selection text to query person candidates."""

from torch import Tensor, nn

from rcr.methods.proposed.binding import EvidenceBinding


class SubjectGrounding(nn.Module):
    """Predict one logit for every subject-person pair."""

    def __init__(self, binding: EvidenceBinding, dim: int) -> None:
        super().__init__()
        self.binding = binding
        self.score = nn.Linear(dim, 1)

    def forward(
        self,
        scene: Tensor,
        persons: Tensor,
        boxes: Tensor,
        selections: Tensor,
        patch_hw: tuple[int, int],
        selection_mask: Tensor | None = None,
    ) -> Tensor:
        """Return logits [B,S,K] from selections [B,S,L,D]."""

        b, s, length, d = selections.shape
        p = scene.shape[1]
        k = persons.shape[1]
        if k == 0:
            return persons.new_empty(b, s, 0)

        scene = scene[:, None].expand(-1, s, -1, -1).reshape(b * s, p, d)
        persons = persons[:, None].expand(-1, s, -1, -1).reshape(b * s, k, d)
        boxes = boxes[:, None].expand(-1, s, -1, -1).reshape(b * s, k, 4)
        selections = selections.reshape(b * s, length, d)
        if selection_mask is not None:
            selection_mask = selection_mask.reshape(b * s, length)
            empty = ~selection_mask.any(-1)
            if empty.any():
                selection_mask = selection_mask.clone()
                selection_mask[empty, 0] = True

        bound = self.binding(
            scene,
            persons,
            boxes,
            selections,
            patch_hw,
            selection_mask,
        ).reshape(b, s, k, d)
        return self.score(bound).squeeze(-1)
