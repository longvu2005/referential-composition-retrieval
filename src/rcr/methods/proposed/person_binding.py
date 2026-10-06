"""Match identity and requested semantics on the same target person."""

import torch
from torch import Tensor

BINDING_MODES = ("both", "identity", "semantic", "none")


def binding_scores(
    query_identity: Tensor,
    target_identity: Tensor,
    composed: Tensor,
    target_semantic: Tensor,
    membership: Tensor,
    subject_mask: Tensor,
    target_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return [B] scores; max follows pairwise fusion, then soft set pooling.

    composed/membership are [B,S,Kq,D] / [B,S,Kq]. Identities are [B,K,Di].
    Identity is detached here, at the only identity input to dual fine scoring.
    Empty person sets contribute zero. This optimistic match permits reuse of a
    target person; it is not a one-to-one assignment or an all-members guarantee.
    """
    with torch.autocast(composed.device.type, enabled=False):
        zero = composed.float().sum(dim=(1, 2, 3)) * 0
        if composed.shape[2] == 0 or target_semantic.shape[1] == 0:
            return dict.fromkeys(BINDING_MODES, zero)
        identity = torch.einsum(
            "bqd,btd->bqt",
            query_identity.detach().float(),
            target_identity.detach().float(),
        )[:, None]
        semantic = torch.einsum(
            "bsqd,btd->bsqt", composed.float(), target_semantic.float()
        )
        weights = membership.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        active = subject_mask.bool()

        def pool(scores):
            if target_mask is not None:
                scores = scores.masked_fill(
                    ~target_mask[:, None, None].bool(), -torch.inf
                )
            best = scores.amax(-1)
            # All-masked targets must not produce -inf * zero or NaN gradients.
            best = best.masked_fill(~torch.isfinite(best), 0)
            per_subject = (best * weights).sum(-1).masked_fill(~active, 0)
            return per_subject.sum(-1) / active.sum(-1).clamp_min(1)

        return {
            "both": pool(identity + semantic),
            "identity": pool(identity.expand_as(semantic)),
            "semantic": pool(semantic),
            "none": zero,
        }
