"""Training losses for the proposed RCR model."""

import torch
import torch.nn.functional as F
from torch import Tensor


def grounding_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Binary Subject-person grounding loss."""

    with torch.autocast(logits.device.type, enabled=False):
        logits = logits.float()
        # Ignore padded positions BEFORE BCE: BCE(-inf, 0) can produce NaN.
        safe_logits = logits if mask is None else logits.masked_fill(~mask.bool(), 0)
        if safe_logits.numel() == 0:
            return safe_logits.sum() * 0
        loss = F.binary_cross_entropy_with_logits(
            safe_logits,
            targets.to(logits.dtype),
            reduction="none",
        )
        if mask is None:
            return loss.mean()

        mask = mask.to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1)


def identity_loss(
    identity: Tensor,
    labels: Tensor,
    temperature: float = 0.1,
    ignore_index: int = -1,
) -> Tensor:
    """Supervised contrastive loss over identity embeddings."""

    with torch.autocast(identity.device.type, enabled=False):
        identity = identity.float()
        identity = identity.reshape(-1, identity.shape[-1])
        labels = labels.reshape(-1)

        keep = labels != ignore_index
        identity = F.normalize(identity[keep], dim=-1)
        labels = labels[keep]

        if identity.shape[0] < 2:
            return identity.sum() * 0

        logits = identity @ identity.T / temperature
        self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        positive = labels[:, None].eq(labels[None, :]) & ~self_mask

        logits = logits.masked_fill(self_mask, -torch.inf)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        count = positive.sum(dim=1)
        valid = count > 0
        if not valid.any():
            return identity.sum() * 0

        loss = -(log_prob.masked_fill(~positive, 0).sum(dim=1) / count.clamp_min(1))
        return loss[valid].mean()


def retrieval_loss(
    scores: Tensor,
    positive_mask: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Mean pairwise softplus(negative - positive) over valid pairs."""

    with torch.autocast(scores.device.type, enabled=False):
        scores = scores.float()
        if valid_mask is None:
            valid_mask = torch.ones_like(positive_mask, dtype=torch.bool)

        valid_mask = valid_mask.bool()
        positive_mask = positive_mask.bool() & valid_mask
        negative_mask = valid_mask & ~positive_mask
        pairs = positive_mask[:, :, None] & negative_mask[:, None, :]
        if not pairs.any():
            return scores.sum() * 0
        differences = scores[:, None, :] - scores[:, :, None]
        return F.softplus(differences[pairs]).mean()
