"""Entropic partial transport: fixed row masses, real capacities <= 1, free null."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def partial_transport(
    logits, mass, real_mask, tau=0.2, max_iterations=512, tolerance=1e-5
):
    """Differentiable log-domain block minimization of the convex transport dual.

    logits [B,Q,1+T], mass [B,Q], real_mask [B,T]. The real column
    multipliers are nonnegative (log scaling <= 0); null multiplier is zero.
    Each update minimizes one dual block, not balanced Sinkhorn projection.
    Stop only after feasibility AND complementary slackness/stationarity checks.
    """
    if tau <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid transport solver settings")
    with torch.autocast(logits.device.type, enabled=False):
        logits, mass = logits.float(), mass.float()
        if (
            not torch.isfinite(logits).all()
            or not torch.isfinite(mass).all()
            or (mass < 0).any()
        ):
            raise ValueError("transport requires finite scores and nonnegative masses")
        b, q, columns = logits.shape
        if q == 0:
            return logits.new_zeros(b, q, columns), {
                "row_residual": 0.0,
                "capacity_violation": 0.0,
                "dual_residual": 0.0,
                "iterations": 0,
            }
        valid = torch.cat((real_mask.new_ones(b, 1), real_mask), -1)
        # Finite masked logits avoid undefined all-masked logsumexp gradients.
        # Invalid rows/columns are explicitly zeroed in the final coupling.
        kernel = (logits / tau).masked_fill(~valid[:, None], -1e9)
        active = mass > 0
        log_mass = mass.clamp_min(1e-30).log()
        v = kernel.new_zeros(b, columns)
        for iteration in range(max_iterations):
            u = log_mass - torch.logsumexp(kernel + v[:, None], -1)
            u = u.masked_fill(~active, -1e9)
            column_logsum = torch.logsumexp(kernel + u[..., None], 1)
            new_v = torch.minimum(-column_logsum, torch.zeros_like(v))
            new_v = torch.cat((torch.zeros_like(new_v[:, :1]), new_v[:, 1:]), -1)
            delta = (new_v - v).abs().amax()
            v = new_v
            if (iteration + 1) % 8 == 0 or iteration + 1 == max_iterations:
                u = (log_mass - torch.logsumexp(kernel + v[:, None], -1)).masked_fill(
                    ~active, -1e9
                )
                x = (
                    (kernel + u[..., None] + v[:, None])
                    .exp()
                    .masked_fill(~active[..., None] | ~valid[:, None], 0)
                )
                row_error = (x.sum(-1) - mass).abs().amax()
                col_error = (
                    (x[..., 1:].sum(1) - 1).clamp_min(0).amax()
                    if columns > 1
                    else x.new_zeros(())
                )
                if (
                    max(
                        row_error.detach().item(),
                        col_error.detach().item(),
                        delta.detach().item(),
                    )
                    <= tolerance
                ):
                    return x, {
                        "row_residual": row_error.detach().item(),
                        "capacity_violation": col_error.detach().item(),
                        "dual_residual": delta.detach().item(),
                        "iterations": iteration + 1,
                    }
        raise RuntimeError(
            f"partial transport did not converge: rows={row_error.item():.3g}, "
            f"capacity={col_error.item():.3g}, dual={delta.item():.3g}; "
            "increase max_iterations or tau"
        )


class IdentityMatching(nn.Module):
    def __init__(self, tau=0.2, max_iterations=512, tolerance=1e-5):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(5.0))))
        self.beta = nn.Parameter(torch.tensor(-2.0))
        self.null_logit = nn.Parameter(torch.tensor(-1.0))
        self.tau, self.max_iterations, self.tolerance = tau, max_iterations, tolerance

    @property
    def alpha(self):
        return F.softplus(self.log_alpha.float()) + 1e-6

    def confidence(self, similarity):
        return torch.sigmoid(self.alpha * similarity.float() + self.beta.float())

    def forward(self, query, target, membership, target_mask):
        with torch.autocast(query.device.type, enabled=False):
            similarity = query.detach().float() @ target.detach().float().transpose(
                -1, -2
            )
            log_prob = F.logsigmoid(self.alpha * similarity + self.beta.float())
            null = self.null_logit.float().expand(*similarity.shape[:2], 1)
            mass = membership.detach().float().sum(1)
            x, diagnostics = partial_transport(
                torch.cat((null, log_prob), -1),
                mass,
                target_mask,
                self.tau,
                self.max_iterations,
                self.tolerance,
            )
            p = x / (mass[..., None] + 1e-8)
            confidence = log_prob.exp().masked_fill(~target_mask[:, None], 0)
        return p, confidence, diagnostics
