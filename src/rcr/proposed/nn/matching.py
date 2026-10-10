"""Entropic partial transport: fixed row masses, real capacities <= 1, free null."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class TransportConvergenceError(RuntimeError):
    def __init__(self, diagnostics, logits, mass, real_mask, settings):
        self.details = {
            **diagnostics,
            **settings,
            "logits": logits.detach().cpu().tolist(),
            "mass": mass.detach().cpu().tolist(),
            "real_mask": real_mask.detach().cpu().tolist(),
        }
        super().__init__(
            "partial transport did not converge: "
            f"rows={diagnostics['row_residual']:.3g}, "
            f"capacity={diagnostics['capacity_violation']:.3g}, "
            f"dual={diagnostics['dual_residual']:.3g}, "
            f"iterations={diagnostics['iterations']}"
        )


def _newton_update(kernel, mass, real_mask, v, p, x, needs_update):
    """Projected, damped Newton step for the row-eliminated convex dual."""
    column_mass = x[..., 1:].sum(1)
    g = column_mass - 1
    free = real_mask & ((v < 0) | (g > 0)) & needs_update[:, None]
    pr = p[..., 1:]
    h = torch.diag_embed(column_mass) - (pr * mass[..., None]).transpose(1, 2) @ pr
    # Inactive variables stay at the upper bound. The tiny damping stabilizes
    # nearly singular Hessians; it changes the step, not the transport objective.
    h = h.masked_fill(~(free[..., None] & free[:, None]), 0)
    h = h + torch.diag_embed((~free).double() + free.double() * 1e-12)
    direction = torch.linalg.solve(h, -g.masked_fill(~free, 0))
    step = 8 / direction.abs().amax(-1).clamp_min(8)

    def objective(value):
        all_v = torch.cat((value.new_zeros(value.shape[0], 1), value), -1)
        return (mass * torch.logsumexp(kernel + all_v[:, None], -1)).sum(-1) - (
            value * real_mask
        ).sum(-1)

    before = objective(v)
    # A projected Newton direction need not descend when it crosses a bound.
    # Replace it by projected gradient descent for those batch entries.
    candidate = (v + step[:, None] * direction).clamp_max(0)
    bad_direction = (g * (candidate - v)).sum(-1) > 0
    direction = torch.where(bad_direction[:, None], -g.masked_fill(~free, 0), direction)
    step = torch.where(bad_direction, torch.ones_like(step), step)
    accepted = torch.zeros_like(needs_update)
    result = v
    roundoff = 8 * torch.finfo(v.dtype).eps * (1 + before.abs())
    for _ in range(24):
        candidate = (v + step[:, None] * direction).clamp_max(0)
        slope = (g * (candidate - v)).sum(-1)
        ok = objective(candidate) <= before + 1e-4 * slope + roundoff
        result = torch.where((ok & ~accepted)[:, None], candidate, result)
        accepted = accepted | ok
        if accepted.all():
            return result
        step = torch.where(accepted, step, step * 0.5)
    # Preserve the original monotone dual block update if line search cannot
    # resolve a decrease numerically. Never accept an unchecked Newton step.
    block = (v - column_mass.clamp_min(1e-30).log()).clamp_max(0)
    block = block.masked_fill(~real_mask, 0)
    return torch.where((accepted | ~needs_update)[:, None], result, block)


def partial_transport(
    logits, mass, real_mask, tau=0.2, max_iterations=512, tolerance=1e-5
):
    """Differentiable hybrid block/Newton minimization of the convex dual.

    logits [B,Q,1+T], mass [B,Q], real_mask [B,T]. The real column
    multipliers are nonnegative (log scaling <= 0); null multiplier is zero.
    Eight log-domain block updates initialize a projected Newton solve. Stop only
    after feasibility AND complementary slackness/stationarity checks. The small
    transport problem is solved in FP64; model outputs remain FP32 (FP64 inputs
    retain FP64 outputs). All accepted updates stay in the autograd graph.
    """
    if (
        not math.isfinite(tau)
        or tau <= 0
        or max_iterations < 1
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("invalid transport solver settings")
    with torch.autocast(logits.device.type, enabled=False):
        output_dtype = (
            torch.float64
            if torch.float64 in (logits.dtype, mass.dtype)
            else torch.float32
        )
        if (
            not torch.isfinite(logits).all()
            or not torch.isfinite(mass).all()
            or (mass < 0).any()
        ):
            raise ValueError("transport requires finite scores and nonnegative masses")
        b, q, columns = logits.shape
        if b == 0 or q == 0:
            return logits.new_zeros(b, q, columns, dtype=output_dtype), {
                "row_residual": 0.0,
                "capacity_violation": 0.0,
                "dual_residual": 0.0,
                "iterations": 0,
            }
        valid = torch.cat((real_mask.new_ones(b, 1), real_mask), -1)
        kernel = (logits.double() / tau).masked_fill(~valid[:, None], -torch.inf)
        # Row shifts do not change the optimum and reduce cancellation in the
        # dual line search. Null is always valid, including empty target sets.
        kernel = kernel - kernel.amax(-1, keepdim=True)
        weights = mass.double()
        v = kernel.new_zeros(b, columns - 1)
        for iteration in range(max_iterations + 1):
            all_v = torch.cat((v.new_zeros(b, 1), v), -1)
            p = torch.softmax(kernel + all_v[:, None], -1)
            x = weights[..., None] * p
            column_mass = x[..., 1:].sum(1)
            g = column_mass - 1
            # For v_j < 0 the capacity is active: column mass must equal 1.
            # At v_j = 0 it may be less than 1, but must not exceed it.
            kkt = torch.where(v < 0, g.abs(), g.clamp_min(0))
            kkt = kkt.masked_fill(~real_mask, 0)
            row_error = (x.sum(-1) - weights).abs().amax()
            col_error = g.clamp_min(0).amax() if columns > 1 else x.new_zeros(())
            dual_error = kkt.amax() if columns > 1 else x.new_zeros(())
            diagnostics = {
                "row_residual": row_error.detach().item(),
                "capacity_violation": col_error.detach().item(),
                "dual_residual": dual_error.detach().item(),
                "iterations": iteration,
            }
            if max(row_error.item(), col_error.item(), dual_error.item()) <= tolerance:
                return x.to(output_dtype), diagnostics
            if iteration == max_iterations:
                break
            needs_update = kkt.amax(-1) > tolerance
            if iteration < 8:
                block = (v - column_mass.clamp_min(1e-30).log()).clamp_max(0)
                block = block.masked_fill(~real_mask, 0)
                v = torch.where(needs_update[:, None], block, v)
            else:
                v = _newton_update(kernel, weights, real_mask, v, p, x, needs_update)
        raise TransportConvergenceError(
            diagnostics,
            logits,
            mass,
            real_mask,
            {"tau": tau, "max_iterations": max_iterations, "tolerance": tolerance},
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
