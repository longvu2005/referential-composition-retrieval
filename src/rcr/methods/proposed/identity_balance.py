"""Parameter-free identity ablations, initial calibration and masked norm reports."""

import hashlib
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


class IdentityBalance(nn.Module):
    """A: P(i); B: P(c*i); C: R*unit(LN(P(i))); D: alpha*R*unit(P(i))."""

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        self.mode = cfg.get("mode", "none")
        self.scale = float(cfg.get("scale", 1.0))
        self.alpha = float(cfg.get("alpha", 1.0))
        if self.mode not in ("none", "pre_scale", "layernorm", "l2"):
            raise ValueError(f"unknown identity_balance mode: {self.mode}")
        if any(not math.isfinite(x) or x <= 0 for x in (self.scale, self.alpha)):
            raise ValueError("identity scale and alpha must be finite and positive")
        # No parameters/random draws: adding an ablation cannot shift initialization.
        # Legacy checkpoints without an explicit config have no radius keys.
        self.register_buffer("radius", torch.tensor(0.0), persistent=config is not None)
        self._radius_checked = False
        self.register_load_state_dict_post_hook(self._reset_radius_check)

    def _reset_radius_check(self, module, incompatible_keys):
        self._radius_checked = False

    def forward(self, identity, projection):
        inputs = identity * self.scale if self.mode == "pre_scale" else identity
        projected = projection(inputs)
        if self.mode in ("none", "pre_scale"):
            return projected
        if not self._radius_checked:
            radius = self.radius.item()
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError(
                    "identity radius is uninitialized; calibrate before training"
                )
            self._radius_checked = True
        value = projected.float()
        if self.mode == "layernorm":
            value = F.layer_norm(value, (value.shape[-1],), weight=None, bias=None)
        value = F.normalize(value, dim=-1, eps=1e-6)
        alpha = self.alpha if self.mode == "l2" else 1.0
        return (value * (alpha * self.radius)).to(projected.dtype)


class BranchNorms:
    """Token-weighted L2 moments; masks exclude padding and absent Subjects."""

    def __init__(self):
        self.values = {}

    @torch.no_grad()
    def add(self, name, tensor, mask=None):
        norms = tensor.detach().float().norm(dim=-1)
        if mask is not None:
            norms = norms[mask.bool()]
        norms = norms.flatten().double()
        if not norms.numel():
            return
        # Accumulate on the current device, synchronize only when reporting.
        row = self.values.setdefault(name, [0, 0.0, 0.0])
        row[0] += norms.numel()
        row[1] = row[1] + norms.sum()
        row[2] = row[2] + norms.square().sum()

    def report(self):
        result = {}
        for key, (n, total, square) in self.values.items():
            total, square = total.item(), square.item()
            if not math.isfinite(total) or not math.isfinite(square):
                raise ValueError(f"nonfinite branch norm: {key}")
            result[key] = {
                "count": n,
                "mean": total / n,
                "std": math.sqrt(max(0.0, square / n - (total / n) ** 2)),
            }
        return result


@contextmanager
def collect_branch_norms(model, enabled=True):
    collector = BranchNorms() if enabled else None
    modules = (model.composition, model.target_builder)
    previous = [getattr(module, "norm_observer", None) for module in modules]
    for module in modules:
        module.norm_observer = collector
    try:
        yield collector
    finally:
        for module, observer in zip(modules, previous, strict=True):
            module.norm_observer = observer


def parameter_fingerprint(*modules):
    """Hash trainable initialization (radii and ablation options are excluded)."""
    digest = hashlib.sha256()
    for index, module in enumerate(modules):
        for name, value in module.named_parameters():
            digest.update(f"{index}:{name}:{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def calibrate_identity(
    model,
    samples,
    cache,
    tokenizer,
    text_encoder,
    device,
    *,
    max_queries,
    text_cache=None,
):
    """Measure on canonical train pairs with the unmodified initial A forward.

    Query role radius averages the role table. Target radius averages valid
    person-token norms of evidence_proj(e) + box_proj(g). No val/test data,
    dropout, updates, or random sampling. Restore RNG and modes even on failure.
    """
    from rcr.methods.proposed.encoders import encode_query_text

    if not isinstance(max_queries, int) or max_queries < 1:
        raise ValueError("calibration_queries must be a positive integer")
    rows = sorted(samples, key=lambda row: row["sample_id"])[:max_queries]
    if not rows:
        raise ValueError("identity calibration requires train samples")
    balances = (
        model.composition.identity_balance,
        model.target_builder.identity_balance,
    )
    modes = [balance.mode for balance in balances]
    training = model.training, text_encoder.training
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        model.eval()
        text_encoder.eval()
        for balance in balances:
            balance.mode = "none"
        with (
            torch.random.fork_rng(devices=devices),
            collect_branch_norms(model) as norms,
        ):
            for sample in rows:
                text = encode_query_text(
                    [sample], tokenizer, text_encoder, device, text_cache=text_cache
                )
                q = cache.load(torch.tensor([cache.by_id[sample["query_image_id"]]]))
                _, _, query, mask, prior = model.encode_query(
                    q[0].to(device),
                    q[1].to(device),
                    q[2].to(device),
                    patch_hw=cache.patch_hw,
                    query_person_mask=q[4].to(device),
                    **text,
                )
                t = cache.load(torch.tensor([cache.by_id[sample["target_image_id"]]]))
                model.score_target(
                    query,
                    mask,
                    prior,
                    t[0].to(device),
                    t[1].to(device),
                    t[2].to(device),
                    cache.patch_hw,
                    t[4].to(device),
                )
        report = norms.report()
        rq = model.composition.role.weight.float().norm(dim=-1).mean().item()
        rt = report.get("target.evidence_geometry", {}).get("mean", 0.0)
        if any(not math.isfinite(x) or x <= 0 for x in (rq, rt)):
            raise ValueError("calibration needs positive radii and valid target people")
        for balance, radius in zip(balances, (rq, rt), strict=True):
            balance.radius.fill_(radius)
        return {
            "R_q": rq,
            "R_t": rt,
            "sample_ids": [r["sample_id"] for r in rows],
            "reference_norms": report,
            "reference_mode": "none",
            "split": "train",
        }
    finally:
        for balance, mode in zip(balances, modes, strict=True):
            balance.mode = mode
        model.train(training[0])
        text_encoder.train(training[1])
