"""One finite optimizer update per batch, with a bounded FP32 retry for AMP."""

import torch


class NonfiniteStepError(FloatingPointError):
    def __init__(self, attempts):
        self.details = {"attempts": attempts}
        super().__init__(
            "nonfinite training step in FP32; optimizer not stepped; "
            "see nonfinite_batch.json"
        )


def train_step(model, optimizer, scaler, loss_fn, device, *, amp, max_grad_norm):
    """Retry the same batch once without autocast/scaling, never skip its update.

    The caller owns diagnostics and epoch accounting. Only GradScaler's rejected
    AMP step may see nonfinite gradients; no optimizer update uses those values.
    """
    device = torch.device(device)
    cpu_rng = torch.get_rng_state() if amp else None
    cuda_rng = (
        torch.cuda.get_rng_state(device) if amp and device.type == "cuda" else None
    )
    attempts = []
    for mixed in [True, False] if amp else [False]:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=mixed):
            loss, parts = loss_fn()
        failure = None
        if not torch.isfinite(loss):
            failure = {"reason": "nonfinite loss"}
        elif loss.requires_grad:
            (scaler.scale(loss) if mixed else loss).backward()
            if mixed:
                scaler.unscale_(optimizer)
            grads = [
                (name, p) for name, p in model.named_parameters() if p.grad is not None
            ]
            finite = torch.stack([torch.isfinite(p.grad).all() for _, p in grads])
            if not finite.all():
                failure = {
                    "reason": "nonfinite gradients",
                    "parameters": [
                        name
                        for (name, _), ok in zip(grads, finite.tolist(), strict=True)
                        if not ok
                    ],
                }
                if mixed:
                    # unscale_ recorded the overflow: step skips AdamW entirely,
                    # including weight decay/moments, then update lowers the scale.
                    scaler.step(optimizer)
                    scaler.update()
            else:
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm, error_if_nonfinite=True
                    )
                except RuntimeError as error:
                    # Finite elements can still overflow the norm reduction.
                    if "non-finite" not in str(error):
                        raise
                    failure = {"reason": "nonfinite gradient norm"}
                    if mixed:
                        # Do not step: scaler only checks elements, not the norm.
                        scaler.update(new_scale=scaler.get_scale())
                else:
                    if mixed:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
        if failure is None:
            return loss.detach(), {**parts, "fp32_retries": len(attempts)}
        attempts.append(
            {
                "precision": "amp_float16" if mixed else "float32",
                "loss": str(loss.detach().item()),
                **failure,
            }
        )
        # Release the failed graph/gradients before recomputing. Restore dropout's
        # RNG stream so a retry does not consume a second batch's random draws.
        del loss
        optimizer.zero_grad(set_to_none=True)
        if mixed:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
    raise NonfiniteStepError(attempts)
