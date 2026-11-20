"""Collect device predicates and read them at an existing CPU boundary."""

from contextvars import ContextVar
from functools import wraps

import torch

_current = ContextVar("rcr_tensor_checks", default=None)


class TensorChecks:
    """Validate before side effects, without one CUDA wait per predicate.

    A scope owns its checks. Nested scoring/loss functions add predicates to
    that scope; standalone calls still validate before returning. CPU consumers
    can copy values and pending predicates in one transfer.
    """

    def __init__(self):
        self.pending = []

    def __enter__(self):
        self.token = _current.set(self)
        return self

    def __exit__(self, *error):
        _current.reset(self.token)
        if error[0] is None:
            self.validate()

    def require(self, condition, message, error=ValueError):
        condition = condition.detach().bool()
        condition = condition.reshape(()) if condition.numel() == 1 else condition.all()
        self.pending.append((condition, message, error))

    def cpu_values(self, values):
        """Return FP32 CPU views and raise pending errors from the same copy."""
        devices = dict.fromkeys(
            [value.device for value in values.values()]
            + [condition.device for condition, _, _ in self.pending]
        )
        result, statuses = {}, [None] * len(self.pending)
        for device in devices:
            local = {
                key: value for key, value in values.items() if value.device == device
            }
            checks = [
                (index, condition)
                for index, (condition, _, _) in enumerate(self.pending)
                if condition.device == device
            ]
            tensors = [value.detach().float().flatten() for value in local.values()]
            if checks:
                tensors.append(torch.stack([value for _, value in checks]).float())
            packet = (torch.cat(tensors) if len(tensors) > 1 else tensors[0]).cpu()
            offset = 0
            for key, value in local.items():
                size = value.numel()
                result[key] = packet[offset : offset + size].reshape(value.shape)
                offset += size
            for (index, _), valid in zip(checks, packet[offset:].tolist(), strict=True):
                statuses[index] = valid
        pending, self.pending = self.pending, []
        for valid, (_, message, error) in zip(statuses, pending, strict=True):
            if not valid:
                raise error(message)
        return {key: result[key] for key in values}

    def validate(self):
        if self.pending:
            self.cpu_values({})


def check_tensor(condition, message, error=ValueError):
    checks = _current.get()
    if checks is None:
        checks = TensorChecks()
        checks.require(condition, message, error)
        checks.validate()
    else:
        checks.require(condition, message, error)


def cpu_tensors(values):
    checks = _current.get()
    return (checks if checks is not None else TensorChecks()).cpu_values(values)


def checked(function):
    """Keep public error checking while coalescing nested device checks."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _current.get() is not None:
            return function(*args, **kwargs)
        with TensorChecks():
            return function(*args, **kwargs)

    return wrapped
