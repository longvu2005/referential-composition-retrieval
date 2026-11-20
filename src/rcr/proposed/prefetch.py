"""Bounded CPU preparation and optional asynchronous CUDA visual transfers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import prod

import torch

from rcr.proposed.cache.store import SceneTargetBatch


@dataclass
class PackedPersons:
    """Transfer actual detections, reconstructing exact zero padding on the device."""

    values: torch.Tensor
    positions: torch.Tensor
    shape: tuple

    @classmethod
    def from_padded(cls, values, mask):
        positions = mask.flatten().nonzero().flatten()
        packed = values.reshape(-1, values.shape[-1]).index_select(0, positions)
        return cls(packed, positions, tuple(values.shape))

    def to_dense(self, device, *, copy=None):
        if copy is None:

            def copy(value):
                result = value.to(device, non_blocking=True)
                return result.float() if result.is_floating_point() else result

        values, positions = copy(self.values), copy(self.positions)
        out = values.new_zeros((prod(self.shape[:-1]), self.shape[-1]))
        return out.index_copy(0, positions, values).reshape(self.shape)


class OrderedPrefetch:
    """Prepare at most one future batch; propagate errors and join on every exit."""

    def __init__(self, inputs, prepare, *, enabled):
        self.inputs = inputs
        self.prepare = prepare
        self.pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="rcr-prefetch")
            if enabled
            else None
        )

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def __iter__(self):
        if self.pool is None:
            for row in self.inputs:
                yield self.prepare(row)
            return
        iterator = iter(self.inputs)
        try:
            row = next(iterator)
        except StopIteration:
            self.close()
            return
        pending = self.pool.submit(self.prepare, row)
        try:
            while pending is not None:
                result = pending.result()
                try:
                    row = next(iterator)
                except StopIteration:
                    pending = None
                else:
                    pending = self.pool.submit(self.prepare, row)
                yield result
        finally:
            self.close()


@dataclass
class StagedVisual:
    visual: dict
    ready: object = None
    device: object = None
    sources: list = field(default_factory=list)

    def consume(self):
        if self.ready is not None:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.ready)
            recorded = set()
            for value in self.visual.values():
                if isinstance(value, SceneTargetBatch):
                    value = value.values
                if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                    pointer = value.untyped_storage().data_ptr()
                    if pointer not in recorded:
                        value.record_stream(stream)
                        recorded.add(pointer)
        return self.visual


class VisualStager:
    """Pin/copy frozen features on the producer; execute no model or random ops."""

    def __init__(self, device, *, enabled):
        self.device = torch.device(device)
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if enabled and self.device.type == "cuda"
            else None
        )

    def __enter__(self):
        return self

    def __exit__(self, *error):
        if self.stream is not None:
            self.stream.synchronize()

    def stage(self, visual):
        if self.stream is None:
            return StagedVisual(visual)
        staged, sources = {}, []

        def copy(value, *, project_dtype=True):
            host = value if value.is_pinned() else value.pin_memory()
            sources.append(host)
            value = host.to(self.device, non_blocking=True)
            return (
                value.float() if project_dtype and value.is_floating_point() else value
            )

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            # Coalesce small masks/labels by dtype. Large visual/native payloads
            # retain their existing bounded transfer path and storage precision.
            groups = {}
            for key, value in visual.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.device.type == "cpu"
                    and value.numel() * value.element_size() <= 64 * 1024
                ):
                    groups.setdefault(value.dtype, []).append((key, value))
            for group in groups.values():
                if len(group) < 2:
                    continue
                packet = copy(torch.cat([value.flatten() for _, value in group]))
                offset = 0
                for key, value in group:
                    staged[key] = packet[offset : offset + value.numel()].reshape(
                        value.shape
                    )
                    offset += value.numel()
            for key, value in visual.items():
                if key in staged:
                    continue
                if isinstance(value, torch.Tensor):
                    value = copy(value)
                elif isinstance(value, PackedPersons):
                    value = value.to_dense(self.device, copy=copy)
                elif isinstance(value, SceneTargetBatch) and value.values is not None:
                    value.values = copy(value.values, project_dtype=False)
                staged[key] = value
            ready = torch.cuda.Event()
            ready.record(self.stream)
        # Retain pinned source buffers through consumption of this batch.
        return StagedVisual(staged, ready, self.device, sources)
