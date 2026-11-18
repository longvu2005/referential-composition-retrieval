"""Read-only slot-query diagnostics on fixed training evidence."""

import torch
import torch.nn.functional as F

from rcr.common.data import inference_samples
from rcr.common.runtime import evaluating
from rcr.proposed.nn.binding import person_attention_query
from rcr.proposed.nn.encoders import encode_tokenized_text
from rcr.proposed.resume import capture_rng, restore_rng


@torch.no_grad()
def slot_query_diagnostics(model, text_encoder, cache, text_cache, samples, device):
    """Measure matched-slot attention on the first query and ordered Full Positive.

    Hooks observe the ordinary evidence path without changing its attention
    options. Only the separate eval passes request per-head attention weights.
    All global RNG streams and the modules' training modes are restored, including
    on failure; sampling and mining generators never enter this function.
    """
    if not model.slot_query:
        return {}
    rng = capture_rng(cuda=torch.cuda.is_initialized())
    try:
        with evaluating(model, text_encoder):
            sample = samples[0]
            query_indices = torch.tensor([cache.by_id[sample["query_image_id"]]])
            target_id = sorted(sample["positive_image_ids"])[0]
            target_indices = torch.tensor([cache.by_id[target_id]])
            query_features, target_features = cache.load_groups(
                query_indices, target_indices
            )
            scene, persons, boxes, mask = (x.to(device) for x in query_features)
            text = encode_tokenized_text(
                text_cache.batch(inference_samples([sample])), text_encoder, device
            )
            text.pop("change_global")
            query = model.encode_query(
                scene,
                persons,
                boxes,
                patch_hw=cache.patch_hw,
                query_person_mask=mask,
                **text,
            )
            binding = model.grounding.binding
            captured = {}

            def capture_binding(_module, args, kwargs):
                captured["persons"] = args[1]
                captured["slots"] = kwargs["person_slots"]

            def capture_attention(_module, args, kwargs):
                captured["query"], captured["scene"] = args[:2]
                captured["bias"] = kwargs["attn_mask"]

            with (
                binding.register_forward_pre_hook(capture_binding, with_kwargs=True),
                binding.person_attn.register_forward_pre_hook(
                    capture_attention, with_kwargs=True
                ),
            ):
                _, matching = model.score_context(
                    query,
                    *(x.to(device) for x in target_features[:3]),
                    cache.patch_hw,
                    target_features[3].to(device),
                )
            matched = ((matching.membership > 0) & matching.matched[:, None]).flatten(1)
            matched = (
                matched & query.mask[:, query.reference.shape[1] - matched.shape[1] :]
            )
            metrics = {
                "slot_gate": model.slot_gate.item(),
                "slot_attn_kl": 0.0,
                "slot_energy_ratio": 0.0,
            }
            if not matched.any():
                return metrics
            with torch.autocast(torch.device(device).type, enabled=False):
                persons = captured["persons"].float()
                slots = captured["slots"].float()
                scene = captured["scene"].float()
                substitute = person_attention_query(persons, persons, model.slot_gate)
                attentions = []
                for person_query in (captured["query"], substitute):
                    _, weights = binding.person_attn(
                        person_query,
                        scene,
                        scene,
                        attn_mask=captured["bias"],
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    attentions.append(weights)
                real, replaced = attentions
                tiny = torch.finfo(real.dtype).tiny
                kl = (
                    real * (real.clamp_min(tiny).log() - replaced.clamp_min(tiny).log())
                ).sum(-1)
                metrics["slot_attn_kl"] = kl.transpose(1, 2)[matched].mean().item()
                weight = binding.person_attn.in_proj_weight[: persons.shape[-1]]
                slot_energy = (model.slot_gate * F.linear(slots, weight)).norm(dim=-1)
                person_energy = F.linear(persons, weight).norm(dim=-1).clamp_min(1e-6)
                metrics["slot_energy_ratio"] = (
                    (slot_energy / person_energy)[matched].mean().item()
                )
            return metrics
    finally:
        restore_rng(rng)
