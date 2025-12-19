# src/pipeline/adapters.py
import torch


def adapt_dino_objects(objects: torch.Tensor, max_objects=10) -> torch.Tensor:
    if objects.numel() == 0:
        return torch.zeros(0, 6)

    objs = objects.clone()

    # subclass = class (학습 당시 동일)
    objs[:, 1] = objs[:, 0]

    # bbox clamp
    objs[:, 2:] = objs[:, 2:].clamp(0.0, 1.0)

    if objs.size(0) > max_objects:
        objs = objs[:max_objects]

    return objs


def adapt_vit_env(vit_logits: dict) -> torch.Tensor:
    return torch.tensor([
        vit_logits["season"].argmax(-1).item(),
        vit_logits["night"].argmax(-1).item(),
        vit_logits["weather"].argmax(-1).item(),
        vit_logits["wave"].argmax(-1).item(),
    ])
