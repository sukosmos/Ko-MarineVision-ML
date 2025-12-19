# src/pipeline/vit_runner.py
import torch
from vit.model import ViTEnvClassifier

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_vit = None


def _load_vit():
    global _vit
    if _vit is None:
        _vit = ViTEnvClassifier().to(_DEVICE)
        _vit.load_state_dict(
            torch.load("../../outputs/vit/vit_env.pth", map_location=_DEVICE)
        )
        _vit.eval()
    return _vit


@torch.no_grad()
def run_vit(img_tensor: torch.Tensor) -> dict:
    vit = _load_vit()
    return vit(img_tensor)
