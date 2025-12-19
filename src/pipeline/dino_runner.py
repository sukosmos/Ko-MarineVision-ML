# src/pipeline/dino_runner.py
import torch
import numpy as np
import os
from mmdet.apis import init_detector, inference_detector

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_FILE = os.path.join(BASE_DIR, "dino/config.py")
CKPT_FILE = os.path.join(os.path.dirname(BASE_DIR), "outputs/dino/epoch_12.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.3
MAX_OBJECTS = 10

_dino = None


def _load_dino():
    global _dino
    if _dino is None:
        _dino = init_detector(CFG_FILE, CKPT_FILE, device=DEVICE)
    return _dino


def _xyxy_to_xywh_norm(bboxes, w, h):
    x1, y1, x2, y2 = bboxes.T
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    return np.stack([bw, bh, cx, cy], axis=1)


@torch.no_grad()
def run_dino(image_path: str) -> torch.Tensor:
    model = _load_dino()
    result = inference_detector(model, image_path)

    pred = result.pred_instances
    if pred is None or len(pred) == 0:
        return torch.zeros(0, 6)

    bboxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()

    keep = scores >= CONF_THRESH
    bboxes = bboxes[keep]
    labels = labels[keep]

    if len(bboxes) == 0:
        return torch.zeros(0, 6)

    if len(bboxes) > MAX_OBJECTS:
        bboxes = bboxes[:MAX_OBJECTS]
        labels = labels[:MAX_OBJECTS]

    h, w = result.img_shape[:2]
    bbox_norm = _xyxy_to_xywh_norm(bboxes, w, h)

    obj = np.concatenate([
        labels[:, None],      # class_id
        labels[:, None],      # subclass_id (학습 때 동일)
        bbox_norm
    ], axis=1)

    return torch.from_numpy(obj).float()
