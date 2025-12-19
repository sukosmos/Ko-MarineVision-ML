# src/dino/infer.py
import torch
import numpy as np
from mmdet.apis import DetInferencer

CFG_FILE = "/data/CodeLLM/ML/src/dino/config.py"
CKPT_FILE = "/data/CodeLLM/ML/outputs/dino/epoch_12.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------
# Initialize inferencer (ONCE)
# ----------------------------------
inferencer = DetInferencer(
    model=CFG_FILE,
    weights=CKPT_FILE,
    device=DEVICE,
)

# class names (from config)
CLASS_NAMES = [
    '어선', '군함', '상선',
    '고정익 유인기', '회전익 유인기',
    '무인항공기', '새', '삐라', '오물폭탄'
]

CONF_THRESH = 0.3
MAX_OBJECTS = 10


def _xyxy_to_xywh_norm(bboxes, img_w, img_h):
    x1, y1, x2, y2 = bboxes.T
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    x = ((x1 + x2) / 2) / img_w
    y = ((y1 + y2) / 2) / img_h
    return np.stack([w, h, x, y], axis=1)


@torch.no_grad()
def run_dino(image_path: str) -> torch.Tensor:
    """
    return: Tensor (N, 6)
    [class_id, subclass_id, w, h, x, y]
    """

    # 🔑 text prompt (ALL classes)
    prompt = ". ".join(CLASS_NAMES)

    result = inferencer(
        image_path,
        texts=prompt,
        return_datasamples=True
    )

    data_sample = result["predictions"][0]

    pred = data_sample.pred_instances
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

    img_h, img_w = data_sample.ori_shape[:2]
    bbox_norm = _xyxy_to_xywh_norm(bboxes, img_w, img_h)

    obj_tensor = torch.from_numpy(
        np.concatenate([
            labels[:, None],      # class_id
            labels[:, None],      # subclass_id (same)
            bbox_norm
        ], axis=1)
    ).float()

    return obj_tensor
