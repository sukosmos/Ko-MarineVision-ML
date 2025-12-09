# src/build_dataset.py
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CaptionDataset(Dataset):
    """
    ExpansionNet v2 학습용 Dataset
    - bbox format: (w, h, x, y)
    - caption: string
    - image: raw image
    """

    def __init__(self, jsonl_path, img_root, tokenizer, max_len=64):
        self.data = [json.loads(l) for l in open(jsonl_path, "r")]
        self.img_root = img_root
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.tf = T.Compose([
            T.Resize((224, 224)),   # ExpansionNet v2 default
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ----- Load Image -----
        img_path = f"{self.img_root}/{item['image']['filename']}"
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)

        # ----- Caption -----
        caption = item["caption"]
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # ---- BBoxes (w, h, x, y) ----
        # 그대로 반환 (ExpansionNetv2에서 필요시 feature로 사용)
        bboxes = []
        for obj in item.get("annotations", []):
            w, h, x, y = obj["bounding_box"]  # ⚠ bbox format: w h x y
            bboxes.append([w, h, x, y])

        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4))

        return {
            "image": image,
            "caption_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "bboxes": bboxes
        }
