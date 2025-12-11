# src/build_dataset.py
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class MultiModalCaptionDataset(Dataset):
    """
    Dataset folder structure:
    root_dir/
        image/VS_XXXX/xxx.jpg
        label/VL_XXXX/xxx.json
    """

    def __init__(self, root_dir, tokenizer, max_len=64):
        self.root_dir = Path(root_dir)

        self.image_root = self.root_dir / "image"
        self.label_root = self.root_dir / "label"

        self.tokenizer = tokenizer
        self.max_len = max_len

        # 모든 이미지 파일 검색
        self.image_paths = sorted(self.image_root.rglob("*.jpg"))

        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # -----------------------------------------------------
        # 1) JSON 파일 위치 계산 (VS → VL 변환)
        # -----------------------------------------------------
        relative = img_path.relative_to(self.image_root)   # ex: VS_EO_SU_DT/img001.jpg
        parts = list(relative.parts)

        # VS → VL 변환
        if parts[0].startswith("VS_"):
            parts[0] = "VL_" + parts[0][3:]  # VS_EO_SU_DT → VL_EO_SU_DT

        json_path = self.label_root / Path(*parts).with_suffix(".json")

        if not json_path.exists():
            raise FileNotFoundError(f"Label not found for {img_path} → {json_path}")

        # -----------------------------------------------------
        # 2) 이미지 로드
        # -----------------------------------------------------
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)

        # -----------------------------------------------------
        # 3) JSON 로드
        # -----------------------------------------------------
        with open(json_path, "r") as f:
            data = json.load(f)

        # caption
        encoded = self.tokenizer(
            data["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # -----------------------------------------------------
        # 4) Environment (season, night, weather, wave)
        # 0-based 인덱싱으로 변환하지 않고 그대로 사용 (EnvEncoder에서 Linear로 처리)
        # -----------------------------------------------------
        env = torch.tensor([
            data["env"]["season"] - 1,   # 1~2 -> 0~1
            data["env"]["night"] - 1,    # 1~2 -> 0~1
            data["env"]["weather"] - 1,  # 1~7 -> 0~6
            data["env"]["wave"] - 1      # 1~7 -> 0~6
        ], dtype=torch.float32)

        # -----------------------------------------------------
        # 5) Object list (bbox = w, h, x, y)
        # -----------------------------------------------------
        objs = []
        for ann in data["annotations"]:
            w, h, x, y = ann["bounding_box"]
            # PyTorch Embedding은 0-based 인덱싱을 사용하므로 1을 빼줌
            objs.append([
                ann["class"] - 1,      # 1~4 -> 0~3
                ann["sub_class"] - 1,  # 1~42 -> 0~41
                w, h, x, y
            ])

        objects = torch.tensor(objs, dtype=torch.float32)

        return {
            "image": image,
            "caption_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "env": env,
            "objects": objects,
            "img_path": str(img_path),
            "json_path": str(json_path)
        }
