# src/vit/build_data.py
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

class EnvDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_paths = list((self.root_dir / "image").rglob("*.jpg"))
        self.label_root = self.root_dir / "label"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 대응되는 JSON 파일 찾기
        relative = img_path.relative_to(self.root_dir / "image")
        json_path = self.label_root / relative.with_suffix(".json")

        img = Image.open(img_path).convert("RGB")

        with open(json_path, "r") as f:
            data = json.load(f)

        env = data["env"]

        pixel_values = processor(img, return_tensors="pt")["pixel_values"].squeeze(0)

        labels = {
            "season": env["season"],
            "night": env["night"],
            "weather": env["weather"],
            "wave": env["wave"],
        }

        return pixel_values, labels
