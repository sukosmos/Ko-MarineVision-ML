# src/vit/model.py
import torch.nn as nn
from transformers import ViTModel

class ViTEnvClassifier(nn.Module):
    def __init__(self, h=768):
        super().__init__()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        self.season_head = nn.Linear(h, 2)     # 1=하절기, 2=동절기
        self.night_head  = nn.Linear(h, 2)     # 낮=1, 밤=2
        self.weather_head = nn.Linear(h, 7)    # 맑음~해무3단계
        self.wave_head = nn.Linear(h, 7)       # 1~7단계

    def forward(self, x):
        out = self.vit(pixel_values=x).pooler_output

        return {
            "season": self.season_head(out),
            "night": self.night_head(out),
            "weather": self.weather_head(out),
            "wave": self.wave_head(out)
        }
