# src/vit/test.py
import torch
from build_data import EnvDataset
from model import ViTEnvClassifier

model = ViTEnvClassifier()
model.load_state_dict(torch.load("outputs/vit/checkpoints/vit_env.pth"))
model.eval()

test_ds = EnvDataset("../../dataset/test")

pixels, label = test_ds[0]

with torch.no_grad():
    out = model(pixels.unsqueeze(0))

pred = {k: v.argmax(-1).item() for k, v in out.items()}
print(pred)
