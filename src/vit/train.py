# src/vit/train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from pathlib import Path

from build_data import EnvDataset
from model import ViTEnvClassifier

device = "cuda"

train_ds = EnvDataset("../../dataset/train")
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = ViTEnvClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(10):
    model.train()
    for pixels, labels in train_loader:
        pixels = pixels.to(device)
        labels = {k: v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device) for k, v in labels.items()}

        preds = model(pixels)

        loss = (
            F.cross_entropy(preds["season"], labels["season"]) +
            F.cross_entropy(preds["night"], labels["night"]) +
            F.cross_entropy(preds["weather"], labels["weather"]) +
            F.cross_entropy(preds["wave"], labels["wave"])
        ) / 4

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"epoch {epoch}: loss={loss.item():.4f}")

# 체크포인트 디렉토리 생성
checkpoint_path = Path("outputs/vit/checkpoints")
checkpoint_path.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), checkpoint_path / "vit_env.pth")
print(f"Model saved to {checkpoint_path / 'vit_env.pth'}")
