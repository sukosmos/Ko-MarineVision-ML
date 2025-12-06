# src/vit/test.py
import torch
from build_data import EnvDataset
from model import ViTEnvClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load model
model = ViTEnvClassifier().to(device)
model.load_state_dict(torch.load("outputs/vit/checkpoints/vit_env.pth"))
model.eval()

# 2. Load test dataset
test_ds = EnvDataset("../../dataset/test")
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 각 task별 정답/예측 저장
y_true = {key: [] for key in ["season", "night", "weather", "wave"]}
y_pred = {key: [] for key in ["season", "night", "weather", "wave"]}

# 3. Inference loop
with torch.no_grad():
    for pixels, labels in test_loader:
        pixels = pixels.to(device)

        # 쉽게 GPU 이동
        labels = {k: v.to(device) for k, v in labels.items()}
        
        outputs = model(pixels)

        for k in outputs.keys():
            preds = outputs[k].argmax(dim=1)

            y_true[k].extend(labels[k].cpu().tolist())
            y_pred[k].extend(preds.cpu().tolist())

# 4. Accuracy 출력
for key in y_true.keys():
    acc = accuracy_score(y_true[key], y_pred[key])
    print(f"{key} accuracy: {acc:.4f}")

    print(classification_report(y_true[key], y_pred[key]))

# 5. Confusion Matrix Visualization
for key in y_true.keys():
    cm = confusion_matrix(y_true[key], y_pred[key])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix - {key}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"outputs/vit/{key}_confusion_matrix.png")
    plt.close()
