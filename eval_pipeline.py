import json
import evaluate
from pathlib import Path

TEST_RESULTS_PATH = Path("test_results_full.json")
TEST_LABEL_DIR = Path("/data/CodeLLM/ML/dataset/test/label")
OUTPUT_DIR = TEST_RESULTS_PATH.parent
METRICS_PATH = "metrics_full.jsonl"
MODEL_NAME = OUTPUT_DIR.name

def find_label_json(filename):
    stem = Path(filename).stem
    for sub in TEST_LABEL_DIR.iterdir():
        candidate = sub / f"{stem}.json"
        if candidate.exists():
            return candidate
    return None


raw = json.load(open(TEST_RESULTS_PATH))

valid = []
for r in raw:
    pred = r.get("predicted", "").strip()
    img = r.get("image", "").strip()

    if not pred or pred.lower().startswith("error"):
        continue

    gt_path = find_label_json(img)
    if gt_path is None:
        continue

    gt_data = json.load(open(gt_path))
    gt = gt_data.get("caption", "").strip()

    if not gt:
        continue

    valid.append({"pred": pred, "gt": gt})


if not valid:
    raise ValueError("No valid prediction samples — all predicted outputs are ERROR.")

print("[INFO] Valid samples:", len(valid))

preds = [v["pred"] for v in valid]
refs = [[v["gt"]] for v in valid]

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

b = bleu.compute(predictions=preds, references=refs)
m = meteor.compute(predictions=preds, references=refs)

metrics = {
    "model": MODEL_NAME,
    "num_samples": len(preds),
    "BLEU": float(b["bleu"]),
    "BLEU-1": float(b["precisions"][0]),
    "BLEU-2": float(b["precisions"][1]),
    "BLEU-3": float(b["precisions"][2]),
    "BLEU-4": float(b["precisions"][3]),
    "METEOR": float(m["meteor"]),
}

with open(METRICS_PATH, "a") as f:
    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

print(metrics)
