import json
import evaluate
from pathlib import Path

# ==================================================
# 사용자 설정 (여기만 수정)
# ==================================================
TEST_RESULTS_PATH = Path(
    "output_ori/test_results.json"
)
OUTPUT_DIR = TEST_RESULTS_PATH.parent
METRICS_PATH = OUTPUT_DIR / "metrics.jsonl"

MODEL_NAME = OUTPUT_DIR.name   # 폴더명 기반 (ori / earlystop / improved)
# ==================================================

# ------------------
# Load test results
# ------------------
results = json.load(open(TEST_RESULTS_PATH, "r"))

predictions = [r["pred_caption"] for r in results]
references = [[r["gt_caption"]] for r in results]

assert len(predictions) == len(references), "Prediction / Reference size mismatch"

# ------------------
# Load metrics
# ------------------
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

bleu_score = bleu.compute(
    predictions=predictions,
    references=references
)

meteor_score = meteor.compute(
    predictions=predictions,
    references=references
)

# ------------------
# Pack metrics
# ------------------
metrics = {
    "model": MODEL_NAME,
    "num_samples": len(predictions),
    "BLEU": float(bleu_score["bleu"]),
    "BLEU-1": float(bleu_score["precisions"][0]),
    "BLEU-2": float(bleu_score["precisions"][1]),
    "BLEU-3": float(bleu_score["precisions"][2]),
    "BLEU-4": float(bleu_score["precisions"][3]),
    "METEOR": float(meteor_score["meteor"]),
    "length_ratio": float(bleu_score["length_ratio"]),
}

# ------------------
# Save metrics.jsonl
# ------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(METRICS_PATH, "a") as f:
    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

# ------------------
# Print summary
# ------------------
print("\n[Evaluation Result]")
for k, v in metrics.items():
    print(f"{k}: {v}")
print(f"\nSaved to: {METRICS_PATH}")
