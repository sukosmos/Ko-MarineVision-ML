import json
import evaluate

# 결과 파일 로드
results = json.load(open("output_improved/test_results.json"))

predictions = [r["pred_caption"] for r in results]
references = [[r["gt_caption"]] for r in results]

# BLEU
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=predictions, references=references)

# METEOR
meteor = evaluate.load("meteor")
meteor_score = meteor.compute(predictions=predictions, references=references)

print("BLEU:", bleu_score)
print("METEOR:", meteor_score)
