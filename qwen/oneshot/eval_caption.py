import jsonlines
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm

PRED_PATH = "result/qwen_results.jsonl"
GT_PATH = "../../dataset/test/label.jsonl"   # 실제 test 라벨 경로에 맞게 수정


def load_jsonl(path):
    data = {}
    with jsonlines.open(path, "r") as f:
        for line in f:
            img = line["image"]
            data[img] = line
    return data


def main():
    pred = load_jsonl(PRED_PATH)
    gt = load_jsonl(GT_PATH)

    rouge = Rouge()
    bleu_scores = []
    rouge_l_scores = []

    smooth = SmoothingFunction().method4

    for img, p in tqdm(pred.items()):
        if img not in gt:
            continue

        pred_cap = p["caption_pred"]
        gt_cap = gt[img]["caption"]

        # ---- BLEU ----
        bleu = sentence_bleu(
            [gt_cap.split()],
            pred_cap.split(),
            smoothing_function=smooth
        )
        bleu_scores.append(bleu)

        # ---- ROUGE-L ----
        try:
            rouge_l = rouge.get_scores(pred_cap, gt_cap)[0]["rouge-l"]["f"]
        except:
            rouge_l = 0.0
        rouge_l_scores.append(rouge_l)

    print("====== Evaluation Result ======")
    print(f"BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"ROUGE-L: {sum(rouge_l_scores)/len(rouge_l_scores):.4f}")


if __name__ == "__main__":
    main()
