# src/test.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import evaluate

from build_dataset import MultiModalCaptionDataset
from model import ExpansionNetV2_Multimodal, load_tokenizer


# ---------------------------------------------------------
# Caption Autoregressive Generation
# ---------------------------------------------------------
def generate_caption(model, image, env, objects, tokenizer, max_len=64, device="cpu"):
    model.eval()

    # 시작 토큰 (CLS 활용)
    ids = torch.tensor([[tokenizer.cls_token_id]], device=device)

    for _ in range(max_len):
        tgt_mask = torch.tril(torch.ones(ids.size(1), ids.size(1))).bool().to(device)

        logits = model(
            image.unsqueeze(0),
            objects.unsqueeze(0),
            env.unsqueeze(0),
            ids,
            tgt_mask
        )

        next_token = logits[0, -1].argmax(dim=-1).item()

        ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

        # 종료 토큰
        if next_token == tokenizer.sep_token_id:
            break

    caption = tokenizer.decode(ids.squeeze().tolist(), skip_special_tokens=True)
    return caption


# ---------------------------------------------------------
# Main Testing Function
# ---------------------------------------------------------
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()

    # Dataset
    test_ds = MultiModalCaptionDataset(
        root_dir="../../dataset/test",
        tokenizer=tokenizer
    )

    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Model
    model = ExpansionNetV2_Multimodal(
        vocab_size=tokenizer.vocab_size,
        num_classes=50,
        num_subclasses=100
    ).to(device)

    model.load_state_dict(torch.load("expnetv2_multimodal.pth", map_location=device))

    predictions = []
    references = []
    results = []

    for batch in tqdm(loader, desc="Testing"):
        img = batch["image"].to(device)
        env = batch["env"].to(device)
        objects = batch["objects"][0].to(device)

        gt_caption = tokenizer.decode(batch["caption_ids"], skip_special_tokens=True)

        pred_caption = generate_caption(
            model, img, env, objects, tokenizer, device=device
        )

        predictions.append(pred_caption)
        references.append(gt_caption)

        results.append({
            "image": batch["img_path"],
            "label": batch["json_path"],
            "gt_caption": gt_caption,
            "pred_caption": pred_caption
        })

    # ---------------------------------------------------------
    # Save raw predictions
    # ---------------------------------------------------------
    json.dump(results, open("test_results.json", "w"), indent=2, ensure_ascii=False)
    print("Saved: test_results.json")

    # ---------------------------------------------------------
    # Evaluation Metrics
    # ---------------------------------------------------------
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    cider = evaluate.load("cider")
    meteor = evaluate.load("meteor")

    metrics = {
        "BLEU": bleu.compute(predictions=predictions, references=references),
        "ROUGE": rouge.compute(predictions=predictions, references=references),
        "CIDEr": cider.compute(predictions=predictions, references=references),
        "METEOR": meteor.compute(predictions=predictions, references=references),
    }

    json.dump(metrics, open("metrics.json", "w"), indent=2, ensure_ascii=False)
    print("Saved: metrics.json")
    print(metrics)


if __name__ == "__main__":
    test()
