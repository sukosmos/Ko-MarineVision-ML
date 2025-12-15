# src/test.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import evaluate

from build_data import MultiModalCaptionDataset
from model import ExpansionNetV2_Multimodal, load_tokenizer


def collate_fn(batch):
    """객체 개수가 다른 샘플들을 배치로 묶기"""
    images = torch.stack([item["image"] for item in batch])
    caption_ids = torch.stack([item["caption_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    env = torch.stack([item["env"] for item in batch])
    
    # 객체 개수가 다르므로 최대 개수만큼 패딩
    max_objs = max(item["objects"].size(0) for item in batch)
    
    padded_objs = []
    for item in batch:
        objs = item["objects"]
        if objs.size(0) < max_objs:
            padding = torch.zeros(max_objs - objs.size(0), 6)
            objs = torch.cat([objs, padding], dim=0)
        padded_objs.append(objs)
    
    objects = torch.stack(padded_objs)
    
    # 메타데이터도 포함
    img_paths = [item["img_path"] for item in batch]
    json_paths = [item["json_path"] for item in batch]
    
    return {
        "image": images,
        "caption_ids": caption_ids,
        "attention_mask": attention_mask,
        "env": env,
        "objects": objects,
        "img_path": img_paths,
        "json_path": json_paths
    }


def create_mask(size):
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask


# ---------------------------------------------------------
# Caption Autoregressive Generation
# ---------------------------------------------------------
def generate_caption(model, image, env, objects, tokenizer, max_len=64, device="cpu"):
    model.eval()
    
    # 시작 토큰
    ids = torch.tensor([[tokenizer.cls_token_id]], device=device)

    with torch.no_grad():
        for _ in range(max_len):
            tgt_mask = create_mask(ids.size(1)).to(device)

            # train.py와 동일한 형식: (B, C, H, W), (B, N_obj, 6), (B, 4), (B, seq_len)
            logits = model(image, objects, env, ids, tgt_mask)

            next_token = logits[0, -1].argmax(dim=-1).item()
            ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

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

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model
    model = ExpansionNetV2_Multimodal(
        vocab_size=tokenizer.vocab_size,
        num_classes=4,
        num_subclasses=42
    ).to(device)

    model.load_state_dict(torch.load("/data/CodeLLM/ML/outputs/expansionnet/expnetv2_multimodal.pth", map_location=device))

    predictions = []
    references = []
    results = []

    for batch in tqdm(loader, desc="Testing"):
        # train.py와 동일한 형식으로 데이터 로드
        img = batch["image"].to(device)      # (1, 3, 224, 224)
        env = batch["env"].to(device)        # (1, 4)
        objs = batch["objects"].to(device)   # (1, N_obj, 6)

        gt_caption = tokenizer.decode(batch["caption_ids"][0].tolist(), skip_special_tokens=True)

        pred_caption = generate_caption(
            model, img, env, objs, tokenizer, device=device
        )

        predictions.append(pred_caption)
        references.append([gt_caption])  # 리스트로 감싸기 (여러 참조 답변 지원)

        results.append({
            "image": batch["img_path"][0],
            "label": batch["json_path"][0],
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
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Total references: {len(references)}")
    if len(predictions) > 0:
        print(f"Sample prediction: {predictions[0][:50]}...")
        print(f"Sample reference: {references[0]}")
    
    metrics = {}
    
    try:
        print("\nLoading BLEU...")
        bleu = evaluate.load("bleu")
        metrics["BLEU"] = bleu.compute(predictions=predictions, references=references)
    except Exception as e:
        print(f"BLEU failed: {e}")
        import traceback
        traceback.print_exc()
        metrics["BLEU"] = None
    
    try:
        print("Loading ROUGE...")
        rouge = evaluate.load("rouge")
        metrics["ROUGE"] = rouge.compute(predictions=predictions, references=references)
    except Exception as e:
        print(f"ROUGE failed: {e}")
        metrics["ROUGE"] = None
    
    try:
        print("Loading CIDEr...")
        cider = evaluate.load("cider")
        metrics["CIDEr"] = cider.compute(predictions=predictions, references=references)
    except Exception as e:
        print(f"CIDEr failed: {e}")
        metrics["CIDEr"] = None
    
    try:
        print("Loading METEOR...")
        meteor = evaluate.load("meteor")
        metrics["METEOR"] = meteor.compute(predictions=predictions, references=references)
    except Exception as e:
        print(f"METEOR failed: {e}")
        metrics["METEOR"] = None

    json.dump(metrics, open("metrics.json", "w"), indent=2, ensure_ascii=False)
    print("Saved: metrics.json")
    print(metrics)


if __name__ == "__main__":
    test()
