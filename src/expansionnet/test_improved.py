# src/test.py
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
import sys

from build_data import MultiModalCaptionDataset
from model import ExpansionNetV2_Multimodal, load_tokenizer
from train_improved_2 import ExpansionNetV2_WithDropout

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
    """
    Causal mask for autoregressive generation.
    Returns additive attention mask where 0.0 = attend, -inf = mask
    """
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask


# ---------------------------------------------------------
# Caption Autoregressive Generation
# ---------------------------------------------------------
def generate_caption(model, image, env, objects, tokenizer, max_len=128, device="cpu", debug=False):
    model.eval()
    
    # 시작 토큰
    ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
    
    if debug:
        print(f"\n[DEBUG] Starting generation...")
        print(f"[DEBUG] CLS token ID: {tokenizer.cls_token_id}")
        print(f"[DEBUG] SEP token ID: {tokenizer.sep_token_id}")
        print(f"[DEBUG] PAD token ID: {tokenizer.pad_token_id}")
        print(f"[DEBUG] Initial ids shape: {ids.shape}")
        print(f"[DEBUG] Device: {device}")

    generated_tokens = []
    with torch.no_grad():
        for step in range(max_len):
            seq_len = ids.size(1)
            # Mask 없이 is_causal=True로 자동 처리되도록 함
            # tgt_mask = create_mask(seq_len).to(device)

            # Forward pass: (B=1, C, H, W), (B=1, N_obj, 6), (B=1, 4), (B=1, seq_len)
            logits = model(image, objects, env, ids, tgt_mask=None)

            # logits shape: (B=1, seq_len, vocab_size)
            # 마지막 위치의 로짓만 사용
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            next_token = next_token_logits.argmax(dim=-1).item()
            
            if debug and step < 10:
                top5_tokens = next_token_logits.topk(5)
                top5_indices = top5_tokens.indices.tolist()
                top5_values = top5_tokens.values.tolist()
                print(f"[DEBUG] Step {step}: next_token={next_token}")
                print(f"        Top5 indices: {top5_indices}")
                print(f"        Top5 values: {[f'{v:.2f}' for v in top5_values]}")
                # 토큰을 텍스트로 디코드
                try:
                    decoded = tokenizer.decode([next_token])
                    print(f"        Decoded: '{decoded}'")
                except:
                    print(f"        Decoded: (error)")
            
            generated_tokens.append(next_token)
            
            # 다음 토큰 추가
            ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

            # 종료 조건
            if next_token == tokenizer.sep_token_id or next_token == tokenizer.pad_token_id:
                if debug:
                    print(f"[DEBUG] Stopped at step {step}: token={next_token} ({'SEP' if next_token == tokenizer.sep_token_id else 'PAD'})")
                break

    # 디코딩
    generated_ids = ids.squeeze().tolist()
    if debug:
        print(f"[DEBUG] Generated token count: {len(generated_tokens)}")
        print(f"[DEBUG] Generated tokens: {generated_tokens[:20]}...")
        print(f"[DEBUG] Full ids (with CLS): {generated_ids[:20]}...")
    
    caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    if debug:
        print(f"[DEBUG] Final caption: '{caption}'")
        print(f"[DEBUG] Caption length: {len(caption)} chars")
    
    return caption


# ---------------------------------------------------------
# Main Testing Function
# ---------------------------------------------------------
def test(max_samples=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()

    # Dataset
    test_ds = MultiModalCaptionDataset(
        root_dir="../../dataset/test",
        tokenizer=tokenizer
    )
    
    # 샘플 수 제한 (빠른 테스트용)
    if max_samples is not None:
        total = len(test_ds)
        test_ds = Subset(test_ds, range(min(max_samples, total)))
        print(f"[INFO] Limited to {len(test_ds)} samples (from {total})")

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model
    model = ExpansionNetV2_WithDropout(
        vocab_size=tokenizer.vocab_size,
        num_classes=4,
        num_subclasses=42,
        dropout=0.15
    ).to(device)

    # Load checkpoint
    checkpoint_path = "/data/CodeLLM/ML/outputs/expansionnet/expnetv2_final.pth"
    print(f"Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    predictions = []
    references = []
    results = []

    print(f"\n{'='*60}")
    print(f"Testing on device: {device}")
    print(f"Dataset size: {len(test_ds)}")
    print(f"{'='*60}\n")

    for idx, batch in enumerate(tqdm(loader, desc="Testing")):
        # train.py와 동일한 형식으로 데이터 로드
        img = batch["image"].to(device)      # (1, 3, 224, 224)
        env = batch["env"].to(device)        # (1, 4)
        objs = batch["objects"].to(device)   # (1, N_obj, 6)

        gt_caption = tokenizer.decode(batch["caption_ids"][0].tolist(), skip_special_tokens=True)

        # 첫 샘플에서는 상세 디버그
        if idx == 0:
            print(f"\n[DEBUG] First sample - checking inputs:")
            print(f"  Image shape: {img.shape}")
            print(f"  Env shape: {env.shape}, values: {env[0].tolist()}")
            print(f"  Objects shape: {objs.shape}")
            print(f"  GT caption length: {len(gt_caption)} chars")
        
        pred_caption = generate_caption(
            model, img, env, objs, tokenizer, device=device, debug=(idx == 0)
        )

        predictions.append(pred_caption)
        references.append([gt_caption])  # 리스트로 감싸기 (여러 참조 답변 지원)

        # 처음 3개 샘플 출력
        if idx < 3:
            print(f"\n--- Sample {idx+1} ---")
            print(f"GT:   {gt_caption[:100]}...")
            print(f"PRED: '{pred_caption[:100] if pred_caption else '(empty)'}...'")
            if idx == 0 and not pred_caption:
                print(f"[WARNING] First prediction is empty! Model may not be generating tokens.")


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
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Total references: {len(references)}")
    
    # 빈 예측 통계
    empty_count = sum(1 for p in predictions if not p or len(p.strip()) == 0)
    print(f"Empty predictions: {empty_count} ({empty_count/len(predictions)*100:.1f}%)")
    print(f"Non-empty predictions: {len(predictions)-empty_count} ({(len(predictions)-empty_count)/len(predictions)*100:.1f}%)")
    
    if len(predictions) > 0:
        print(f"\nFirst prediction: '{predictions[0][:80]}{'...' if len(predictions[0]) > 80 else ''}")
        print(f"First reference: '{references[0][0][:80]}{'...' if len(references[0][0]) > 80 else ''}")
    
    metrics = {
        "total_samples": len(predictions),
        "empty_predictions": empty_count,
        "non_empty_predictions": len(predictions) - empty_count,
        "empty_ratio": empty_count / len(predictions) if len(predictions) > 0 else 0
    }
    
    print("\n[INFO] Skipping online metrics (BLEU, ROUGE, etc.) for faster testing.")
    print("[INFO] To compute full metrics, install evaluate library offline or use local implementation.")

    json.dump(metrics, open("metrics.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: metrics.json")
    print(f"Saved: test_results.json")


if __name__ == "__main__":
    # Usage: python test.py [max_samples]
    # Example: python test.py 5  (test only 5 samples)
    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if max_samples:
        print(f"[INFO] Running test with {max_samples} samples")
    test(max_samples=max_samples)
