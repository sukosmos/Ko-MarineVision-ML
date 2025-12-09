# src/test.py
import torch
from build_dataset import CaptionDataset
from model import ExpansionNetV2, load_tokenizer
from tqdm import tqdm

def generate_caption(model, image, tokenizer, max_len=64, device="cpu"):
    model.eval()
    ids = torch.tensor([tokenizer.cls_token_id], device=device).unsqueeze(0)

    for _ in range(max_len):
        tgt_mask = torch.tril(torch.ones(ids.size(1), ids.size(1))).bool().to(device)

        logits = model(image.unsqueeze(0), ids, tgt_mask)
        next_token = logits[0, -1].argmax().item()

        ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

        if next_token == tokenizer.sep_token_id:
            break

    return tokenizer.decode(ids.squeeze().tolist())


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()
    test_ds = CaptionDataset(
        jsonl_path="../../dataset/test.jsonl",
        img_root="../../dataset/test/images",
        tokenizer=tokenizer,
        max_len=64
    )

    model = ExpansionNetV2(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load("expansionnetv2.pth", map_location=device))

    results = []
    for item in tqdm(test_ds):
        caption = generate_caption(model, item["image"].to(device), tokenizer, device=device)
        results.append({
            "filename": item["image"],
            "generated_caption": caption
        })

    import json
    json.dump(results, open("test_results.json", "w"), ensure_ascii=False, indent=2)
    print("테스트 결과 저장 완료: test_results.json")


if __name__ == "__main__":
    test()
