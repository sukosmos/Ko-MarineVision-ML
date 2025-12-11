# src/train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ExpansionNetV2_Multimodal, load_tokenizer
from build_data import MultiModalCaptionDataset


def create_mask(size):
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask


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
            # 패딩: [0, 0, 0, 0, 0, 0]으로 채움
            padding = torch.zeros(max_objs - objs.size(0), 6)
            objs = torch.cat([objs, padding], dim=0)
        padded_objs.append(objs)
    
    objects = torch.stack(padded_objs)
    
    return {
        "image": images,
        "caption_ids": caption_ids,
        "attention_mask": attention_mask,
        "env": env,
        "objects": objects
    }


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()

    train_ds = MultiModalCaptionDataset(
        root_dir="../../dataset/train",
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = ExpansionNetV2_Multimodal(
        vocab_size=tokenizer.vocab_size,
        num_classes=4,   # 실제 클래스: 1~4 -> 0~3
        num_subclasses=42  # 실제 서브클래스: 1~42 -> 0~41
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(10):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        epoch_loss = 0.0
        num_batches = 0

        for batch in loop:
            img = batch["image"].to(device)
            ids = batch["caption_ids"].to(device)
            env = batch["env"].to(device)
            objs = batch["objects"].to(device)

            tgt_mask = create_mask(ids.size(1)).to(device)

            logits = model(img, objs, env, ids, tgt_mask)

            loss = criterion(
                logits[:, :-1].reshape(-1, tokenizer.vocab_size),
                ids[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            loop.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / num_batches
        print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

    # 체크포인트 디렉토리 생성 및 모델 저장
    from pathlib import Path
    output_dir = Path("../../outputs/expansionnet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "expnetv2_multimodal.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
