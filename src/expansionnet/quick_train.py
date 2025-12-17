# Quick training script for testing
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from model import ExpansionNetV2_Multimodal, load_tokenizer
from build_data import MultiModalCaptionDataset
import sys


def create_mask(size):
    """
    Causal mask for autoregressive generation.
    Returns additive attention mask where 0.0 = attend, -inf = mask
    """
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
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


def quick_train(num_samples=100, epochs=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    tokenizer = load_tokenizer()

    # Small subset for quick testing
    full_ds = MultiModalCaptionDataset(
        root_dir="../../dataset/train",
        tokenizer=tokenizer
    )
    train_ds = Subset(full_ds, range(min(num_samples, len(full_ds))))
    
    print(f"Training on {len(train_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = ExpansionNetV2_Multimodal(
        vocab_size=tokenizer.vocab_size,
        num_classes=4,
        num_subclasses=42
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        epoch_loss = 0.0
        num_batches = 0

        for batch in loop:
            img = batch["image"].to(device)
            ids = batch["caption_ids"].to(device)
            env = batch["env"].to(device)
            objs = batch["objects"].to(device)

            # Check bbox values
            if num_batches == 0 and epoch == 0:
                print(f"\nFirst batch bbox check:")
                print(f"  Objects shape: {objs.shape}")
                print(f"  First object bbox: {objs[0, 0, 2:6].tolist()}")
                print(f"  Should be normalized [0,1]\n")

            logits = model(img, objs, env, ids, tgt_mask=None)

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

    # Save checkpoint
    from pathlib import Path
    output_dir = Path("../../outputs/expansionnet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "expnetv2_quick.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, tokenizer


if __name__ == "__main__":
    # Usage: python quick_train.py [num_samples] [epochs]
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print(f"Quick training: {num_samples} samples, {epochs} epochs\n")
    quick_train(num_samples, epochs)
