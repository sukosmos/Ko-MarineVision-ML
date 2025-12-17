# src/train_earlystop.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ExpansionNetV2_Multimodal, load_tokenizer
from build_data import MultiModalCaptionDataset


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


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"TRAINING WITH EARLY STOPPING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Max Epochs: 5")
    print(f"Early Stop Patience: 2")
    print(f"{'='*60}\n")

    tokenizer = load_tokenizer()

    train_ds = MultiModalCaptionDataset(
        root_dir="../../dataset/train",
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = ExpansionNetV2_Multimodal(
        vocab_size=tokenizer.vocab_size,
        num_classes=4,
        num_subclasses=42
    ).to(device)
    
    # Verify model is on GPU
    if device == "cuda":
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )

    # Early stopping parameters
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 2
    max_epochs = 5

    for epoch in range(max_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        epoch_loss = 0.0
        num_batches = 0

        for batch in loop:
            img = batch["image"].to(device)
            ids = batch["caption_ids"].to(device)
            env = batch["env"].to(device)
            objs = batch["objects"].to(device)

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
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            from pathlib import Path
            output_dir = Path("../../outputs/expansionnet")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / "expnetv2_earlystop.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Best model saved: {model_path} (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (patience: {patience_counter}/{early_stop_patience})")
            
        if patience_counter >= early_stop_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best loss: {best_loss:.4f}")
            print(f"{'='*60}\n")
            break
    
    print(f"\nTraining completed!")
    print(f"Final best loss: {best_loss:.4f}")
    print(f"Model saved to: ../../outputs/expansionnet/expnetv2_earlystop.pth")


if __name__ == "__main__":
    train()
