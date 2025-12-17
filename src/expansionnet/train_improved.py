# src/train_improved.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import ExpansionNetV2_Multimodal, load_tokenizer
from build_data import MultiModalCaptionDataset
import random
import numpy as np


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """객체 개수가 다른 샘플들을 배치로 묶기"""
    images = torch.stack([item["image"] for item in batch])
    caption_ids = torch.stack([item["caption_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    env = torch.stack([item["env"] for item in batch])
    
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


class ImprovedExpansionNetV2(nn.Module):
    """Dropout과 LayerNorm을 추가한 개선 버전"""
    def __init__(self, vocab_size, num_classes, num_subclasses, embed_dim=768, dropout=0.1):
        super().__init__()
        
        # 기존 모델 로드
        self.base_model = ExpansionNetV2_Multimodal(vocab_size, num_classes, num_subclasses, embed_dim)
        
        # Regularization 추가
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(vocab_size)
    
    def forward(self, img, obj_tensor, env_vec, tgt_ids, tgt_mask=None):
        logits = self.base_model(img, obj_tensor, env_vec, tgt_ids, tgt_mask)
        
        # Dropout + LayerNorm 적용
        logits = self.dropout(logits)
        logits = self.layer_norm(logits)
        
        return logits


def validate(model, val_loader, criterion, tokenizer, device):
    """Validation loop"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            ids = batch["caption_ids"].to(device)
            env = batch["env"].to(device)
            objs = batch["objects"].to(device)

            logits = model(img, objs, env, ids, tgt_mask=None)

            loss = criterion(
                logits[:, :-1].reshape(-1, tokenizer.vocab_size),
                ids[:, 1:].reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def train():
    set_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"IMPROVED TRAINING (Anti-Overfitting)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nRegularization Techniques:")
    print(f"  • Dropout: 0.15")
    print(f"  • Label Smoothing: 0.1")
    print(f"  • Weight Decay: 0.01")
    print(f"  • Gradient Clipping: 1.0")
    print(f"  • Train/Val Split: 90/10")
    print(f"  • Early Stopping: Val loss based, patience=3")
    print(f"{'='*70}\n")

    tokenizer = load_tokenizer()

    # Train/Val split
    full_ds = MultiModalCaptionDataset(
        root_dir="../../dataset/train",
        tokenizer=tokenizer
    )
    
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    print(f"Dataset split:")
    print(f"  Training:   {len(train_ds)} samples")
    print(f"  Validation: {len(val_ds)} samples\n")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Improved model with dropout
    model = ImprovedExpansionNetV2(
        vocab_size=tokenizer.vocab_size,
        num_classes=4,
        num_subclasses=42,
        dropout=0.15  # Dropout 추가
    ).to(device)
    
    if device == "cuda":
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.01  # L2 regularization
    )
    
    # Label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.1  # Label smoothing
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    max_epochs = 10

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        train_loss = 0.0
        num_batches = 0

        for batch in train_loop:
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / num_batches
        
        # Validation
        avg_val_loss = validate(model, val_loader, criterion, tokenizer, device)
        
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"            Learning Rate: {current_lr:.6f}")
        
        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            from pathlib import Path
            output_dir = Path("../../outputs/expansionnet")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / "expnetv2_improved.pth"
            torch.save(model.state_dict(), model_path)
            print(f"            ✓ Best model saved (Val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"            ✗ No improvement (patience: {patience_counter}/{patience})")
            
        if patience_counter >= patience:
            print(f"\n{'='*70}")
            print(f"Early stopping at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*70}\n")
            break
        
        print()
    
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: ../../outputs/expansionnet/expnetv2_improved.pth")


if __name__ == "__main__":
    train()
