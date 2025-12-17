import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np

from model import ExpansionNetV2_Multimodal, load_tokenizer
from build_data import MultiModalCaptionDataset


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Collate function (variable number of objects)
# ------------------------------------------------------------------
def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    caption_ids = torch.stack([item["caption_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    env = torch.stack([item["env"] for item in batch])

    max_objs = max(item["objects"].size(0) for item in batch)
    padded_objs = []

    for item in batch:
        objs = item["objects"]
        if objs.size(0) < max_objs:
            pad = torch.zeros(max_objs - objs.size(0), objs.size(1))
            objs = torch.cat([objs, pad], dim=0)
        padded_objs.append(objs)

    objects = torch.stack(padded_objs)

    return {
        "image": images,
        "caption_ids": caption_ids,
        "attention_mask": attention_mask,
        "env": env,
        "objects": objects,
    }


# ------------------------------------------------------------------
# Wrapper model (Dropout only – no LayerNorm on logits)
# ------------------------------------------------------------------
class ExpansionNetV2_WithDropout(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        num_subclasses,
        embed_dim=768,
        dropout=0.15,
    ):
        super().__init__()
        self.base_model = ExpansionNetV2_Multimodal(
            vocab_size=vocab_size,
            num_classes=num_classes,
            num_subclasses=num_subclasses,
            embed_dim=embed_dim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, img, obj_tensor, env_vec, tgt_ids, tgt_mask=None):
        logits = self.base_model(img, obj_tensor, env_vec, tgt_ids, tgt_mask)
        logits = self.dropout(logits)
        return logits


# ------------------------------------------------------------------
# Validation loop
# ------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, tokenizer, device):
    model.eval()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        img = batch["image"].to(device)
        ids = batch["caption_ids"].to(device)
        env = batch["env"].to(device)
        objs = batch["objects"].to(device)

        logits = model(img, objs, env, ids)

        loss = criterion(
            logits[:, :-1].reshape(-1, tokenizer.vocab_size),
            ids[:, 1:].reshape(-1),
        )

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
def train():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = load_tokenizer()

    # Dataset split
    full_ds = MultiModalCaptionDataset(
        root_dir="../../dataset/train",
        tokenizer=tokenizer,
    )

    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = ExpansionNetV2_WithDropout(
        vocab_size=tokenizer.vocab_size,
        num_classes=4,
        num_subclasses=42,
        dropout=0.15,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.1,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # Early stopping
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    max_epochs = 10

    save_dir = Path("../../outputs/expansionnet")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "expnetv2_final.pth"

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in loop:
            img = batch["image"].to(device)
            ids = batch["caption_ids"].to(device)
            env = batch["env"].to(device)
            objs = batch["objects"].to(device)

            logits = model(img, objs, env, ids)

            loss = criterion(
                logits[:, :-1].reshape(-1, tokenizer.vocab_size),
                ids[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, tokenizer, device)

        print(
            f"[Epoch {epoch}] "
            f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
        )

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✓ Best model saved ({best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

        print()

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    train()
