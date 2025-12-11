# src/train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ExpansionNetV2_Multimodal, load_tokenizer
from build_dataset import MultiModalCaptionDataset


def create_mask(size):
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()

    train_ds = MultiModalCaptionDataset(
        root_dir="../../dataset/train",
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    model = ExpansionNetV2_Multimodal(
        vocab_size=tokenizer.vocab_size,
        num_classes=50,
        num_subclasses=100
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

    torch.save(model.state_dict(), "expnetv2_multimodal.pth")
    print("Saved!")


if __name__ == "__main__":
    train()
