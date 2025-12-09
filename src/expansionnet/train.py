# src/train.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from build_dataset import CaptionDataset
from model import ExpansionNetV2, load_tokenizer


def create_mask(size):
    """Transformer decoder용 causal mask"""
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()

    train_ds = CaptionDataset(
        jsonl_path="../../dataset/train.jsonl",
        img_root="../../dataset/train/images",
        tokenizer=tokenizer,
        max_len=64
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = ExpansionNetV2(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in loop:
            img = batch["image"].to(device)
            ids = batch["caption_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            tgt_mask = create_mask(ids.size(1)).to(device)

            logits = model(img, ids, tgt_mask)

            loss = criterion(
                logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                ids[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "expansionnetv2.pth")
    print("모델 저장 완료: expansionnetv2.pth")


if __name__ == "__main__":
    train()
