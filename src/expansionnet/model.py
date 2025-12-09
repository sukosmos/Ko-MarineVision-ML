# src/model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# 예시: ExpansionNet v2 임의 구조 (실제 구조에 맞게 수정 가능)
class ExpansionNetV2(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.enc_linear = nn.Linear(64 * 112 * 112, embed_dim)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8),
            num_layers=4
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, img, tgt_ids, tgt_mask):
        # image → feature
        img_feat = self.encoder_cnn(img)
        img_feat = self.enc_linear(img_feat).unsqueeze(0)  # (1, B, D)

        # caption → tokens → embedding
        tgt_embed = self.embedding(tgt_ids).transpose(0, 1)  # (T, B, D)

        # Transformer decoding
        dec_out = self.decoder(tgt_embed, memory=img_feat, tgt_mask=tgt_mask)

        logits = self.output(dec_out)  # (T, B, vocab)

        return logits.transpose(0, 1)  # (B, T, vocab)


def load_tokenizer():
    # BERT tokenizer or GPT tokenizer 등 원하는 것을 사용하면 됨
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    return tokenizer
