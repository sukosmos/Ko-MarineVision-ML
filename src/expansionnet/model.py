# src/model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer


def load_tokenizer():
    """Load or create a tokenizer for Korean text"""
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    return tokenizer


# ------------------------------
# 1. Object Embedding Module
# ------------------------------
class ObjectFeatureEncoder(nn.Module):
    def __init__(self, num_classes, num_subclasses, embed_dim):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, embed_dim)
        self.subclass_emb = nn.Embedding(num_subclasses, embed_dim)
        self.bbox_fc = nn.Linear(4, embed_dim)   # bbox: (w,h,x,y)

        self.proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, obj_tensor):
        """
        obj_tensor: (N_obj, 6)
            [class_id, subclass_id, w, h, x, y]
        """
        cls = self.class_emb(obj_tensor[:, 0].long())
        sub = self.subclass_emb(obj_tensor[:, 1].long())
        bbox = self.bbox_fc(obj_tensor[:, 2:].float())

        x = torch.cat([cls, sub, bbox], dim=-1)
        return self.proj(x)  # (N_obj, embed_dim)


# ------------------------------
# 2. Env Feature Encoder (season, night, weather, wave)
# ------------------------------
class EnvEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(4, embed_dim)  # 4 env values

    def forward(self, env_vec):
        return self.fc(env_vec.float()).unsqueeze(1)  # (B, 1, D)


# ------------------------------
# 3. Expansion Layers (핵심 구조)
# ------------------------------
class ExpansionLayer(nn.Module):
    def __init__(self, embed_dim, expand=4):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expand)
        self.fc2 = nn.Linear(embed_dim * expand, embed_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ------------------------------
# 4. Full ExpansionNetv2 (Encoder + Decoder)
# ------------------------------
class ExpansionNetV2_Multimodal(nn.Module):
    def __init__(self, vocab_size, num_classes, num_subclasses, embed_dim=512):
        super().__init__()

        # 1) image encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.img_proj = nn.Linear(128 * 56 * 56, embed_dim)

        # 2) object & env encoder
        self.obj_encoder = ObjectFeatureEncoder(num_classes, num_subclasses, embed_dim)
        self.env_encoder = EnvEncoder(embed_dim)

        # 3) expansion (visual feature richness 증가)
        self.expand = ExpansionLayer(embed_dim)

        # 4) transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # 5) vocab embedding & output head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def encode_image(self, img):
        feat = self.cnn(img)
        feat = feat.reshape(feat.size(0), -1)
        feat = self.img_proj(feat)
        feat = self.expand(feat)
        return feat.unsqueeze(1)  # (B, 1, D)

    def forward(self, img, obj_tensor, env_vec, tgt_ids, tgt_mask):
        B = img.size(0)

        img_feat = self.encode_image(img)
        obj_feat = self.obj_encoder(obj_tensor)  # (N_obj, D)
        obj_feat = obj_feat.unsqueeze(0).repeat(B, 1, 1)

        env_feat = self.env_encoder(env_vec)

        encoder_memory = torch.cat([img_feat, obj_feat, env_feat], dim=1)

        tgt_embed = self.embedding(tgt_ids)

        dec_out = self.decoder(tgt_embed, encoder_memory, tgt_mask=tgt_mask)

        return self.output(dec_out)
