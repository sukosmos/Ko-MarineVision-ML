# src/model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, ViTModel, AutoModel


def load_tokenizer():
    """Load or create a tokenizer for Korean text"""
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    return tokenizer


# ------------------------------
# 1. Object Embedding Module (DINO features + metadata)
# ------------------------------
class ObjectFeatureEncoder(nn.Module):
    def __init__(self, num_classes, num_subclasses, embed_dim):
        super().__init__()
        # DINO v2 for object visual features
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()  # Freeze DINO
        for param in self.dino.parameters():
            param.requires_grad = False
        
        dino_dim = 384  # dinov2_vits14 output dimension
        
        self.class_emb = nn.Embedding(num_classes, embed_dim)
        self.subclass_emb = nn.Embedding(num_subclasses, embed_dim)
        self.bbox_fc = nn.Linear(4, embed_dim)   # bbox: (w,h,x,y)
        
        # Project DINO features to embed_dim
        self.dino_proj = nn.Linear(dino_dim, embed_dim)
        
        self.proj = nn.Linear(embed_dim * 4, embed_dim)  # DINO + class + subclass + bbox

    def forward(self, obj_tensor, obj_images=None):
        """
        obj_tensor: (B, N_obj, 6) [class_id, subclass_id, w, h, x, y]
        obj_images: (B, N_obj, 3, H, W) cropped object images (optional)
        """
        B, N_obj = obj_tensor.shape[:2]
        
        cls = self.class_emb(obj_tensor[:, :, 0].long())
        sub = self.subclass_emb(obj_tensor[:, :, 1].long())
        bbox = self.bbox_fc(obj_tensor[:, :, 2:].float())
        
        # If object crops provided, use DINO features
        if obj_images is not None:
            with torch.no_grad():
                dino_feats = self.dino(obj_images.view(-1, 3, 224, 224))
            dino_feats = dino_feats.view(B, N_obj, -1)
            dino_feats = self.dino_proj(dino_feats)
        else:
            # Use zero features if no crops
            dino_feats = torch.zeros(B, N_obj, cls.size(-1), device=cls.device)
        
        x = torch.cat([dino_feats, cls, sub, bbox], dim=-1)
        return self.proj(x)  # (B, N_obj, embed_dim)


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
# 4. Full ExpansionNetv2 with DINO + ViT
# ------------------------------
class ExpansionNetV2_Multimodal(nn.Module):
    def __init__(self, vocab_size, num_classes, num_subclasses, embed_dim=768):
        super().__init__()
        
        # 1) ViT image encoder (global scene understanding)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        vit_dim = 768  # ViT-base hidden size
        
        # 2) object & env encoder (with DINO)
        self.obj_encoder = ObjectFeatureEncoder(num_classes, num_subclasses, embed_dim)
        self.env_encoder = EnvEncoder(embed_dim)

        # 3) expansion (visual feature richness 증가)
        self.expand = ExpansionLayer(embed_dim)

        # 4) transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # 5) vocab embedding & output head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def encode_image(self, img):
        """ViT로 전체 이미지 인코딩"""
        outputs = self.vit(pixel_values=img)
        # Use CLS token
        feat = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        feat = self.expand(feat)
        return feat.unsqueeze(1)  # (B, 1, D)

    def forward(self, img, obj_tensor, env_vec, tgt_ids, tgt_mask):
        B = img.size(0)

        # ViT global features
        img_feat = self.encode_image(img)
        
        # DINO object features
        obj_feat = self.obj_encoder(obj_tensor)  # (B, N_obj, D)
        
        # Environment features
        env_feat = self.env_encoder(env_vec)  # (B, 1, D)

        # Concatenate all encoder features
        encoder_memory = torch.cat([img_feat, obj_feat, env_feat], dim=1)

        # Decoder
        tgt_embed = self.embedding(tgt_ids)
        dec_out = self.decoder(tgt_embed, encoder_memory, tgt_mask=tgt_mask)

        return self.output(dec_out)
