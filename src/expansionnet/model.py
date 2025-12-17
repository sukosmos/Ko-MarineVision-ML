# src/model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, ViTModel, AutoModel


def load_tokenizer():
    """Load or create a tokenizer for Korean text"""
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base", local_files_only=True)
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

    def forward(self, obj_tensor, full_images=None):
        """
        obj_tensor: (B, N_obj, 6) [class_id, subclass_id, w, h, x, y]
        full_images: (B, 3, H, W) full scene images
        
        bbox format: [w, h, x, y] - normalized [0, 1]
        DINO는 bbox 영역을 crop해서 사용
        """
        B, N_obj = obj_tensor.shape[:2]
        
        cls = self.class_emb(obj_tensor[:, :, 0].long())
        sub = self.subclass_emb(obj_tensor[:, :, 1].long())
        bbox = self.bbox_fc(obj_tensor[:, :, 2:].float())
        
        # Extract DINO features from bbox regions
        if full_images is not None:
            # Crop objects from full image using bbox
            obj_crops = self._crop_objects(full_images, obj_tensor[:, :, 2:6])
            
            with torch.no_grad():
                # obj_crops: (B*N_obj, 3, 224, 224)
                dino_feats = self.dino(obj_crops)
            
            dino_feats = dino_feats.view(B, N_obj, -1)
            dino_feats = self.dino_proj(dino_feats)
        else:
            # Fallback: use metadata only
            dino_feats = torch.zeros(B, N_obj, self.dino_proj.out_features, device=cls.device)
        
        x = torch.cat([dino_feats, cls, sub, bbox], dim=-1)
        return self.proj(x)  # (B, N_obj, embed_dim)
    
    def _crop_objects(self, images, bboxes):
        """
        Crop object regions from images using bboxes
        images: (B, 3, H, W)
        bboxes: (B, N_obj, 4) [w, h, x, y] normalized
        Returns: (B*N_obj, 3, 224, 224)
        """
        B, N_obj = bboxes.shape[:2]
        _, _, H, W = images.shape
        
        crops = []
        for b in range(B):
            for n in range(N_obj):
                w_norm, h_norm, x_norm, y_norm = bboxes[b, n]
                
                # Skip padding objects (all zeros)
                if w_norm == 0 and h_norm == 0:
                    # Use center crop as fallback
                    crops.append(torch.nn.functional.interpolate(
                        images[b:b+1, :, H//4:3*H//4, W//4:3*W//4],
                        size=(224, 224), mode='bilinear', align_corners=False
                    ))
                    continue
                
                # Convert normalized coords to pixel coords
                x1 = int((x_norm - w_norm/2) * W)
                y1 = int((y_norm - h_norm/2) * H)
                x2 = int((x_norm + w_norm/2) * W)
                y2 = int((y_norm + h_norm/2) * H)
                
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                # Crop and resize to 224x224
                if x2 > x1 and y2 > y1:
                    crop = images[b:b+1, :, y1:y2, x1:x2]
                    crop = torch.nn.functional.interpolate(
                        crop, size=(224, 224), mode='bilinear', align_corners=False
                    )
                else:
                    # Fallback if bbox is invalid
                    crop = torch.nn.functional.interpolate(
                        images[b:b+1], size=(224, 224), mode='bilinear', align_corners=False
                    )
                
                crops.append(crop)
        
        return torch.cat(crops, dim=0)  # (B*N_obj, 3, 224, 224)


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
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", local_files_only=True)
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

    def forward(self, img, obj_tensor, env_vec, tgt_ids, tgt_mask=None):
        B = img.size(0)

        # ViT global features
        img_feat = self.encode_image(img)
        
        # DINO object features (with full image for cropping)
        obj_feat = self.obj_encoder(obj_tensor, full_images=img)  # (B, N_obj, D)
        
        # Environment features
        env_feat = self.env_encoder(env_vec)  # (B, 1, D)

        # Concatenate all encoder features
        encoder_memory = torch.cat([img_feat, obj_feat, env_feat], dim=1)

        # Decoder
        tgt_embed = self.embedding(tgt_ids)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            seq_len = tgt_ids.size(1)
            # Upper triangular mask with -inf for future positions
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt_ids.device) * float('-inf'), diagonal=1)
        
        dec_out = self.decoder(tgt_embed, encoder_memory, tgt_mask=tgt_mask)

        return self.output(dec_out)
