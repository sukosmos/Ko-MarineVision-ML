# src/expansionnet/expnet_model.py
import torch
import torch.nn as nn

from expansionnet.model import ExpansionNetV2_Multimodal


class ExpansionNetV2_WithDropout(ExpansionNetV2_Multimodal):
    def __init__(
        self,
        vocab_size,
        num_classes,
        num_subclasses,
        dropout=0.15
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            num_subclasses=num_subclasses
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, img, obj_tensor, env_vec, tgt_ids, tgt_mask=None):
        out = super().forward(img, obj_tensor, env_vec, tgt_ids, tgt_mask)
        return self.dropout_layer(out)
