# models/convnextv2_tiny.py

import torch
import torch.nn as nn
import timm

class ConvNeXtV2TinyRegressor(nn.Module):
    """
    ConvNeXtV2-tiny backbone (single-channel input) + improved regression head.
    Suitable for ~128x128 grayscale images; outputs one scalar per image.
    """
    def __init__(self, pretrained=False, in_chans=1):
        super().__init__()
        # Backbone
        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae',
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,     # remove default classification head
            global_pool='avg'  # [B, C]
        )
        # (7) Stronger head for stability
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # x: [B, in_chans, H, W]
        features = self.backbone(x)   # [B, C]
        out = self.head(features)     # [B, 1]
        return out.squeeze(-1)        # [B]

if __name__ == "__main__":
    net = ConvNeXtV2TinyRegressor(pretrained=False)
    dummy = torch.randn(8, 1, 128, 128)
    out = net(dummy)
    print(f"Input: {dummy.shape}  Output: {out.shape}")
