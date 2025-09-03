# models/convnextv2_tiny.py

import torch
import torch.nn as nn
import timm

class ConvNeXtV2TinyRegressor(nn.Module):
    """
    ConvNeXtV2-tiny backbone (single-channel input) + custom regression head.
    Suitable for 128x128 grayscale images as input, outputs a single value per image.
    """
    def __init__(self, pretrained=False, in_chans=1):
        super().__init__()
        # Create ConvNeXtV2-tiny backbone
        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae',
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,     # removes default classification head
            global_pool='avg'  # get [B, C] after backbone
        )
        # Regression head: outputs scalar per image (B,1)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        # x: [B, 1, 128, 128]  (batch of grayscale images)
        features = self.backbone(x)  # [B, C]
        out = self.head(features)    # [B, 1]
        return out.squeeze(-1)       # [B]

if __name__ == "__main__":
    # Demo/test: check input/output shapes
    net = ConvNeXtV2TinyRegressor(pretrained=False)
    dummy = torch.randn(8, 1, 128, 128)
    out = net(dummy)
    print(f"Input: {dummy.shape}  Output: {out.shape}")
