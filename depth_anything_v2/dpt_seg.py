import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2_lora import DINOv2_LoRA

class DepthAnythingV2_Seg(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        seg_head_type='BNHead', 
        num_classes=2,
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        lora_bias: str = 'none',
    ):
        super(DepthAnythingV2_Seg, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2_LoRA(
            model_name=encoder,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias
        )
        
        if seg_head_type == 'BNHead':
            from ..mutitask.seg_heads import LinearBNHead
            self.seg_head = LinearBNHead(in_channels=self.pretrained.embed_dim, num_classes=num_classes)
        elif seg_head_type == 'Mask2FormerLikeHead':
            from ..mutitask.seg_heads import Mask2FormerHead
            self.seg_head = Mask2FormerHead(in_channels=self.pretrained.embed_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown segmentation head type: {seg_head_type}")

    def forward(self, x):
        features = self.pretrained.get_intermediate_layers(x, [self.intermediate_layer_idx[self.encoder][-1]], return_class_token=False)[0]
        
        patch_h = x.shape[-2] // 14
        patch_w = x.shape[-1] // 14
        
        features = features.permute(0, 2, 1).reshape(features.shape[0], features.shape[-1], patch_h, patch_w)

        seg_output = self.seg_head(features)
        
        if seg_output.shape[-2:] != x.shape[-2:]:
            seg_output = F.interpolate(seg_output, size=x.shape[-2:], mode="bilinear", align_corners=True)
        
        return seg_output
