import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ONNXCompatibleAttention(nn.Module):
    """Standard scaled dot-product attention that's compatible with ONNX export"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make B, num_heads, N, head_dim
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def replace_xformers_attention(model):
    """Replace xformers attention modules with ONNX-compatible ones"""
    
    def replace_module(module, name, parent):
        """Recursively replace attention modules"""
        # Import the original attention class
        from depth_anything_v2.dinov2_layers.attention import Attention as XFormersAttention
        
        if isinstance(module, XFormersAttention):
            # Create ONNX-compatible attention with same parameters
            new_attn = ONNXCompatibleAttention(
                dim=module.qkv.in_features,
                num_heads=module.num_heads,
                qkv_bias=module.qkv.bias is not None,
                attn_drop=module.attn_drop.p if hasattr(module, 'attn_drop') else 0.0,
                proj_drop=module.proj_drop.p if hasattr(module, 'proj_drop') else 0.0,
            )
            
            # Copy weights
            with torch.no_grad():
                new_attn.qkv.weight.copy_(module.qkv.weight)
                if module.qkv.bias is not None:
                    new_attn.qkv.bias.copy_(module.qkv.bias)
                new_attn.proj.weight.copy_(module.proj.weight)
                if module.proj.bias is not None:
                    new_attn.proj.bias.copy_(module.proj.bias)
            
            setattr(parent, name, new_attn)
            return True
        
        # Recursively check child modules
        replaced = False
        for child_name, child_module in module.named_children():
            if replace_module(child_module, child_name, module):
                replaced = True
        
        return replaced
    
    # Start replacement from the model root
    for name, module in model.named_children():
        replace_module(module, name, model)
    
    return model
