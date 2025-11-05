import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, # Set True if the layer is ViT's c_proj
        merge_weights: bool = True,
        bias: bool = True, # Add bias to the layer
        **kwargs
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out

        # Original linear layer
        # We store the original linear layer instance passed to us or create a new one
        # Note: This assumes the original nn.Linear layer's weights will be loaded externally
        # For compatibility, we keep the nn.Linear layer instance
        self.linear = nn.Linear(in_features, out_features, bias=bias, **kwargs)

        # LoRA matrices
        if r > 0:
            self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix is crucial
            self.linear.weight.requires_grad = False
            if bias and self.linear.bias is not None:
                self.linear.bias.requires_grad = False # Also freeze bias if it exists
            # Initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            # fan_in_fan_out handling removed for simplicity, assume standard init is fine
            # If specific init needed, it should be handled during model setup

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = self.linear(x)
            # Ensure requires_grad state is respected during forward pass if needed
            # (Though typically handled by optimizer and train/eval modes)
            lora_result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            result += lora_result
            return result
        else:
            # If r=0 or merged, just use the original linear layer
            return self.linear(x)

    def train(self, mode: bool = True):
        # Set train mode for dropout, etc.
        super().train(mode)
        # Manage merging/unmerging weights based on mode and config
        if self.merge_weights and self.merged:
            # Unmerge weights when switching to training mode
            if self.r > 0:
                self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

        # Ensure requires_grad status reflects training mode for LoRA params
        # Original weights remain frozen.
        if self.r > 0:
            self.lora_A.requires_grad = mode
            self.lora_B.requires_grad = mode
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False

    def eval(self):
        # Set eval mode for dropout, etc.
        super().eval()
        # Manage merging/unmerging weights based on mode and config
        if self.merge_weights and not self.merged:
            # Merge weights when switching to evaluation mode
            if self.r > 0:
                self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

        # Ensure requires_grad status reflects eval mode (all should be false)
        if self.r > 0:
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    Freezes all parameters except LoRA parameters.

    Args:
        model: The model to modify.
        bias: Strategy for handling bias parameters ('none', 'lora_only', 'all').
              'none': All bias parameters are frozen.
              'lora_only': Only bias parameters in LoRALinear layers are trainable.
              'all': All bias parameters are trainable. (Not typically used with LoRA)
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

    if bias == 'none':
        return # All biases are already frozen (requires_grad=False handled in LoRALinear init/train/eval)
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALinear) and m.bias is not None:
                 m.bias.requires_grad = True
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError(f"Bias type {bias} is not implemented.")

def lora_state_dict(model: nn.Module, bias: str = 'none') -> dict:
    """
    Returns a state dict containing only the trainable LoRA parameters (and potentially biases).
    """
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'lora_only':
        result = {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
        # Add bias parameters from LoRALinear layers
        for n, m in model.named_modules():
            if isinstance(m, LoRALinear) and 'bias' in my_state_dict and hasattr(m, 'bias') and m.bias is not None:
                bias_key = n + '.bias' if n else 'bias'
                if bias_key in my_state_dict:
                    result[bias_key] = my_state_dict[bias_key]
        return result
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    else:
        raise NotImplementedError(f"Bias type {bias} is not implemented.")