import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x

class LoRACompatibleLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            in_dim=linear_layer.in_features,
            out_dim=linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    Freezes all parameters except LoRA parameters.

    Args:
        model: The model to modify.
        bias: Strategy for handling bias parameters ('none', 'lora_only', 'all').
              'none': All bias parameters are frozen.
              'lora_only': Only bias parameters in LoRACompatibleLinear layers are trainable.
              'all': All bias parameters are trainable.
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

    if bias == 'none':
        return
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRACompatibleLinear) and m.bias is not None:
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
        for n, m in model.named_modules():
            if isinstance(m, LoRACompatibleLinear) and hasattr(m, 'bias') and m.bias is not None:
                bias_key = n + '.bias'
                if bias_key in my_state_dict:
                    result[bias_key] = my_state_dict[bias_key]
        return result
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    else:
        raise NotImplementedError(f"Bias type {bias} is not implemented.")