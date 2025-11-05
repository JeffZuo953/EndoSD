import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

from .lora import LoRALinear


class Expert(nn.Module):
    """An expert module, which is a simple MLP with LoRA applied to its linear layers."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: int = 1,
    ):
        super().__init__()
        self.fc1 = LoRALinear(in_features, hidden_features, r=lora_r, lora_alpha=lora_alpha, bias=bias)
        self.act = act_layer()
        self.fc2 = LoRALinear(hidden_features, out_features, r=lora_r, lora_alpha=lora_alpha, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TopKRouter(nn.Module):
    """Router module that selects the top-k experts for each token."""
    def __init__(self, in_features: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.linear = nn.Linear(in_features, num_experts)

    def forward(self, x: torch.Tensor):
        logits = self.linear(x)
        gates = F.softmax(logits, dim=-1)
        top_k_gates, top_k_indices = gates.topk(self.top_k, dim=-1, largest=True, sorted=False)
        return top_k_gates, top_k_indices


class MoELayer(nn.Module):
    """Mixture of Experts layer."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(in_features, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(
                in_features, hidden_features, out_features, act_layer, drop, bias, lora_r, lora_alpha
            ) for _ in range(num_experts)
        ])
        
        self.aux_loss_coef = 0.01 # Coefficient for the auxiliary load balancing loss

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, seq_len, in_features)
        batch_size, seq_len, in_features = x.shape
        x_flat = x.view(-1, in_features) # (batch_size * seq_len, in_features)

        gates, indices = self.router(x_flat) # gates: (N, top_k), indices: (N, top_k) where N = batch_size * seq_len
        
        # Auxiliary load balancing loss
        # This encourages the router to distribute tokens evenly across experts.
        gates_flat = gates.view(-1)
        expert_counts = x.new_zeros(self.num_experts)
        expert_counts.index_add_(0, indices.view(-1), torch.ones_like(gates_flat))
        total_tokens = expert_counts.sum().float()
        if total_tokens > 0:
            load_balance_loss = (expert_counts * F.softmax(expert_counts, dim=0)).sum() / total_tokens
        else:
            load_balance_loss = torch.tensor(0.0)
        self.aux_loss = self.aux_loss_coef * load_balance_loss

        # Dispatch tokens to experts
        final_output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            expert_mask = (indices == i).any(dim=-1)
            if expert_mask.any():
                tokens_for_expert = x_flat[expert_mask]
                gate_values = gates[expert_mask, (indices[expert_mask] == i).nonzero(as_tuple=True)[1]]
                
                expert_output = self.experts[i](tokens_for_expert)
                
                # Use non-inplace operation
                final_output = final_output.scatter_add(0, expert_mask.nonzero(as_tuple=True)[0].unsqueeze(-1).expand_as(tokens_for_expert), expert_output * gate_values.unsqueeze(-1))
        
        output = final_output
        
        return output.view(batch_size, seq_len, -1)