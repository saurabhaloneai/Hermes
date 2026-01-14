import torch.nn as nn
from ..layers import RMSNorm, SelfAttention, DenseMLP
from ..layers.moe import MoEMLP


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = MoEMLP(config) if config.n_experts else DenseMLP(config)

    def forward(self, x, cos, sin, layer_past_kv=None):
        
        attn_out, present_kv = self.self_attn(
            self.input_layernorm(x), cos, sin, past_kv=layer_past_kv
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present_kv

