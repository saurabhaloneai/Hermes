import torch
import torch.nn as nn


class VisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.n_embed, config.n_mlp, bias=True)
        self.linear_fc2 = nn.Linear(config.n_mlp, config.n_embed, bias=True)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))
