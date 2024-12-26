import torch
from torch import nn
from torch.nn import Functional as F

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        pass

class time_embedding(nn.Module):
    def __init__(self, num_embed):
        super().__init__()
        self.linear_layer_1 = nn.Linear(num_embed, 4*num_embed)
        self.linear_layer_2 = nn.Linear(4*num_embed, 4*num_embed)

    def forward(self, x:torch.tensor) -> torch.tensor:
        x = self.linear_layer_1(x)
        x = F.Silu(x)
        x=self.linear_layer_2(x)
        return x
