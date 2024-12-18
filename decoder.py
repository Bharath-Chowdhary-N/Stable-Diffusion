import torch
from torch import nn
from torch.nn import functional as F
import math
from attention import SelfAttention
class decoder():
    def __init__(self):
        pass

class VAE_resiudal_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.num_groups = 32
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.group_norm_1 = nn.GroupNorm(self.num_groups, self.in_channel)
        self.conv_1 = nn.Conv2d(self.in_channel, self.out_channel, padding=1, kernel_size=3)
        
        self.group_norm_2 = nn.GroupNorm(self.num_groups, self.in_channel)
        self.conv_2 = nn.Conv2d(self.in_channel, self.out_channel, padding=1, kernel_size=3)
        
        if self.in_channel == self.out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(self.in_channel,self.out_channel, kernel_size=1, padding=0)
    
    def forward(self, x):
        x_input = x

        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        x = x + self.residual_layer(x_input)

        return x 

class VAE_attention(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(self.num_groups, out_channel)
        self.self_attention = SelfAttention()


