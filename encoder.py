import torch
from torch import nn
from torch.nn import functional as F
import math
from decoder import VAE_residual_block, VAE_attention

class VAE_encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
        #(Batch size, channel, h, w)  --> # (Batch_size, 128, h, w)
        nn.Conv2d(3, 128, kernel_size=3, padding=1), # 3 = input channel, 128 = output_channel
        VAE_residual_block(128,128),
        VAE_residual_block(128,128),

        #at the end (Batch size, 128, h, w)  --> #(Batch_size, 256, h/2, w/2)
        nn.Conv2d(128,128,kernel_size=3, padding=0, stride=2),
        VAE_residual_block(128, 256),
        VAE_residual_block(256, 256),


        #(Batch_size, 256, h/2, w/2) --> #(Batch_size, 512, h/4 , w/4)
        nn.Conv2d(256,256,kernel_size=3, padding=0, stride=2),
        VAE_residual_block(256, 512),
        VAE_residual_block(512, 512),

        #(Batch_size, 512, h/4, w/4) --> #(Batch_size, 512, h/8 , w/8)
        nn.Conv2d(512,512,kernel_size=3, padding=0, stride=2),
        VAE_residual_block(512, 512),
        VAE_residual_block(512, 512),
        VAE_residual_block(512, 512),
        VAE_attention(512),
        VAE_residual_block(512,512),
        nn.GroupNorm(32,512),
        nn.SiLU(),

        #bottleneck of encoder, reducing features
        #(Batch_size, 512, h/8, w/8) --> (Batch_size, 8, h/8, w/8)   
        nn.Conv2d(512,8, kernel_size=3,padding=1) ,
        nn.Conv2d(8, 8, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor):
        pass
        


