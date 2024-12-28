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
        x=self.linear_layer_2(x) #input x: [1, num_embded] output x: [1, 4*num_embed]
        return x

class Upsample(nn.Module):
    def __init__(self, channels):
        self.conv_layer = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_layer(x)
        return x

class Diffusion(nn.Module):
    def __init_(self, op_layer_inp=320,  op_layer_op=4, num_embed=320):
        
        self.time_embedding = time_embedding(num_embed)

        self.Unet_instance = UNET()

        self.Unet_output = UNET_output(op_layer_inp,op_layer_op)

    def forward(self, latent_vec, time_embed, text_prompt):
        
        time_embed = self.time_embedding(time_embed) #shape: [1, 320] --> [1,1280]

        #from encoder (Batchsize, 4, h/8, w/8)  --> (Batchsize, 320, h/8, w/8)
        unet_output = self.Unet_instance(latent_vec, time_embed, text_prompt)
        
        #back again (Batchsize, 320, h/8, w/8)  --> (Batchsize, 4, h/8, w/8)
        final_ouput = self.Unet_output(unet_output)

        return final_ouput 
        

class UNET_residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, num_time_embed):
        super().__init__()
        #Build same as VAE residual block, except we are adding a time layer here
        self.num_groups = 32
        self.group_norm_1 = nn.GroupNorm(self.num_groups, in_channel)
        self.conv_layer_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)

        self.time_layer = nn.Linear(num_time_embed, out_channel)

        self.group_norm_2 = nn.GroupNorm(self.num_groups, out_channel)
        self.conv_layer_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0)
    
    def forward(self, x, time_embed):
        residue = x

        x = self.group_norm_1(x)
        x = F.Silu(x)
        x = self.conv_layer_1(x)

        time_embed = F.Silu(time_embed) 
        time_embed = self.time_layer(time_embed) #this will be converted into (1, time_embed_shape) into (1, out_channel)   
        x += time_embed.unsqueeze(-1).unsqueeze(-1)

        x = self.group_norm_2(x)
        x = F.Silu(x)
        x = self.conv_layer_2(x)
         
        return x + self.residual_layer(residue)








