import torch
from torch import nn
from torch.nn import Functional as F
from attention import SelfAttention, CrossAttention

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
        super().__init__()
        self.conv_layer = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_layer(x)
        return x

class UNET_output(nn.Module):
    def __init__(self, in_channel, out_channel=4):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(32, in_channel)
    def forward(self, x):
        # x shape: (batch_size, 320, h/8, w/8)
        x = self.group_norm(x)
        x = F.Silu(x)
        x = self.conv_layer(x) # converted to (batch_size, 4, h/8, w/8)

        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([  
            # (Batch_size, 4, Height/8, Width/8) --> (Batch_size, 320, Height/8, Width/8)
            Switch_Sequential(nn.Conv2d(4,320,kernel_size=3, padding=1)),
            Switch_Sequential(UNET_residual_block(320,320), UNet_attention_block(8, 40)), # in attention block num_heads= 8 here, 8*40 = 320
            Switch_Sequential(UNET_residual_block(320,320), UNet_attention_block(8, 40)),

            # (Batch_size, 320, Height/8, Width/8) --> (Batch_size, 640, Height/16, Width/16)
            Switch_Sequential(nn.Conv2d(320,640,kernel_size=3, padding=1, stride=2)),
            Switch_Sequential(UNET_residual_block(640,640), UNet_attention_block(8, 80)),
            Switch_Sequential(UNET_residual_block(640,640), UNet_attention_block(8, 80)),

            # (Batch_size, 640, Height/16, Width/16) --> (Batch_size, 1280, Height/32, Width/32)
            Switch_Sequential(nn.Conv2d(640,1280,kernel_size=3, padding=1, stride=2)),
            Switch_Sequential(UNET_residual_block(1280,1280), UNet_attention_block(8, 80)),
            Switch_Sequential(UNET_residual_block(1280,1280), UNet_attention_block(8, 80)),

            # (Batch_size, 1280, Height/32, Width/32) --> (Batch_size, 1280, Height/64, Width/64)
            Switch_Sequential(nn.Conv2d(1280,1280,kernel_size=3, padding=1, stride=2)),
            Switch_Sequential(UNET_residual_block(1280,1280)),
            Switch_Sequential(UNET_residual_block(1280,1280))         
        ])

        self.bottle_neck = nn.ModuleList([
            Switch_Sequential(UNET_residual_block(1280,1280)),
            Switch_Sequential(UNet_attention_block(8,80)),
            Switch_Sequential(UNET_residual_block(1280,1280))
        ])

        self.decoders = nn.ModuleList([
            # (Batchsize, 2560, Height/64, Width/64)  --> (Batchsize, 1280, Height/64, Width/64)
            Switch_Sequential(UNET_residual_block(2560,1280)),
            Switch_Sequential(UNET_residual_block(2560,1280)),
              
            # (Batchsize, 1280, Height/64, Width/64)  --> (Batchsize, 1280, Height/32, Width/32) 
            Switch_Sequential(UNET_residual_block(2560,1280), Upsample(1280)),
            Switch_Sequential(UNET_residual_block(2560,1280), UNet_attention_block(8, 160)),
            Switch_Sequential(UNET_residual_block(2560,1280), UNet_attention_block(8, 160)),

            # (Batchsize, 1280, Height/32, Width/32)  --> (Batchsize, 640, Height/16, Width/16) 
            Switch_Sequential(UNET_residual_block(1920,640), UNet_attention_block(8, 80), Upsample(640)),
            Switch_Sequential(UNET_residual_block(1280,640), UNet_attention_block(8, 80)),
            Switch_Sequential(UNET_residual_block(1280,640), UNet_attention_block(8, 80)),

            # (Batchsize, 640, Height/16, Width/16)  --> (Batchsize, 320, Height/8, Width/8) 
            Switch_Sequential(UNET_residual_block(960,320), UNet_attention_block(8, 40), Upsample(320)),
            Switch_Sequential(UNET_residual_block(640,320), UNet_attention_block(8, 40)),
            Switch_Sequential(UNET_residual_block(640,320), UNet_attention_block(8, 40)),
        ])
    def forward(self, x, context, time):

        skip_connections = []

        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
        
        x = self.bottle_neck(x, context, time)

        for layers in self.decoders:
            x = torch.cat(x, skip_connections.pop(), dim=1)
            x = layers(x, context, time)
        
        return x

class UNet_attention_block(nn.Module):
    def __init__(self, n_heads, n_embed, d_context=768):
        super().__init__()
        channels = n_heads*n_embed  # here n_embed can litteraly translate to width of k', q' and v' in attention

        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_layer_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)

        self.layernorm_1    = nn.LayerNorm(channels)
        self.self_attention = SelfAttention(n_heads=n_heads, d_model=channels)

        self.layernorm_2    = nn.LayerNorm(channels)
        self.cross_attention = CrossAttention(n_heads=n_heads, d_model=channels, d_cross=d_context) #d_context is for text used for query and value vectors

        self.layernorm_3    = nn.LayerNorm(channels)
        
        self.linear_layer_1 = nn.Linear(channels,8*channels)
        self.linear_layer_2 = nn.Linear(4*channels, channels)

        self.conv_layer_2 = nn.Conv2d()
    
    def forward(self, x, context):
        
        residue_start = x

        x = self.groupnorm(x)
        x = self.conv_layer_1(x)

         
        n, c, h, w = x.shape
        
        x  = x.view(n,c,h*w)
        x  = x.transpose(-1,-2) #shape becomes (n, h*w, c)
        residue_mid = x

        x = self.layernorm_1(x)
        x = self.self_attention(x)
        x += residue_mid

        residue_mid = x

        x = self.layernorm_2(x)
        x = self.cross_attention(x, context)
        x += residue_mid

        residue_mid = x

        x = self.layernorm_3(x)
        x, interim = self.linear_layer_1(x).chunk(2,dim=-1) #interim will have  4*channels
        x = x*F.gelu(interim) # x will have 4 channels
        x = self.linear_layer_2(x) #will make it back to 1*channels 
        x += residue_mid

        x = x.transpose(-1,-2)
        x = x.view(n, c, h, w)

        x = self.conv_layer_2(x) + residue_start

        return x









   

class Diffusion(nn.Module):
    def __init_(self, op_layer_inp=320,  op_layer_op=4, num_embed=320):
        super().__init__()
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


class Switch_Sequential(nn.Sequential):
    def __init__(self):
        super().__init__() 
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNet_attention_block):
                x = layer(x, context)
            elif isinstance(layer, UNET_residual_block):
                x = layer(x, time)
            else:
                x = layer(x)
        return x



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







