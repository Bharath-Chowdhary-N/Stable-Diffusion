#Purpose: # To connect various modules of the stable diffusion 
#Date: 1 Jan 2025
#By: BCN

import torch
import numpy as np
import tqdm as tqdm
from encoder import VAE_encoder


W = 512 #Height
H = 512 #Width
Latent_h = W // 8 # check the encoder, they take input and the output is h/8, w/8
Latent_w = H // 8
num_latent_channels = 4

def generate(prompt: str, neg_prompt: str, input_image=None, strength=0.9, do_cfg=True, cfg_scale=8, sampler_name="DDPM", 
             n_interference=50, models={}, seed=None, device=None, idle_device=None, tokenizer=None):
    """
    prompt      : text prompt
    neg_prompt  : this will gide the generator to not create an image based on this text
    input_image : for img to img
    strength    : attention to initial image
    do_cfg      : classifier free guidance   --> w * (out_cond - ouput_uncond) + out_uncond
    cfg_scale   : varies from 7 to 14
    sampler_name: ddpm 
    n_interference: no. of steps
    models      : store weights
    seed        : starting point for random generation
    device      : usually a GPU
    idle_device : usually a cpu (temp strage)
    tokenizer   : tokenizer model 
    
    
    Given all this info, this should generate an image
    """

    with torch.no_grad():
        if not (0<strength<=1):
            raise ValueError("Strength should be between 0 and 1")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x
        
        generator = torch.Generator()

        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # prompts to tokens
            conditional_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids #We have set sequence length=77 in CLIP, hence using the same.

            conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device=device)

            # (batch_size, seq_len) --> (batch_size, seq_len, dim)
            conditional_tokens = clip(conditional_tokens)


            unconditional_tokens = tokenizer.batch_encode_plus([neg_prompt], padding = "max_length", max_length=77).input_ids
            unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device=device)
            unconditional_tokens = clip(unconditional_tokens)

            # (batchsize, seq_len, dim) --> (2*batch_size, seq_len, dim)
            context = torch.cat([conditional_tokens,unconditional_tokens])
        
        else:

            conditional_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids #We have set sequence length=77 in CLIP, hence using the same.

            conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device=device)

            # (batch_size, seq_len) --> (batch_size, seq_len, dim)
            context = clip(conditional_tokens)
        
        to_idle(clip)

        if sampler_name == "DDPM":
            sampler = DDPM_sampler(generator)
            sampler.set_inference_steps(n_interference)
        else:
            raise ValueError("Sampler not set in pipleline")

        latent_shape = (1, num_latent_channels, Latent_h, Latent_w)

        if input_image:

            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((W,H))
            input_image_tensor = np.array(input_image_tensor)

            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = custom_rescale(input_image_tensor,(0, 255),(-1, 1)) #because the image has to be between -1 and 1
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch size, Height, Width, Channel) --> (Batch_size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute (0,3,1,2)

            encoder_noise = torch.randn(latent_shape, generator=generator)

            latents = encoder(x=input_image_tensor, noise=encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encoder)
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device) # here sampler is not needed
        


def custom_rescale(input, prev_lim, new_lim):
    
    prev_lim_min, prev_lim_max = prev_lim
    new_lim_min, new_lim_max = new_lim

    norm_input = (new_lim_max-new_lim_min)*(input-prev_lim_min)/(prev_lim_max-prev_lim_min) + new_lim_min

    if min(norm_input)!=-1:
        raise ValueError("Check min value")
    
    if max(norm_input)!=1:
        raise ValueError("Check max value")

    return norm_input

        



             
    

    return None