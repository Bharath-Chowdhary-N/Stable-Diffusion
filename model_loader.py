from CLIP import CLIP
from encoder import VAE_encoder
from decoder import VAE_decoder
from diffusion import Diffusion
import model_converter

def preload_models_based_on_std_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path,device)

    encoder = VAE_encoder().to(device)
    encoder.load_state_dict(state_dict=state_dict["encoder"], strict=True)


    decoder = VAE_decoder().to(device)
    decoder.load_state_dict()
